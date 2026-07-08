#    Copyright 2021-2024 SECTRA AB
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
from hashlib import md5
from pathlib import Path

import numpy as np
import pytest
from imagecodecs import JPEG8, jpeg8_decode, jpeg8_encode
from tifffile import TiffFile, TiffPage

from opentile.geometry import Size
from opentile.jpeg import Jpeg, JpegProcess

test_data_dir = os.environ.get("OPENTILE_TESTDIR", "tests/testdata")
slide_folder = Path(test_data_dir).joinpath("slides")
ndpi_file_path = slide_folder.joinpath("ndpi/CMU-1/CMU-1.ndpi")
svs_file_path = slide_folder.joinpath("svs/CMU-1/CMU-1.svs")


@pytest.fixture()
def ndpi_tiff():
    with TiffFile(ndpi_file_path) as tiff:
        yield tiff


@pytest.fixture()
def ndpi_level(ndpi_tiff: TiffFile):
    yield ndpi_tiff.series[0].levels[0].pages[0]


@pytest.fixture()
def ndpi_header(ndpi_level: TiffPage):
    yield ndpi_level.jpegheader


@pytest.fixture()
def svs_tiff():
    with TiffFile(svs_file_path) as tiff:
        yield tiff


@pytest.fixture()
def svs_overview(svs_tiff: TiffFile):
    yield svs_tiff.series[3].pages[0]


@pytest.mark.unittest
class TestJpeg:
    @staticmethod
    def read_frame(tiff: TiffFile, level: TiffPage, index: int) -> bytes:
        offset = level.dataoffsets[index]
        length = level.databytecounts[index]
        tiff.filehandle.seek(offset)
        return tiff.filehandle.read(length)

    @pytest.mark.parametrize(
        ["marker", "expected_bytes"],
        [
            (Jpeg.start_of_frame(), bytes([0xFF, 0xC0])),
            (Jpeg.end_of_image(), bytes([0xFF, 0xD9])),
            (Jpeg.restart_mark(0), bytes([0xD0])),
            (Jpeg.restart_mark(7), bytes([0xD7])),
            (Jpeg.restart_mark(9), bytes([0xD1])),
        ],
    )
    def test_tags(self, marker: bytes, expected_bytes: bytes):
        # Arrange

        # Act

        # Assert
        assert marker == expected_bytes

    def test_find_tag(self, ndpi_header: bytes):
        # Arrange

        # Act
        index, length = Jpeg._find_tag(ndpi_header, Jpeg.start_of_frame())

        # Assert
        assert index == 621
        assert length == 17

    def test_update_header(self, ndpi_header: bytes):
        # Arrange
        target_size = Size(512, 200)
        jpeg = Jpeg()

        # Act
        updated_header = Jpeg.manipulate_header(ndpi_header, target_size)

        # Assert
        stripe_width, stripe_height, _, _ = jpeg._turbo_jpeg.decode_header(
            updated_header
        )
        assert target_size == Size(stripe_width, stripe_height)

    def test_concatenate_fragments(
        self, ndpi_tiff: TiffFile, ndpi_level: TiffPage, ndpi_header: bytes
    ):
        # Arrange
        jpeg = Jpeg()

        # Act
        frame = jpeg.concatenate_fragments(
            (self.read_frame(ndpi_tiff, ndpi_level, index) for index in range(10)),
            ndpi_header,
        )

        # Assert
        assert md5(frame).hexdigest() == "ea40e78b081c42a6aabf8da81f976f11"

    def test_concatenate_scans(self, svs_tiff: TiffFile, svs_overview: TiffPage):
        # Arrange
        jpeg = Jpeg()

        # Act
        frame = jpeg.concatenate_scans(
            (
                self.read_frame(svs_tiff, svs_overview, index)
                for index in range(len(svs_overview.databytecounts))
            ),
            svs_overview.jpegtables,
            True,
        )

        # Assert
        assert md5(frame).hexdigest() == "7528e846bcd0374d4500924395aebfc0"

        # Quantization tables must precede the start of frame that references
        # them, and the Adobe APP14 marker must be present (rgb color space fix).
        sof = frame.find(Jpeg.start_of_frame())
        assert 0 < frame.find(bytes([0xFF, 0xDB])) < sof  # DQT before SOF
        assert frame.find(bytes([0xFF, 0xEE])) < sof  # APP14 before SOF
        # Components renamed to ASCII 'R', 'G', 'B' as a marker-independent RGB
        # signal (survives DICOM APPn stripping).
        assert frame[sof + 9] == 3  # three components
        assert [frame[sof + 10 + c * 3] for c in range(3)] == [0x52, 0x47, 0x42]
        sos = frame.find(Jpeg.start_of_scan())
        assert [frame[sos + 5 + c * 2] for c in range(3)] == [0x52, 0x47, 0x42]

    @pytest.mark.parametrize(
        "width",
        [
            64,  # multiple of the mcu width (8): restart interval is unchanged
            70,  # not a multiple of the mcu width (8): minimal reproducer
            574,  # not a multiple of the mcu width (8): width from issue #155
        ],
    )
    def test_concatenate_scans_decodes_to_expected_pixels(self, width: int):
        """Concatenated scans must decode to the same pixels as the individual
        scans stacked vertically. Regression test for a too small restart
        interval being written when the scan width is not a multiple of the mcu
        size, corrupting e.g. odd-width svs thumbnails."""
        # Arrange
        jpeg = Jpeg()
        strip_height = 16  # rows per strip, matching striped svs images
        strip_count = 3  # >= 2 so restart markers are inserted between strips
        # Fill each pixel from its coordinates (a ramp that wraps at 256).
        height = strip_height * strip_count
        image = np.fromfunction(
            lambda r, c, ch: (c * 3 + r * 5 + ch * 2) % 256, (height, width, 3)
        ).astype(np.uint8)
        # Encode each strip as no subsampling jpeg
        scans = [
            jpeg8_encode(
                image[index * strip_height : (index + 1) * strip_height],
                subsampling="444",
                optimize=False,
            )
            for index in range(strip_count)
        ]
        expected = np.concatenate([jpeg8_decode(scan) for scan in scans], axis=0)

        # Act
        frame = jpeg.concatenate_scans(iter(scans), None, False)

        # Assert
        assert np.array_equal(jpeg8_decode(frame), expected)

    def test_code_short(self):
        # Arrange
        jpeg = Jpeg()

        # Act

        #
        assert jpeg.code_short(6) == bytes([0x00, 0x06])


def _jpeg_header(
    sof_marker: int,
    component_ids: list[int],
    sampling_factors: list[tuple[int, int]],
    precision: int = 8,
    app14_transform: int | None = None,
    sos_predictor: int = 0,
) -> bytes:
    """Craft a minimal JPEG header (SOI + optional APP14 + SOF + SOS) for the
    marker parser. No valid scan data; the parser stops at SOS."""
    frame = b"\xff\xd8"  # SOI
    if app14_transform is not None:
        payload = b"Adobe" + bytes(6) + bytes([app14_transform])  # 12 bytes
        frame += b"\xff\xee" + (len(payload) + 2).to_bytes(2, "big") + payload
    sof = bytes([precision]) + (64).to_bytes(2, "big") + (64).to_bytes(2, "big")
    sof += bytes([len(component_ids)])
    for component_id, (horizontal, vertical) in zip(component_ids, sampling_factors):
        sof += bytes([component_id, (horizontal << 4) | vertical, 0])
    frame += bytes([0xFF, sof_marker]) + (len(sof) + 2).to_bytes(2, "big") + sof
    sos = bytes([len(component_ids)])
    for component_id in component_ids:
        sos += bytes([component_id, 0])
    sos += bytes([sos_predictor, 0, 0])  # Ss (predictor), Se, Ah/Al
    frame += b"\xff\xda" + (len(sos) + 2).to_bytes(2, "big") + sos
    return frame


@pytest.mark.unittest
class TestJpegInfo:
    @pytest.mark.parametrize(
        ["subsampling", "expected"],
        [((1, 1), (1, 1)), ((2, 1), (2, 1)), ((2, 2), (2, 2))],
    )
    def test_info_subsampling(
        self, subsampling: tuple[int, int], expected: tuple[int, int]
    ):
        # Arrange
        image = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
        frame = jpeg8_encode(image, colorspace=JPEG8.CS.YCbCr, subsampling=subsampling)

        # Act
        info = Jpeg.info(frame)

        # Assert
        assert info.subsampling == expected

    def test_info_baseline_8bit(self):
        # Arrange
        image = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
        frame = jpeg8_encode(image, colorspace=JPEG8.CS.YCbCr)

        # Act
        info = Jpeg.info(frame)

        # Assert
        assert info.process == JpegProcess.BASELINE
        assert info.bit_depth == 8
        assert info.components == 3

    def test_info_extended_12bit(self):
        # Arrange — 12-bit encodes as extended sequential (SOF1)
        image = (np.random.rand(64, 64, 3) * 4095).astype(np.uint16)
        frame = jpeg8_encode(image, bitspersample=12)

        # Act
        info = Jpeg.info(frame)

        # Assert
        assert info.process == JpegProcess.EXTENDED
        assert info.bit_depth == 12

    @pytest.mark.parametrize(
        ["sof_marker", "expected"],
        [
            (0xC0, JpegProcess.BASELINE),
            (0xC1, JpegProcess.EXTENDED),
            (0xC2, JpegProcess.PROGRESSIVE),
            (0xC3, JpegProcess.LOSSLESS),
            (0xC5, JpegProcess.OTHER),  # differential sequential, unmapped
        ],
    )
    def test_info_process_from_sof_marker(
        self, sof_marker: int, expected: JpegProcess
    ):
        # Arrange
        frame = _jpeg_header(sof_marker, [1, 2, 3], [(1, 1), (1, 1), (1, 1)])

        # Act & Assert
        assert Jpeg.info(frame).process == expected

    def test_info_grayscale_has_no_subsampling(self):
        # Arrange
        image = (np.random.rand(64, 64) * 255).astype(np.uint8)
        frame = jpeg8_encode(image, colorspace=JPEG8.CS.GRAYSCALE)

        # Act
        info = Jpeg.info(frame)

        # Assert
        assert info.components == 1
        assert info.subsampling is None

    @pytest.mark.parametrize(
        ["component_ids", "app14_transform", "expected"],
        [
            ([0x52, 0x47, 0x42], None, True),  # R/G/B component ids
            ([1, 2, 3], 0, True),  # Adobe APP14 transform 0
            ([0x52, 0x47, 0x42], 1, True),  # ids signal RGB despite transform 1
            ([1, 2, 3], 1, False),  # APP14 transform 1 (YCbCr)
            ([1, 2, 3], 2, False),  # APP14 transform 2 (YCCK)
            ([1, 2, 3], None, False),  # no RGB signal
        ],
    )
    def test_rgb_signalled(
        self, component_ids: list[int], app14_transform: int | None, expected: bool
    ):
        # Arrange
        frame = _jpeg_header(
            0xC0,
            component_ids,
            [(1, 1)] * len(component_ids),
            app14_transform=app14_transform,
        )

        # Act & Assert
        assert Jpeg.info(frame).rgb_signalled is expected

    def test_lossless_process_and_predictor(self):
        # Arrange — SOF3 lossless with first-order predictor (selection value 1)
        frame = _jpeg_header(0xC3, [1, 2, 3], [(1, 1), (1, 1), (1, 1)], sos_predictor=1)

        # Act
        info = Jpeg.info(frame)

        # Assert
        assert info.process == JpegProcess.LOSSLESS
        assert info.lossless_predictor == 1

    def test_missing_sof_raises(self):
        # Arrange — SOI then SOS, no SOF
        frame = b"\xff\xd8\xff\xda\x00\x08\x01\x01\x00\x00\x00"

        # Act & Assert
        with pytest.raises(ValueError):
            Jpeg.info(frame)
