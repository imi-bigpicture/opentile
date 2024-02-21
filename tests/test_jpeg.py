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

import pytest
from opentile.geometry import Size
from opentile.jpeg import Jpeg
from tifffile import TiffFile, TiffPage

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
        (stripe_width, stripe_height, _, _) = jpeg._turbo_jpeg.decode_header(
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
        assert md5(frame).hexdigest() == "fdde19f6d10994c5b866b43027ff94ed"

    def test_code_short(self):
        # Arrange
        jpeg = Jpeg()

        # Act

        #
        assert jpeg.code_short(6) == bytes([0x00, 0x06])
