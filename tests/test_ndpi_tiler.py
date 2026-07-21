#    Copyright 2021-2023 SECTRA AB
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

from collections.abc import Sequence
from datetime import datetime
from hashlib import md5

import pytest
from tifffile import COMPRESSION, PHOTOMETRIC

from opentile.formats import NdpiTiler
from opentile.formats.ndpi.ndpi_image import (
    NdpiJpegXrImage,
    NdpiOneFrameImage,
    NdpiStripedImage,
)
from opentile.formats.ndpi.ndpi_metadata import NdpiMetadata
from opentile.formats.ndpi.ndpi_tile import NdpiFrameJob, NdpiTile
from opentile.geometry import Point, Size, SizeMm
from opentile.tiff_image import BaseTiffImage

from .filepaths import ndpi_file_path, ndpi_jpegxr_file_path, ndpi_z_file_path


@pytest.fixture()
def tile_size():
    yield Size(512, 512)


@pytest.fixture()
def tiler(tile_size: Size):
    try:
        with NdpiTiler(ndpi_file_path, tile_size.width) as tiler:
            yield tiler
    except FileNotFoundError:
        pytest.skip("Ndpi test file not found, skipping")


@pytest.fixture()
def z_tiler(tile_size: Size):
    try:
        with NdpiTiler(ndpi_z_file_path, tile_size.width) as tiler:
            yield tiler
    except FileNotFoundError:
        pytest.skip("Ndpi test file not found, skipping")


@pytest.fixture()
def jpegxr_tiler(tile_size: Size):
    try:
        with NdpiTiler(ndpi_jpegxr_file_path, tile_size.width) as tiler:
            yield tiler
    except FileNotFoundError:
        pytest.skip("Ndpi JPEG XR test file not found, skipping")


@pytest.fixture()
def level(tiler: NdpiTiler):
    yield tiler.get_level(0)


@pytest.fixture()
def one_frame_level(tiler: NdpiTiler):
    yield tiler.get_level(3)


@pytest.mark.unittest
class TestNdpiTiler:
    @pytest.mark.parametrize(
        ["point", "expected_index"], [(Point(50, 0), 50), (Point(20, 20), 520)]
    )
    def test_get_stripe_position_to_index(
        self, level: NdpiStripedImage, point: Point, expected_index: int
    ):
        # Arrange

        # Act
        index = level._get_stripe_position_to_index(point)

        # Assert
        assert index == expected_index

    def test_file_handle_read(self, level: NdpiStripedImage):
        # Arrange
        offset = level._page.dataoffsets[50]
        length = level._page.databytecounts[50]

        # Act
        data = level._file.read(offset, length)

        # Assert
        assert md5(data).hexdigest() == "2a903c6e05bd10f10d856eecceb591f0"

    def test_level_read(self, level: NdpiStripedImage):
        # Arrange

        # Act
        data = level._read_frame(50)
        assert md5(data).hexdigest() == "2a903c6e05bd10f10d856eecceb591f0"

    def test_read_frame(self, level: NdpiStripedImage):
        # Arrange
        index = level._get_stripe_position_to_index(Point(50, 0))

        # Act
        stripe = level._read_frame(index)

        # Assert
        assert md5(stripe).hexdigest() == "2a903c6e05bd10f10d856eecceb591f0"

    def test_get_frame(self, level: NdpiStripedImage):
        # Arrange
        position = Point(10, 10)
        indices = level._stripe_indices(position, level.frame_size)
        stripes = dict(zip(indices, level._read_frames(indices)))

        # Act
        image = level._jpeg.concatenate_fragments(
            (stripes[index] for index in indices), level._header(level.frame_size)
        )

        # Assert
        assert md5(image).hexdigest() == "aeffd12997ca6c232d0ef35aaa35f6b7"

    @pytest.mark.parametrize(
        ["tile_point", "hash"],
        [
            ((0, 0), "30c69cab610e5b3db4beac63806d6513"),
            ((20, 20), "fec8116d05485df513f4f41e13eaa994"),
        ],
    )
    def test_get_tile(
        self, level: NdpiStripedImage, tile_point: tuple[int, int], hash: str
    ):
        # Arrange

        # Act
        tile = level.get_tile(tile_point)

        # Assert
        assert md5(tile).hexdigest() == hash

    @pytest.mark.parametrize(
        ["tile_point", "hash"],
        [
            ((0, 0), "170165575a3e9cc564e7ac18d5520f1b"),
            ((1, 1), "4a48dc3f72584170497e793d44a2372d"),
        ],
    )
    def test_get_tile_one_frame_level(
        self, one_frame_level: NdpiOneFrameImage, tile_point: tuple[int, int], hash: str
    ):
        # Arrange

        # Act
        tile = one_frame_level.get_tile(tile_point)

        # Assert
        assert md5(tile).hexdigest() == hash

    @pytest.mark.parametrize(
        ["tile_points", "hashes"],
        [
            (
                [(0, 0), (20, 20)],
                [
                    "30c69cab610e5b3db4beac63806d6513",
                    "fec8116d05485df513f4f41e13eaa994",
                ],
            ),
        ],
    )
    def test_get_tiles(
        self,
        level: NdpiStripedImage,
        tile_points: Sequence[tuple[int, int]],
        hashes: Sequence[str],
    ):
        # Arrange

        # Act
        tiles = level.get_tiles(tile_points)

        # Assert
        for tile, hash in zip(tiles, hashes):
            assert md5(tile).hexdigest() == hash

    def test_create_tiles(self, level: NdpiStripedImage, tile_size: Size):
        # Arrange
        frame_job = NdpiFrameJob(
            [NdpiTile(Point(x, 0), tile_size, level.frame_size) for x in range(4)]
        )
        tiles_single = [level.get_tile((x, 0)) for x in range(4)]

        # Act
        ((job, frame),) = list(level._read_job_frames([frame_job]))
        tiles = list(level._crop_to_tiles(job, frame).values())

        # Assert
        assert tiles == tiles_single

    def test_crop_to_tiles(self, level: NdpiStripedImage, tile_size: Size):
        # Arrange
        frame_job = NdpiFrameJob(
            [NdpiTile(Point(x, 0), tile_size, level.frame_size) for x in range(4)]
        )
        tiles_single = {Point(x, 0): level.get_tile((x, 0)) for x in range(4)}
        ((job, frame),) = list(level._read_job_frames([frame_job]))

        # Act
        tiles = level._crop_to_tiles(job, frame)

        assert tiles == tiles_single

    def test_map_tile_to_frame(self, level: NdpiStripedImage, tile_size: Size):
        # Arrange

        # Act
        tile = NdpiTile(Point(5, 5), tile_size, level.frame_size)

        assert (tile.left, tile.top, tile.width, tile.height) == (512, 0, 512, 512)

    @pytest.mark.parametrize(
        ["tile_point", "expected_frame_position"],
        [
            (Point(3, 0), Point(0, 0)),
            (Point(7, 0), Point(4, 0)),
            (Point(5, 2), Point(4, 2)),
        ],
    )
    def test_frame_position_tile(
        self,
        level: NdpiStripedImage,
        tile_point: Point,
        tile_size: Size,
        expected_frame_position: Point,
    ):
        # Arrange
        tile = NdpiTile(tile_point, tile_size, level.frame_size)

        # Act
        frame_position = tile.frame_position

        # Assert
        assert frame_position == expected_frame_position

    def test_frame_job(self, level: NdpiStripedImage, tile_size: Size):
        # Arrange
        tile0 = NdpiTile(Point(3, 0), tile_size, level.frame_size)

        # Act
        frame_job = NdpiFrameJob([tile0])

        # Assert
        assert frame_job.position == Point(0, 0)

    def test_frame_job_append(self, level: NdpiStripedImage, tile_size: Size):
        # Arrange
        tile0 = NdpiTile(Point(3, 0), tile_size, level.frame_size)
        frame_job = NdpiFrameJob([tile0])
        tile1 = NdpiTile(Point(2, 0), tile_size, level.frame_size)

        # Act
        frame_job.append(tile1)

        # Assert
        assert frame_job.tiles == [tile0, tile1]

    def test_frame_jobs_raises_when_not_on_row(
        self, level: NdpiStripedImage, tile_size: Size
    ):
        # Arrange
        tile0 = NdpiTile(Point(3, 0), tile_size, level.frame_size)
        frame_job = NdpiFrameJob([tile0])
        tile2 = NdpiTile(Point(2, 1), tile_size, level.frame_size)

        # Act & Assert
        with pytest.raises(ValueError):
            frame_job.append(tile2)

    def test_sort_into_frame_jobs(self, level: NdpiStripedImage, tile_size: Size):
        # Arrange
        expected = [
            NdpiFrameJob(
                [
                    NdpiTile(
                        Point(index_x_0 + index_x_1, index_y),
                        tile_size,
                        level.frame_size,
                    )
                    for index_x_1 in range(4)
                ]
            )
            for index_x_0 in range(0, 8, 4)
            for index_y in range(2)
        ]

        # Act
        frame_jobs = level._sort_into_frame_jobs(
            [(index_x, index_y) for index_x in range(8) for index_y in range(2)]
        )

        assert frame_jobs == expected

    def test_stripe_size(self, level: NdpiStripedImage):
        # Arrange

        # Act
        stripe_size = level.stripe_size

        # Assert
        assert stripe_size == Size(2048, 8)

    def test_striped_size(self, level: NdpiStripedImage):
        # Arrange

        # Act
        striped_size = level.striped_size

        # Assert
        assert striped_size == Size(25, 4768)

    def test_header(self, level: NdpiStripedImage):
        # Arrange

        # Act
        header = level._page.jpegheader

        # Assert
        assert isinstance(header, bytes)
        assert md5(header).hexdigest() == "579211c6b9fedca17d94b95840f4b985"

    def test_get_file_frame_size(self, level: NdpiStripedImage):
        # Arrange

        # Act
        file_frame_size = level._get_file_frame_size()

        # Assert
        assert file_frame_size == Size(2048, 8)

    def test_get_frame_size(self, level: NdpiStripedImage):
        # Arrange

        # Act
        frame_size = level.frame_size

        # Assert
        assert frame_size == Size(2048, 512)

    @pytest.mark.parametrize(
        ["point", "expected_size"],
        [(Point(0, 0), Size(2048, 512)), (Point(99, 74), Size(2048, 256))],
    )
    def test_get_frame_size_for_tile(
        self, level: NdpiStripedImage, point: Point, expected_size: Size
    ):
        # Arrange

        # Act
        frame_size = level._get_frame_size_for_tile(point)

        # Assert
        assert frame_size == expected_size

    def test_tiled_size(self, level: NdpiStripedImage):
        # Arrange

        # Act
        tiled_size = level.tiled_size

        # Assert
        assert tiled_size == Size(100, 75)

    def test_get_smallest_stripe_width(self, tiler: NdpiTiler):
        # Arrange

        # Act
        smallest_width = tiler._get_smallest_stripe_width()

        # Assert
        assert smallest_width == 128

    @pytest.mark.parametrize(
        ["requested_size", "stripe_width", "expected_adjusted_size"],
        [
            (512, 512, Size(512, 512)),
            (512, 56, Size(448, 448)),
            (512, 248, Size(496, 496)),
        ],
    )
    def test_adjust_tile_size(
        self,
        tiler: NdpiTiler,
        requested_size: int,
        stripe_width: int,
        expected_adjusted_size: Size,
    ):
        # Arrange

        # Act
        adjusted_size = tiler._adjust_tile_size(requested_size, stripe_width)

        # Assert
        assert adjusted_size == expected_adjusted_size

    def test_photometric_interpretation(self, level: BaseTiffImage):
        # Arrange

        # Act
        photometric_interpretation = level.photometric_interpretation

        # Assert
        assert photometric_interpretation == PHOTOMETRIC.YCBCR

    def test_subsampling(self, level: BaseTiffImage):
        # Arrange

        # Act
        subsampling = level.subsampling

        # Assert
        assert subsampling == (1, 1)

    def test_sumples_per_pixel(self, level: BaseTiffImage):
        # Arrange

        # Act
        samples_per_pixel = level.samples_per_pixel

        # Assert
        assert samples_per_pixel == 3

    def test_metadata_magnification(self, tiler: NdpiTiler):
        # Arrange

        # Act
        magnification = tiler.metadata.magnification

        # Assert
        assert magnification == 20.0

    def test_metadata_scanner_manufacturer(self, tiler: NdpiTiler):
        # Arrange

        # Act
        scanner_manufacturer = tiler.metadata.scanner_manufacturer

        # Assert
        assert scanner_manufacturer == "Hamamatsu"

    def test_metadata_scanner_model(self, tiler: NdpiTiler):
        # Arrange

        # Act
        scanner_model = tiler.metadata.scanner_model

        # Assert
        assert scanner_model == "NanoZoomer"

    def test_metadata_scanner_software_versions(self, tiler: NdpiTiler):
        # Arrange

        # Act
        scanner_software_versions = tiler.metadata.scanner_software_versions

        # Assert
        assert scanner_software_versions == ["NDP.scan"]

    def test_metadata_scanner_serial_number(self, tiler: NdpiTiler):
        # Arrange

        # Act
        scanner_serial_number = tiler.metadata.scanner_serial_number

        # Assert
        assert scanner_serial_number is None

    def test_metadata_acquisition_datetime(self, tiler: NdpiTiler):
        # Arrange

        # Act
        acquisition_datetime = tiler.metadata.acquisition_datetime

        # Assert
        assert acquisition_datetime == datetime(2009, 12, 31, 9, 11, 46)

    def test_label(self, tiler: NdpiTiler):
        # Arrange

        # Act
        label = tiler.get_label().get_tile((0, 0))

        # Assert
        assert md5(label).hexdigest() == "ff78b53c3d483adb04b3e24e413cc96f"

    def test_labels_list(self, tiler: NdpiTiler):
        # Arrange

        # Act
        labels = tiler.labels

        # Assert
        assert len(labels) == 1
        assert (
            md5(labels[0].get_tile((0, 0))).hexdigest()
            == "ff78b53c3d483adb04b3e24e413cc96f"
        )

    def test_overview(self, tiler: NdpiTiler):
        # Arrange

        # Act
        overview = tiler.get_overview().get_tile((0, 0))

        # Assert
        assert md5(overview).hexdigest() == "3c35de47f6137ba2c118ec4703c393c2"

    def test_compressed_size(self, level: BaseTiffImage):
        # Arrange

        # Act
        compressed_size = level.compressed_size

        # Assert
        assert compressed_size == 262256667

    @pytest.mark.parametrize(
        ["level", "expected_size"],
        [
            (0, SizeMm(0.00045641259698767686, 0.0004550625711035267)),
            (1, SizeMm(0.0018258170531312763, 0.0018204988166757691)),
            (2, SizeMm(0.007304601899196494, 0.007283321194464677)),
        ],
    )
    def test_pixel_spacing(self, tiler: NdpiTiler, level: int, expected_size: SizeMm):
        # Arrange
        base_level = tiler.get_level(level)

        # Act
        base_pixel_spacing = base_level.pixel_spacing

        # Assert
        assert base_pixel_spacing == expected_size

    def test_focal_planes(self, tiler: NdpiTiler):
        # Arrange

        # Act
        focal_planes = set([level.focal_plane for level in tiler.levels])

        # Assert
        assert focal_planes == {0}

    def test_focal_planes_z_tiler(self, z_tiler: NdpiTiler):
        # Arrange
        expected = {0.0, 1.2, 2.4, 3.6, -3.6, -2.4, -1.2}

        # Act
        focal_planes = set([level.focal_plane for level in z_tiler.levels])

        # Assert
        assert focal_planes == expected


@pytest.mark.unittest
class TestParseNdpiComments:
    def test_global_entries_hoisted_to_top_level(self):
        # Arrange
        comments = "Created=2016/01/01\r\nProduct=C13210\r\nNDP.S/N=000003\r\n"

        # Act
        parsed = NdpiMetadata._parse_comments(comments)

        # Assert
        assert parsed == {
            "Created": "2016/01/01",
            "Product": "C13210",
            "NDP.S/N": "000003",
        }

    def test_section_becomes_nested_dict(self):
        # Arrange — mirrors the real file layout: CRLF in globals, bare CR in section
        comments = (
            "Product=C13210\r\n;NDP Shading Data\r;Version=0005\r;ID=4\r;Name=TxRed"
        )

        # Act
        parsed = NdpiMetadata._parse_comments(comments)

        # Assert
        assert parsed == {
            "Product": "C13210",
            "NDP Shading Data": {
                "Version": "0005",
                "ID": "4",
                "Name": "TxRed",
            },
        }

    def test_value_with_equals_sign_preserved(self):
        # Arrange
        comments = ";Section\n;Formula=a=b+c\n"

        # Act
        parsed = NdpiMetadata._parse_comments(comments)

        # Assert
        assert parsed == {"Section": {"Formula": "a=b+c"}}

    def test_empty_string_returns_empty_dict(self):
        # Arrange
        comments = ""

        # Act
        parsed = NdpiMetadata._parse_comments(comments)

        # Assert
        assert parsed == {}


@pytest.mark.unittest
class TestNdpiJpegXrTiler:
    """Tests for ndpi files with JPEG XR compressed levels. JPEG XR cannot be passed
    through to DICOM, so the native (non-overlapping) tiles are exposed with a
    zero-overlap placement for the consumer to decode and stitch."""

    def test_levels_are_jpegxr_images(self, jpegxr_tiler: NdpiTiler):
        # Arrange

        # Act
        levels = jpegxr_tiler.levels

        # Assert
        assert len(levels) > 1
        assert all(isinstance(level, NdpiJpegXrImage) for level in levels)

    def test_compression(self, jpegxr_tiler: NdpiTiler):
        # Arrange
        level = jpegxr_tiler.get_level(0)

        # Act
        compression = level.compression

        # Assert
        assert compression == COMPRESSION.JPEGXR_NDPI

    def test_overlap_is_zero_overlap(self, jpegxr_tiler: NdpiTiler):
        # Arrange
        level = jpegxr_tiler.get_level(0)

        # Act
        overlap = level.overlap

        # Assert: every native tile is placed, and the composed size equals the level
        assert overlap is not None
        assert overlap.image_size == level.image_size
        assert len(overlap.placements) == level.tiled_size.area

    def test_get_tile_is_raw_passthrough(self, jpegxr_tiler: NdpiTiler):
        # Arrange
        level = jpegxr_tiler.get_level(0)

        # Act
        tile = level.get_tile((0, 0))

        # Assert: raw JPEG XR bytes are passed through, not re-encoded to jpeg
        assert len(tile) > 0
        assert not tile.startswith(b"\xff\xd8")

    def test_get_decoded_tile_shape(self, jpegxr_tiler: NdpiTiler):
        # Arrange
        level = jpegxr_tiler.get_level(0)

        # Act
        decoded = level.get_decoded_tile((0, 0))

        # Assert
        assert decoded.shape == (
            level.tile_size.height,
            level.tile_size.width,
            level.samples_per_pixel,
        )

    def test_coarsest_level_is_single_tile(self, jpegxr_tiler: NdpiTiler):
        # Arrange
        levels = jpegxr_tiler.levels

        # Act
        coarsest = levels[-1]

        # Assert: a single-frame level is exposed as one tile covering the level
        assert coarsest.tiled_size.area == 1

    def test_overview_and_label_skipped(self, jpegxr_tiler: NdpiTiler):
        # Arrange

        # Act
        overviews = jpegxr_tiler.overviews
        labels = jpegxr_tiler.labels

        # Assert: the macro is jpeg xr, which the associated-image classes cannot
        # decode yet, so it is skipped rather than raising
        assert overviews == []
        assert labels == []
