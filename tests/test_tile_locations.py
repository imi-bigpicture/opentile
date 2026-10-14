#    Copyright 2026 SECTRA AB
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

from pathlib import Path

import numpy as np
import pytest

from opentile import OpenTile
from opentile.geometry import Point
from opentile.tiff_image import TiffImage
from opentile.tiff_image_bases import NativeTiledTiffImage
from opentile.tile_locations import TileLocations
from tests.filepaths import (
    argos_file_path,
    histech_file_path,
    huron_file_path,
    leica_scn_file_path,
    mikroscan_file_path,
    motic_file_path,
    ndpi_file_path,
    ome_tiff_file_path,
    philips_file_path,
    qptiff_file_path,
    svs_file_path,
    trestle_file_path,
    ventana_file_path,
)

MAX_TILES_CHECKED = 400
"""Tiles checked per image, spread over the image."""


def read_tile_from_locations(
    locations: TileLocations, tiff_image: TiffImage, x: int, y: int
) -> bytes | None:
    """Read tile as a consumer of the locations would, or None if it is to be read
    with `get_tile()`."""
    index = y * tiff_image.tiled_size.width + x
    length = int(locations.lengths[index])
    if length == TileLocations.READ_THROUGH_IMAGE:
        return None
    return tiff_image.frame_reader.read(int(locations.positions[index]), length)


def images_of(tiler) -> list[TiffImage]:
    return [
        *tiler.levels,
        *tiler.labels,
        *tiler.overviews,
        *tiler.thumbnails,
    ]


@pytest.mark.unittest
class TestTileLocations:
    @pytest.mark.parametrize(
        "file_path",
        [
            svs_file_path,
            ndpi_file_path,
            philips_file_path,
            histech_file_path,
            ome_tiff_file_path,
            trestle_file_path,
            ventana_file_path,
            huron_file_path,
            leica_scn_file_path,
            mikroscan_file_path,
            motic_file_path,
            qptiff_file_path,
            argos_file_path,
        ],
    )
    def test_tile_read_from_locations_is_tile_from_get_tile(self, file_path: Path):
        # Arrange
        if not file_path.exists():
            pytest.skip(f"{file_path} test file not found, skipping")
        with OpenTile.open(file_path) as tiler:
            mismatches: list[tuple[TiffImage, int, int]] = []
            checked = 0

            # Act
            for tiff_image in images_of(tiler):
                locations = tiff_image.tile_locations
                if locations is None:
                    continue
                tiled_size = tiff_image.tiled_size
                step = max(1, tiled_size.area // MAX_TILES_CHECKED)
                for index in range(0, tiled_size.area, step):
                    x, y = index % tiled_size.width, index // tiled_size.width
                    tile = read_tile_from_locations(locations, tiff_image, x, y)
                    if tile is None:
                        continue
                    checked += 1
                    if tile != tiff_image.get_tile((x, y)):
                        mismatches.append((tiff_image, x, y))

        # Assert
        assert mismatches == []
        if checked == 0:
            pytest.skip(f"No image of {file_path.name} has tile locations")

    @pytest.mark.parametrize(
        "file_path", [ndpi_file_path, trestle_file_path, ventana_file_path]
    )
    def test_levels_that_are_assembled_or_composed_have_no_tile_locations(
        self, file_path: Path
    ):
        # Arrange
        if not file_path.exists():
            pytest.skip(f"{file_path} test file not found, skipping")
        with OpenTile.open(file_path) as tiler:
            # Act
            locations = [level.tile_locations for level in tiler.levels]

        # Assert
        assert all(location is None for location in locations)

    @pytest.mark.parametrize(
        "file_path",
        [
            svs_file_path,
            philips_file_path,
            histech_file_path,
            ome_tiff_file_path,
            leica_scn_file_path,
            mikroscan_file_path,
            motic_file_path,
            qptiff_file_path,
            argos_file_path,
        ],
    )
    def test_frame_indices_of_tiles_are_frame_index_of_each_tile(self, file_path: Path):
        """The array form of the frame index of each tile, which the locations are
        made from, is the frame index `get_tile()` reads, also where a format maps
        tiles to frames its own way (e.g. the label of Leica scn)."""
        # Arrange
        if not file_path.exists():
            pytest.skip(f"{file_path} test file not found, skipping")
        with OpenTile.open(file_path) as tiler:
            native_images = [
                tiff_image
                for tiff_image in images_of(tiler)
                if isinstance(tiff_image, NativeTiledTiffImage)
            ]

            # Act
            mismatches = [
                str(tiff_image)
                for tiff_image in native_images
                if tiff_image._tile_frame_indices().tolist()
                != [
                    tiff_image._tile_point_to_frame_index(Point(x, y))
                    for y in range(tiff_image.tiled_size.height)
                    for x in range(tiff_image.tiled_size.width)
                ]
            ]

        # Assert
        if not native_images:
            pytest.skip(f"No natively tiled image in {file_path.name}")
        assert mismatches == []

    def test_from_frames_marks_tiles_not_stored_as_stored_to_read_through(self):
        # Arrange - frames of tiles 0 to 4: stored, length 0, too short, stored but
        # read through, and past the stored frames.
        frame_indices = np.array([0, 1, 2, 3, 4], dtype=np.int64)
        offsets = [100, 200, 300, 400]
        bytecounts = [10, 0, 3, 10]

        # Act
        locations = TileLocations.from_frames(
            frame_indices, offsets, bytecounts, minimum_length=4, read_through=[3]
        )

        # Assert
        read_through = TileLocations.READ_THROUGH_IMAGE
        assert locations.lengths.tolist() == [10] + [read_through] * 4
        assert locations.positions[0] == 100

    def test_from_frames_takes_frame_of_each_tile(self):
        # Arrange - tiles in another order than their frames.
        frame_indices = np.array([2, 0, 1], dtype=np.int64)

        # Act
        locations = TileLocations.from_frames(
            frame_indices, [100, 200, 300], [1, 2, 3], minimum_length=1, read_through=[]
        )

        # Assert
        assert locations.positions.tolist() == [300, 100, 200]
        assert locations.lengths.tolist() == [3, 1, 2]

    def test_arrays_are_made_read_only(self):
        """Consumers may hold views of the arrays rather than copies, so they are
        not to be changed after."""
        # Arrange
        positions = np.arange(4, dtype=np.int64)
        lengths = np.full(4, 10, dtype=np.int64)

        # Act
        locations = TileLocations(positions, lengths)

        # Assert
        assert not locations.positions.flags.writeable
        assert not locations.lengths.flags.writeable

    def test_svs_tile_read_at_location_has_jpeg_tables_added(self):
        """The svs level stores abbreviated jpeg tiles, read complete with the jpeg
        tables of the page by the frame reader of the level."""
        # Arrange
        if not svs_file_path.exists():
            pytest.skip(f"{svs_file_path} test file not found, skipping")
        with OpenTile.open(svs_file_path) as tiler:
            level = tiler.get_level(0)
            locations = level.tile_locations
            assert locations is not None
            position, length = int(locations.positions[0]), int(locations.lengths[0])

            # Act
            tile = level.frame_reader.read(position, length)

        # Assert
        define_quantization_table = b"\xff\xdb"
        assert tile.startswith(b"\xff\xd8")
        assert define_quantization_table in tile[:1000]
