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

from hashlib import md5

import numpy as np
import pytest
from tifffile import PHOTOMETRIC

from opentile.formats import TrestleTiffTiler
from opentile.geometry import Point, PointF, Size, SizeMm
from opentile.tiff_image import OverlappingLevelTiffImage

from .filepaths import trestle_file_path


@pytest.fixture()
def tiler():
    try:
        with TrestleTiffTiler(trestle_file_path) as tiler:
            yield tiler
    except FileNotFoundError:
        pytest.skip("Trestle tiff test file not found, skipping")


@pytest.fixture()
def level(tiler: TrestleTiffTiler):
    yield tiler.get_level(0)


@pytest.mark.unittest
class TestTrestleTiffTiler:
    @pytest.mark.parametrize(
        ["tile_point", "hash"],
        [
            ((50, 50), "a422623de2ab302a4db55af48e181bc4"),
            ((80, 60), "27e2e5203c311a2df0d449d89887867e"),
        ],
    )
    def test_get_tile(
        self,
        level: OverlappingLevelTiffImage,
        tile_point: tuple[int, int],
        hash: str,
    ):
        # Arrange

        # Act
        tile = level.get_tile(tile_point)

        # Assert
        assert md5(tile).hexdigest() == hash

    def test_photometric_interpretation(self, level: OverlappingLevelTiffImage):
        # Arrange

        # Act
        photometric_interpretation = level.photometric_interpretation

        # Assert
        assert photometric_interpretation == PHOTOMETRIC.YCBCR

    def test_samples_per_pixel(self, level: OverlappingLevelTiffImage):
        # Arrange

        # Act
        samples_per_pixel = level.samples_per_pixel

        # Assert
        assert samples_per_pixel == 3

    def test_magnification(self, tiler: TrestleTiffTiler):
        # Arrange

        # Act
        magnification = tiler.metadata.magnification

        # Assert
        assert magnification == 10.0

    def test_level_count(self, tiler: TrestleTiffTiler):
        # Arrange

        # Act
        levels = tiler.levels

        # Assert
        assert len(levels) == 7

    @pytest.mark.parametrize(
        ["level_index", "expected_pyramid_index", "expected_composed_size"],
        [
            (0, 0, Size(40000, 27712)),
            (1, 1, Size(20000, 13856)),
            (2, 2, Size(10000, 6928)),
        ],
    )
    def test_composed_size(
        self,
        tiler: TrestleTiffTiler,
        level_index: int,
        expected_pyramid_index: int,
        expected_composed_size: Size,
    ):
        # Arrange
        level = tiler.get_level(level_index)

        # Act
        overlap = level.overlap

        # Assert
        assert overlap is not None
        assert level.pyramid_index == expected_pyramid_index
        assert overlap.image_size == expected_composed_size

    def test_pixel_spacing(self, level: OverlappingLevelTiffImage):
        # Arrange

        # Act
        pixel_spacing = level.pixel_spacing

        # Assert
        assert pixel_spacing == SizeMm(0.000574691891670227, 0.0005750624537467957)

    def test_tile_positions_de_overlap(self, level: OverlappingLevelTiffImage):
        # Arrange
        overlap = level.overlap
        tile_width, tile_height = level.tile_size.to_tuple()

        # Act - each stored tile contributes a single whole-tile piece
        origin = overlap.placements[Point(0, 0)][0].position
        right = overlap.placements[Point(1, 0)][0].position
        down = overlap.placements[Point(0, 1)][0].position

        # Assert - each tile advances by (tile - overlap), overlap is 64 on base level
        assert origin == PointF(0, 0)
        assert right == PointF(tile_width - 64, 0)
        assert down == PointF(0, tile_height - 64)

    def test_overlapping_tiles_share_pixels(self, level: OverlappingLevelTiffImage):
        # Arrange - a tissue tile and its right neighbour
        overlap_x = 64
        tile_width = level.tile_size.width
        left = level.get_decoded_tile((30, 40))
        right = level.get_decoded_tile((31, 40))

        # Act - the right edge of a tile covers the same pixels as the left edge of
        # its neighbour, so de-overlapping by 64 px aligns them
        overlap_difference = np.abs(
            left[:, tile_width - overlap_x :].astype(int) - right[:, :overlap_x]
        ).mean()

        # Assert
        assert overlap_difference < 1.0
