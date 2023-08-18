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

from datetime import datetime
from hashlib import md5
from typing import Sequence, Tuple, cast

import pytest
from tifffile.tifffile import PHOTOMETRIC

from opentile.formats import SvsTiler
from opentile.formats.svs.svs_image import SvsTiledImage
from opentile.geometry import Point

from .filepaths import svs_file_path


@pytest.fixture()
def tiler():
    try:
        with SvsTiler(svs_file_path) as tiler:
            yield tiler
    except FileNotFoundError:
        pytest.skip("Svs test file not found, skipping")


@pytest.fixture()
def level(tiler: SvsTiler):
    yield cast(SvsTiledImage, tiler.get_level(0))


@pytest.mark.unittest
class TestSvsTiler:
    @pytest.mark.parametrize(
        ["tile_point", "hash"],
        [
            ((15, 25), "dd8402d5a7250ebfe9e33b3dfee3c1e1"),
            ((35, 30), "50f885cd7299bd3fff22743bfb4a4930"),
        ],
    )
    def test_get_tile(
        self, level: SvsTiledImage, tile_point: Tuple[int, int], hash: str
    ):
        # Arrange

        # Act
        tile = level.get_tile(tile_point)

        # Assert
        assert md5(tile).hexdigest() == hash

    @pytest.mark.parametrize(
        ["tile_points", "hashes"],
        [
            (
                [(15, 25), (35, 30)],
                [
                    "dd8402d5a7250ebfe9e33b3dfee3c1e1",
                    "50f885cd7299bd3fff22743bfb4a4930",
                ],
            ),
        ],
    )
    def test_get_tiles(
        self,
        level: SvsTiledImage,
        tile_points: Sequence[Tuple[int, int]],
        hashes: Sequence[str],
    ):
        # Arrange

        # Act
        tiles = level.get_tiles(tile_points)

        # Assert
        for tile, hash in zip(tiles, hashes):
            assert md5(tile).hexdigest() == hash

    @pytest.mark.parametrize(
        ["tile_point", "hash"],
        [
            ((0, 0), "d9135db3b0bcc0d9e785754e760f80c4"),
            ((50, 50), "bd0599aa1becf3511fa122582ecc7e3d"),
        ],
    )
    def test_get_scaled_tile(
        self, tiler: SvsTiler, tile_point: Tuple[int, int], hash: str
    ):
        # Arrange
        level = cast(SvsTiledImage, tiler.get_level(1))

        # Act
        tile = level._get_scaled_tile(Point.from_tuple(tile_point))

        # Assert
        assert md5(tile).hexdigest() == hash

    @pytest.mark.parametrize(
        ["tile", "right", "bottom"],
        [
            (Point(178, 127), False, False),
            (Point(178, 128), False, True),
            (Point(179, 127), True, False),
            (Point(179, 128), True, True),
        ],
    )
    def test_tile_is_at_edge(
        self, level: SvsTiledImage, tile: Point, right: bool, bottom: bool
    ):
        # Arrange

        # Act
        tile_is_at_right_edge = level._tile_is_at_right_edge(tile)
        tile_is_at_bottom_edge = level._tile_is_at_bottom_edge(tile)

        # Assert
        assert tile_is_at_right_edge == right
        assert tile_is_at_bottom_edge == bottom

    def test_detect_corrupt_edges(self, level: SvsTiledImage):
        # Arrange

        # Act
        is_corrupt = level._detect_corrupt_edges()

        # Assert
        assert is_corrupt == (False, False)

    def test_photometric_interpretation(self, level: SvsTiledImage):
        # Arrange

        # Act
        photometric_interpretatation = level.photometric_interpretation

        # Assert
        assert photometric_interpretatation == PHOTOMETRIC.RGB

    def test_subsampling(self, level: SvsTiledImage):
        # Arrange

        # Act
        subsampling = level.subsampling

        # Assert
        assert subsampling == (2, 2)

    def test_sumples_per_pixel(self, level: SvsTiledImage):
        # Arrange

        # Act
        samples_per_pixel = level.samples_per_pixel

        # Assert
        assert samples_per_pixel == 3

    def test_metadata_magnification(self, tiler: SvsTiler):
        # Arrange

        # Act
        magnification = tiler.metadata.magnification

        # Assert
        assert magnification == 20.0

    def test_metadata_aquisition_datetime(self, tiler: SvsTiler):
        # Arrange

        # Act
        aquisition_datetime = tiler.metadata.aquisition_datetime

        # Assert
        assert aquisition_datetime == datetime(2009, 12, 29, 9, 59, 15)
