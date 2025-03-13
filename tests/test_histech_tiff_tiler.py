#    Copyright 2022-2023 SECTRA AB
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
from typing import Sequence, Tuple

import pytest
from tifffile import PHOTOMETRIC

from opentile.formats import HistechTiffTiler
from opentile.geometry import SizeMm
from opentile.tiff_image import TiffImage

from .filepaths import histech_file_path


@pytest.fixture()
def tiler():
    try:
        with HistechTiffTiler(histech_file_path) as tiler:
            yield tiler
    except FileNotFoundError:
        pytest.skip("Histech tiff test file not found, skipping")


@pytest.fixture()
def level(tiler: HistechTiffTiler):
    yield tiler.get_level(0)


@pytest.mark.unittest
class TestHistechTiffTiler:
    @pytest.mark.parametrize(
        ["tile_point", "hash"],
        [
            ((50, 50), "221f47792ebef7b9e394fc6c8ed7cb64"),
            ((100, 100), "0a2e459e94e9237bb866b3bc1ac10cb8"),
        ],
    )
    def test_get_tile(self, level: TiffImage, tile_point: Tuple[int, int], hash: str):
        # Arrange

        # Act
        tile = level.get_tile(tile_point)

        # Assert
        assert md5(tile).hexdigest() == hash

    @pytest.mark.parametrize(
        ["tile_points", "hashes"],
        [
            (
                [(50, 50), (100, 100)],
                [
                    "221f47792ebef7b9e394fc6c8ed7cb64",
                    "0a2e459e94e9237bb866b3bc1ac10cb8",
                ],
            ),
        ],
    )
    def test_get_tiles(
        self,
        level: TiffImage,
        tile_points: Sequence[Tuple[int, int]],
        hashes: Sequence[str],
    ):
        # Arrange

        # Act
        tiles = level.get_tiles(tile_points)

        # assert
        for tile, hash in zip(tiles, hashes):
            assert md5(tile).hexdigest() == hash

    def test_photometric_interpretation(self, level: TiffImage):
        # Arrange

        # Act
        photometric_interpretation = level.photometric_interpretation

        # Assert
        assert photometric_interpretation == PHOTOMETRIC.YCBCR

    def test_subsampling(self, level: TiffImage):
        # Arrange

        # Act
        subsampling = level.subsampling

        # Assert
        assert subsampling is None

    def test_sumples_per_pixel(self, level: TiffImage):
        # Arrange

        # Act
        samples_per_pixel = level.samples_per_pixel

        # Assert
        assert samples_per_pixel == 3

    def test_compressed_size(self, level: TiffImage):
        # Arrange

        # Act
        compressed_size = level.compressed_size

        # Assert
        assert compressed_size == 425684721

    @pytest.mark.parametrize(
        ["level", "expected_size"],
        [
            (0, SizeMm(0.0002325, 0.0002325)),
        ],
    )
    def test_pixel_spacing(
        self, tiler: HistechTiffTiler, level: int, expected_size: SizeMm
    ):
        # Arrange
        base_level = tiler.get_level(level)

        # Act
        base_pixel_spacing = base_level.pixel_spacing

        # Assert
        assert base_pixel_spacing == expected_size
