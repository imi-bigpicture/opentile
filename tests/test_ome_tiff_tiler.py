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
from typing import Tuple

import pytest

from opentile.formats import OmeTiffTiler
from opentile.tiff_image import TiffImage

from .filepaths import ome_tiff_file_path


@pytest.fixture()
def tiler():
    try:
        with OmeTiffTiler(ome_tiff_file_path) as tiler:
            yield tiler
    except FileNotFoundError:
        pytest.skip("Ome tiff test file not found, skipping")


@pytest.fixture()
def level(tiler: OmeTiffTiler):
    yield tiler.get_level(0)


@pytest.mark.unittest
class TestOmeTiffTiler:
    @pytest.mark.parametrize(
        ["tile_point", "hash"],
        [
            ((0, 0), "646c70833b30aab095950424b59a0cf7"),
            ((20, 20), "4c37c335b697aaf1550f77fd9e367f69"),
        ],
    )
    def test_get_tile(self, level: TiffImage, tile_point: Tuple[int, int], hash: str):
        # Arrange

        # Act
        tile = level.get_tile(tile_point)

        # Assert
        assert md5(tile).hexdigest() == hash

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
        assert compressed_size == 104115549
