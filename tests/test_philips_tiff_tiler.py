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
from typing import Sequence, Tuple

import pytest
from tifffile.tifffile import PHOTOMETRIC

from opentile.formats import PhilipsTiffTiler
from opentile.tiff_image import TiffImage

from .filepaths import philips_file_path


@pytest.fixture()
def tiler():
    try:
        with PhilipsTiffTiler(philips_file_path) as tiler:
            yield tiler
    except FileNotFoundError:
        pytest.skip("Philips tiff test file not found, skipping")


@pytest.fixture()
def level(tiler: PhilipsTiffTiler):
    yield tiler.get_level(0)


@pytest.mark.unittest
class TestPhilipsTiffTiler:
    @pytest.mark.parametrize(
        ["tile_point", "hash"],
        [
            ((0, 0), "570d069f9de5d2716fb0d7167bc79195"),
            ((20, 20), "db28efb73a72ef7e2780fc72c624d7ae"),
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
                [(0, 0), (20, 20)],
                [
                    "570d069f9de5d2716fb0d7167bc79195",
                    "db28efb73a72ef7e2780fc72c624d7ae",
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

        # Assert
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
        assert subsampling == (2, 2)

    def test_sumples_per_pixel(self, level: TiffImage):
        # Arrange

        # Act
        samples_per_pixel = level.samples_per_pixel

        # Assert
        assert samples_per_pixel == 3

    def test_metadata_scanner_manufacturer(self, tiler: PhilipsTiffTiler):
        # Arrange

        # Act
        scanner_manufacturer = tiler.metadata.scanner_manufacturer

        # Assert
        assert scanner_manufacturer == "PHILIPS"

    def test_metadata_scanner_software_versions(self, tiler: PhilipsTiffTiler):
        # Arrange

        # Act
        scanner_software_versions = tiler.metadata.scanner_software_versions

        # Assert
        assert scanner_software_versions == ["1.6.5505", "20111209_R44", "4.0.3"]

    def test_metadata_scanner_serial_number(self, tiler: PhilipsTiffTiler):
        # Arrange

        # Act
        scanner_serial_number = tiler.metadata.scanner_serial_number

        # Assert
        assert scanner_serial_number is not None

    def test_metadata_aquisition_datetime(self, tiler: PhilipsTiffTiler):
        # Arrange

        # Act
        aquisition_datetime = tiler.metadata.aquisition_datetime

        # Assert
        assert aquisition_datetime == datetime(2013, 7, 1, 18, 59, 4)
