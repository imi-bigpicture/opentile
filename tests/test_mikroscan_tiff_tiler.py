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

import pytest

from opentile.formats import MikroscanTiffTiler
from opentile.geometry import Size, SizeMm
from opentile.tiff_format import TiffFormat
from opentile.tiff_image import LevelTiffImage

from .filepaths import mikroscan_file_path


@pytest.fixture()
def tiler():
    try:
        with MikroscanTiffTiler(mikroscan_file_path) as tiler:
            yield tiler
    except FileNotFoundError:
        pytest.skip("Mikroscan tiff test file not found, skipping")


@pytest.fixture()
def level(tiler: MikroscanTiffTiler):
    yield tiler.get_level(0)


@pytest.mark.unittest
class TestMikroscanTiffTiler:
    def test_format(self, tiler: MikroscanTiffTiler):
        # Arrange

        # Act
        format = tiler.format

        # Assert
        assert format == TiffFormat.MIKROSCAN

    def test_levels(self, tiler: MikroscanTiffTiler):
        # Arrange

        # Act
        levels = tiler.levels

        # Assert
        assert len(levels) == 4
        assert levels[0].image_size == Size(26880, 42240)
        assert all(level.tile_size == Size(256, 256) for level in levels)

    def test_pixel_spacing(self, level: LevelTiffImage):
        # Arrange

        # Act
        pixel_spacing = level.pixel_spacing

        # Assert: MPP = 0.453023 um -> 0.000453023 mm/pixel
        assert pixel_spacing == SizeMm(0.000453023, 0.000453023)

    def test_metadata(self, tiler: MikroscanTiffTiler):
        # Arrange

        # Act
        metadata = tiler.metadata

        # Assert: magnification is a scan parameter and fine to assert; the serial and
        # acquisition datetime come from a private file, so only their presence is
        # checked (their parsing is covered by the mocked TestMikroscanMetadata).
        assert metadata.scanner_manufacturer == "Mikroscan"
        assert metadata.magnification == 20.0
        assert metadata.scanner_model
        assert metadata.scanner_serial_number
        assert metadata.acquisition_datetime is not None

    def test_get_decoded_tile(self, level: LevelTiffImage):
        # Arrange

        # Act
        tile = level.get_decoded_tile((0, 0))

        # Assert
        assert tile.shape == (
            level.tile_size.height,
            level.tile_size.width,
            level.samples_per_pixel,
        )

    def test_get_tile_is_native_passthrough(self, level: LevelTiffImage):
        # Arrange

        # Act
        tile = level.get_tile((0, 0))

        # Assert: the native jpeg tile is served as-is
        assert len(tile) > 0
