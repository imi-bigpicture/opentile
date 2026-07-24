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

from opentile.formats import MoticTiffTiler
from opentile.geometry import Size, SizeMm
from opentile.tiff_format import TiffFormat
from opentile.tiff_image import LevelTiffImage

from .filepaths import motic_file_path


@pytest.fixture()
def tiler():
    try:
        with MoticTiffTiler(motic_file_path) as tiler:
            yield tiler
    except FileNotFoundError:
        pytest.skip("Motic tiff test file not found, skipping")


@pytest.fixture()
def level(tiler: MoticTiffTiler):
    yield tiler.get_level(0)


@pytest.mark.unittest
class TestMoticTiffTiler:
    def test_format(self, tiler: MoticTiffTiler):
        # Arrange

        # Act
        format = tiler.format

        # Assert
        assert format == TiffFormat.MOTIC

    def test_levels(self, tiler: MoticTiffTiler):
        # Arrange

        # Act
        levels = tiler.levels

        # Assert: a 3-level pyramid (scales 1, 4, 16)
        assert len(levels) == 3
        assert levels[0].image_size == Size(53046, 51735)
        assert levels[1].image_size == Size(13261, 12933)
        assert levels[2].image_size == Size(3315, 3233)
        assert all(level.tile_size == Size(512, 512) for level in levels)

    def test_pixel_spacing(self, level: LevelTiffImage):
        # Arrange

        # Act
        pixel_spacing = level.pixel_spacing

        # Assert: MPP = 0.260417 um -> 0.000260417 mm/pixel
        assert pixel_spacing == SizeMm(0.000260417, 0.000260417)

    def test_metadata(self, tiler: MoticTiffTiler):
        # Arrange

        # Act
        metadata = tiler.metadata

        # Assert: magnification is a scan parameter and fine to assert; the barcode
        # comes from a private file, so only its presence is checked (its parsing is
        # covered by the mocked TestMoticMetadata).
        assert metadata.scanner_manufacturer == "Motic"
        assert metadata.magnification == 40.0
        assert metadata.scanner_software_versions
        assert metadata.barcode

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

    def test_label(self, tiler: MoticTiffTiler):
        # Arrange

        # Act
        label = tiler.get_label()

        # Assert: the lzw label is decoded to raw pixels
        assert label.image_size == Size(588, 472)
        assert label.get_decoded_tile((0, 0)).shape == (472, 588, 3)

    def test_overview(self, tiler: MoticTiffTiler):
        # Arrange

        # Act
        overview = tiler.get_overview()

        # Assert: the lzw macro is decoded to raw pixels
        assert overview.image_size == Size(160, 456)
        assert overview.get_decoded_tile((0, 0)).shape == (456, 160, 3)

    def test_thumbnail(self, tiler: MoticTiffTiler):
        # Arrange

        # Act
        thumbnail = tiler.get_thumbnail()

        # Assert: the jpeg thumbnail
        assert thumbnail.image_size == Size(847, 826)
        assert thumbnail.get_decoded_tile((0, 0)).shape == (826, 847, 3)
