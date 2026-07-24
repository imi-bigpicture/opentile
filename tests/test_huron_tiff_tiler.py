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
from tifffile import COMPRESSION

from opentile.formats import HuronTiffTiler
from opentile.geometry import Size, SizeMm
from opentile.tiff_format import TiffFormat
from opentile.tiff_image import LevelTiffImage

from .filepaths import huron_file_path


@pytest.fixture()
def tiler():
    try:
        with HuronTiffTiler(huron_file_path) as tiler:
            yield tiler
    except FileNotFoundError:
        pytest.skip("Huron tiff test file not found, skipping")


@pytest.fixture()
def level(tiler: HuronTiffTiler):
    yield tiler.get_level(0)


@pytest.mark.unittest
class TestHuronTiffTiler:
    def test_format(self, tiler: HuronTiffTiler):
        # Arrange

        # Act
        format = tiler.format

        # Assert
        assert format == TiffFormat.HURON

    def test_levels(self, tiler: HuronTiffTiler):
        # Arrange

        # Act
        levels = tiler.levels

        # Assert
        assert len(levels) == 3
        assert levels[0].image_size == Size(6022, 10503)
        assert all(level.tile_size == Size(256, 256) for level in levels)

    def test_pixel_spacing(self, level: LevelTiffImage):
        # Arrange

        # Act
        pixel_spacing = level.pixel_spacing

        # Assert: Resolution = 0.40 um -> 0.4 um/pixel -> 0.0004 mm/pixel
        assert pixel_spacing == SizeMm(0.0004, 0.0004)

    def test_metadata(self, tiler: HuronTiffTiler):
        # Arrange

        # Act
        metadata = tiler.metadata

        # Assert
        assert metadata.scanner_manufacturer == "Huron Digital Pathology"
        assert metadata.scanner_model == "LE176"
        assert metadata.scanner_serial_number == "LE176"
        assert metadata.scanner_software_versions == ["MACROscan LE2.0 v1.1.2.0"]
        assert metadata.properties["Resolution"] == "0.40 um"

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

        # Assert: the native tile is served as-is
        assert len(tile) > 0

    @pytest.mark.parametrize(
        ["associated", "expected_size"],
        [
            ("labels", Size(250, 230)),
            ("overviews", Size(273, 530)),
            ("thumbnails", Size(377, 657)),
        ],
    )
    def test_associated_images(
        self, tiler: HuronTiffTiler, associated: str, expected_size: Size
    ):
        # Arrange
        images = getattr(tiler, associated)

        # Act
        image = images[0]
        decoded = image.get_decoded_tile((0, 0))

        # Assert: uncompressed strips are decoded and served as raw pixels
        assert image.image_size == expected_size
        assert image.compression == COMPRESSION.NONE
        assert image.get_tile((0, 0)) == decoded.tobytes()
        assert decoded.shape == (
            expected_size.height,
            expected_size.width,
            image.samples_per_pixel,
        )
