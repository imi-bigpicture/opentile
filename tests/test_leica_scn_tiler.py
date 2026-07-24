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

from datetime import datetime, timezone
from hashlib import md5

import pytest
from tifffile import PHOTOMETRIC

from opentile.formats import LeicaScnTiler
from opentile.geometry import SizeMm
from opentile.tiff_image import LevelTiffImage

from .filepaths import leica_scn_file_path


@pytest.fixture()
def tiler():
    try:
        with LeicaScnTiler(leica_scn_file_path) as tiler:
            yield tiler
    except FileNotFoundError:
        pytest.skip("Leica SCN test file not found, skipping")


@pytest.fixture()
def level(tiler: LeicaScnTiler):
    yield tiler.get_level(0)


@pytest.mark.unittest
class TestLeicaScnTiler:
    @pytest.mark.parametrize(
        ["tile_point", "hash"],
        [
            ((0, 0), "6cca7207316528fc6a26b294ebcd0e98"),
            ((10, 10), "89b700b2a8d0996ac84659e70186b73c"),
        ],
    )
    def test_get_tile(
        self, level: LevelTiffImage, tile_point: tuple[int, int], hash: str
    ):
        # Arrange

        # Act
        tile = level.get_tile(tile_point)

        # Assert
        assert md5(tile).hexdigest() == hash

    def test_levels_are_dyadic(self, tiler: LeicaScnTiler):
        # Arrange

        # Act
        indices = [level.pyramid_index for level in tiler.levels]

        # Assert
        assert indices == [0, 2, 4, 6, 8]

    def test_photometric_interpretation(self, level: LevelTiffImage):
        # Arrange

        # Act
        photometric_interpretation = level.photometric_interpretation

        # Assert
        assert photometric_interpretation == PHOTOMETRIC.YCBCR

    def test_subsampling(self, level: LevelTiffImage):
        # Arrange

        # Act
        subsampling = level.subsampling

        # Assert
        assert subsampling == (2, 2)

    def test_pixel_spacing(self, level: LevelTiffImage):
        # Arrange

        # Act
        pixel_spacing = level.pixel_spacing

        # Assert
        assert pixel_spacing == SizeMm(0.0005, 0.0005)

    def test_overview(self, tiler: LeicaScnTiler):
        # Arrange

        # Act
        overviews = tiler.overviews

        # Assert
        assert len(overviews) == 1
        assert md5(overviews[0].get_tile((0, 0))).hexdigest() is not None

    def test_label_cropped_from_overview(self, tiler: LeicaScnTiler):
        # Arrange
        overview = tiler.get_overview()

        # Act
        label = tiler.get_label()

        # Assert - the label is the bottom tile rows of the macro, cropped down to a
        # whole tile row so it is never clipped
        assert len(tiler.labels) == 1
        assert label.image_size.width == overview.image_size.width
        assert label.image_size.height < overview.image_size.height
        cropped_off = overview.image_size.height - label.image_size.height
        assert cropped_off % label.tile_size.height == 0

    def test_label_tiles_are_macro_tiles(self, tiler: LeicaScnTiler):
        # Arrange
        overview = tiler.get_overview()
        label = tiler.get_label()
        row_offset = overview.tiled_size.height - label.tiled_size.height

        # Act
        label_tile = label.get_tile((0, 0))

        # Assert - tiles are served unchanged from the macro (lossless crop)
        assert label_tile == overview.get_tile((0, row_offset))

    def test_metadata_barcode(self, tiler: LeicaScnTiler):
        # Arrange

        # Act
        barcode = tiler.metadata.barcode

        # Assert
        assert barcode == "04050629C"

    def test_metadata_magnification(self, tiler: LeicaScnTiler):
        # Arrange

        # Act
        magnification = tiler.metadata.magnification

        # Assert
        assert magnification == 20.0

    def test_metadata_scanner(self, tiler: LeicaScnTiler):
        # Arrange

        # Act
        metadata = tiler.metadata

        # Assert
        assert metadata.scanner_manufacturer == "Leica"
        assert metadata.scanner_model == "Leica SCN400"

    def test_metadata_acquisition_datetime(self, tiler: LeicaScnTiler):
        # Arrange

        # Act
        acquisition_datetime = tiler.metadata.acquisition_datetime

        # Assert
        assert acquisition_datetime == datetime(
            2011, 5, 31, 9, 43, 6, 873000, tzinfo=timezone.utc
        )
