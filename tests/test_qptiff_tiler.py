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

from datetime import datetime
from hashlib import md5

import pytest
from tifffile import COMPRESSION, PHOTOMETRIC, TiffFile

from opentile.formats import QptiffTiler
from opentile.tiff_image import LevelTiffImage

from .filepaths import qptiff_file_path, qptiff_fluorescence_file_path


@pytest.mark.unittest
class TestQptiffTiler:
    @pytest.fixture()
    def tiler(self):
        try:
            with QptiffTiler(qptiff_file_path) as tiler:
                yield tiler
        except FileNotFoundError:
            pytest.skip("qptiff test file not found, skipping")

    @pytest.fixture()
    def level(self, tiler: QptiffTiler):
        yield tiler.get_level(0)

    def test_levels(self, tiler: QptiffTiler):
        # Arrange

        # Act
        levels = tiler.levels

        # Assert
        # 4 tiled levels and the coarsest (1920x1665) level, which is striped.
        assert len(levels) == 5
        assert [level.image_size.width for level in levels] == [
            30720,
            15360,
            7680,
            3840,
            1920,
        ]

    def test_get_tile(self, level: LevelTiffImage):
        # Arrange

        # Act
        tile = level.get_tile((30, 25))

        # Assert
        assert md5(tile).hexdigest() == "9018e0a292003e0ce00040388f88d85d"

    def test_tile_matches_page_data(self, level: LevelTiffImage):
        # Arrange
        with TiffFile(qptiff_file_path) as tiff_file:
            expected = tiff_file.pages[0].asarray()[
                25 * 512 : 26 * 512, 30 * 512 : 31 * 512
            ]

        # Act
        decoded = level.get_decoded_tile((30, 25))

        # Assert
        assert (decoded == expected).all()

    def test_striped_level_is_single_tile(self, tiler: QptiffTiler):
        # Arrange
        # The coarsest level is stored as strips and served as one tile.

        # Act
        level = tiler.get_level(4)

        # Assert
        assert level.tiled_size.to_tuple() == (1, 1)
        assert level.tile_size == level.image_size
        assert level.pyramid_index == 4

    def test_photometric_interpretation(self, level: LevelTiffImage):
        # Arrange

        # Act
        photometric_interpretation = level.photometric_interpretation

        # Assert
        assert photometric_interpretation == PHOTOMETRIC.RGB

    def test_pixel_spacing(self, level: LevelTiffImage):
        # Arrange

        # Act
        x, y = level.pixel_spacing.to_tuple()

        # Assert
        assert x == pytest.approx(0.000499, abs=1e-6)
        assert y == pytest.approx(0.000499, abs=1e-6)

    def test_associated_images(self, tiler: QptiffTiler):
        # Arrange

        # Act
        thumbnails = tiler.thumbnails
        overviews = tiler.overviews
        labels = tiler.labels

        # Assert
        assert len(thumbnails) == 1
        assert len(overviews) == 1
        assert len(labels) == 1

    def test_lzw_associated_image_is_served_decoded(self, tiler: QptiffTiler):
        # Arrange
        # The label is stored as lzw strips, which cannot be concatenated, so it is
        # served as raw bytes.
        with TiffFile(qptiff_file_path) as tiff_file:
            expected = tiff_file.pages[7].asarray().tobytes()

        # Act
        label = tiler.get_label()
        tile = label.get_tile((0, 0))

        # Assert
        assert label.compression == COMPRESSION.NONE
        assert tile == expected

    def test_metadata_magnification(self, tiler: QptiffTiler):
        # Arrange

        # Act
        magnification = tiler.metadata.magnification

        # Assert
        assert magnification == 20.0

    def test_metadata_scanner(self, tiler: QptiffTiler):
        # Arrange

        # Act
        metadata = tiler.metadata

        # Assert
        assert metadata.scanner_manufacturer == "PerkinElmer"
        assert metadata.scanner_model == "VectraPolaris"
        assert metadata.scanner_software_versions == ["1.0"]

    def test_metadata_acquisition_datetime(self, tiler: QptiffTiler):
        # Arrange

        # Act
        acquisition_datetime = tiler.metadata.acquisition_datetime

        # Assert
        assert acquisition_datetime == datetime(2017, 10, 25, 15, 46, 4)

    def test_metadata_properties(self, tiler: QptiffTiler):
        # Arrange

        # Act
        properties = tiler.metadata.properties

        # Assert
        assert properties["SlideID"] == "HandEcompressed"
        assert properties["Objective"] == "20x"
        assert properties["IsUnmixedComponent"] == "False"


@pytest.mark.unittest
class TestQptiffFluorescenceTiler:
    """Multi-band (fluorescence) qptiff, holding one grayscale image per band."""

    @pytest.fixture()
    def tiler(self):
        try:
            with QptiffTiler(qptiff_fluorescence_file_path) as tiler:
                yield tiler
        except FileNotFoundError:
            pytest.skip("qptiff fluorescence test file not found, skipping")

    def test_levels(self, tiler: QptiffTiler):
        # Arrange

        # Act
        levels = tiler.levels

        # Assert
        # 6 pyramid levels of 5 bands each.
        assert len(levels) == 30
        assert [level.pyramid_index for level in levels[:6]] == [0, 0, 0, 0, 0, 1]

    def test_bands_are_optical_paths(self, tiler: QptiffTiler):
        # Arrange

        # Act
        optical_paths = [
            level.optical_path for level in tiler.levels if level.pyramid_index == 0
        ]

        # Assert
        assert optical_paths == ["DAPI", "FITC", "CY3", "Texas Red", "CY5"]

    def test_band_is_grayscale(self, tiler: QptiffTiler):
        # Arrange

        # Act
        level = tiler.get_level(0, 1)

        # Assert
        assert level.samples_per_pixel == 1
        assert level.photometric_interpretation == PHOTOMETRIC.MINISBLACK
        assert level.compression == COMPRESSION.LZW

    def test_tile_matches_page_data(self, tiler: QptiffTiler):
        # Arrange
        # Level 2, band 2 (CY3) is directory 13.
        with TiffFile(qptiff_fluorescence_file_path) as tiff_file:
            expected = tiff_file.pages[13].asarray()[
                10 * 512 : 11 * 512, 6 * 512 : 7 * 512
            ]

        # Act
        decoded = tiler.get_level(2, 2).get_decoded_tile((6, 10))

        # Assert
        assert (decoded == expected).all()

    def test_associated_images(self, tiler: QptiffTiler):
        # Arrange

        # Act
        thumbnails = tiler.thumbnails
        overviews = tiler.overviews
        labels = tiler.labels

        # Assert
        assert len(thumbnails) == 1
        assert len(overviews) == 1
        assert len(labels) == 1

    def test_metadata(self, tiler: QptiffTiler):
        # Arrange

        # Act
        metadata = tiler.metadata

        # Assert
        assert metadata.magnification == 10.0
        assert metadata.acquisition_datetime == datetime(2017, 10, 5, 9, 23, 3)
        assert metadata.properties["SlideID"] == "LuCa-7color"
