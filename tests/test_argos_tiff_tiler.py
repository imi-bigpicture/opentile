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
from tifffile import PHOTOMETRIC

from opentile.formats import ArgosTiffTiler
from opentile.tiff_image import LevelTiffImage

from .filepaths import argos_file_path, argos_z_file_path


@pytest.mark.unittest
class TestArgosTiffTiler:
    @pytest.fixture()
    def tiler(self):
        try:
            with ArgosTiffTiler(argos_file_path) as tiler:
                yield tiler
        except FileNotFoundError:
            pytest.skip("Argos avs test file not found, skipping")

    @pytest.fixture()
    def level(self, tiler: ArgosTiffTiler):
        yield tiler.get_level(0)

    def test_levels(self, tiler: ArgosTiffTiler):
        # Arrange

        # Act
        levels = tiler.levels

        # Assert
        assert len(levels) == 8

    def test_get_populated_tile(self, level: LevelTiffImage):
        # Arrange
        # (50, 11) is the first populated tile; sparse tiles elsewhere return a blank.

        # Act
        tile = level.get_tile((50, 11))

        # Assert
        assert md5(tile).hexdigest() == "311f5563c89b4081009957d37d8eda44"

    def test_sparse_tile_is_blank(self, level: LevelTiffImage):
        # Arrange
        # (0, 0) has zero byte count and is served as a white tile that decodes to max.

        # Act
        decoded = level.get_decoded_tile((0, 0))

        # Assert
        assert decoded.mean() == 255.0

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

    def test_samples_per_pixel(self, level: LevelTiffImage):
        # Arrange

        # Act
        samples_per_pixel = level.samples_per_pixel

        # Assert
        assert samples_per_pixel == 3

    def test_pixel_spacing(self, level: LevelTiffImage):
        # Arrange

        # Act
        x, y = level.pixel_spacing.to_tuple()

        # Assert
        assert x == pytest.approx(0.000383, abs=1e-6)
        assert y == pytest.approx(0.000383, abs=1e-6)

    def test_associated_images(self, tiler: ArgosTiffTiler):
        # Arrange

        # Act
        thumbnails = tiler.thumbnails
        overviews = tiler.overviews
        labels = tiler.labels

        # Assert
        assert len(thumbnails) == 1
        assert len(overviews) == 1
        assert len(labels) == 1

    def test_label_cropped_from_macro(self, tiler: ArgosTiffTiler):
        # Arrange

        # Act
        label = tiler.get_label()
        tile = label.get_tile((0, 0))

        # Assert
        # Label is the right-hand crop of the 1489-wide macro (label_crop_position 0.76).
        assert label.image_size.width < tiler.get_overview().image_size.width
        assert md5(tile).hexdigest() == "1196354a4d1c299df85a63dad3c5a0fe"

    def test_metadata_magnification(self, tiler: ArgosTiffTiler):
        # Arrange

        # Act
        magnification = tiler.metadata.magnification

        # Assert
        assert magnification == 20.0

    def test_metadata_barcode(self, tiler: ArgosTiffTiler):
        # Arrange

        # Act
        barcode = tiler.metadata.barcode

        # Assert
        assert barcode == "$63844615"

    def test_metadata_acquisition_datetime(self, tiler: ArgosTiffTiler):
        # Arrange

        # Act
        acquisition_datetime = tiler.metadata.acquisition_datetime

        # Assert
        assert acquisition_datetime == datetime(2024, 1, 23, 10, 57, 59)

    def test_single_plane_focal_plane_is_zero(self, level: LevelTiffImage):
        # Arrange

        # Act
        focal_plane = level.focal_plane

        # Assert
        assert focal_plane == 0.0

    def test_metadata_scan_area(self, tiler: ArgosTiffTiler):
        # Arrange

        # Act
        properties = tiler.metadata.properties

        # Assert
        assert properties["ScanArea.X1"] == "0.15"
        assert properties["ScanArea.X2"] == "5.15"
        assert properties["ScanArea.Y1"] == "0.15"
        assert properties["ScanArea.Y2"] == "2.35"


@pytest.mark.unittest
class TestArgosStackedTiffTiler:
    @pytest.fixture()
    def tiler(self):
        try:
            with ArgosTiffTiler(argos_z_file_path) as tiler:
                yield tiler
        except FileNotFoundError:
            pytest.skip("Argos stacked avs test file not found, skipping")

    def test_number_of_level_images(self, tiler: ArgosTiffTiler):
        # Arrange

        # Act
        levels = tiler.levels

        # Assert
        # 5 planes (MinZ=-2..MaxZ=2) x 8 pyramid levels.
        assert len(levels) == 40

    def test_base_focal_planes(self, tiler: ArgosTiffTiler):
        # Arrange

        # Act
        base_planes = sorted(
            level.focal_plane for level in tiler.levels if level.pyramid_index == 0
        )

        # Assert
        # ZRange=8 um between planes, symmetric about the center plane.
        assert base_planes == [-16.0, -8.0, 0.0, 8.0, 16.0]

    def test_associated_images(self, tiler: ArgosTiffTiler):
        # Arrange

        # Act
        thumbnails = tiler.thumbnails
        overviews = tiler.overviews

        # Assert
        assert len(thumbnails) == 1
        assert len(overviews) == 1
