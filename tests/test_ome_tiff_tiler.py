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
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import tifffile
from decoy import Decoy
from tifffile import COMPRESSION, PHOTOMETRIC, TiffPage

from opentile.formats import OmeTiffTiler
from opentile.formats.ome.ome_tiff_image import OmeTiffStripedImage
from opentile.geometry import Size, SizeMm
from opentile.tiff_image import LevelTiffImage

from .filepaths import ome_tiff_file_path


@pytest.fixture()
def tiler():
    try:
        with OmeTiffTiler(ome_tiff_file_path) as tiler:
            yield tiler
    except FileNotFoundError:
        pytest.skip("Ome tiff test file not found, skipping")


@pytest.fixture(scope="session")
def striped_ome_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    # A generated 3-plane, strip-stored (uncompressed) OME z-stack, small enough to
    # write on the fly so the strip-stored path is covered without a large fixture file.
    path = tmp_path_factory.mktemp("ome") / "zstack.ome.tiff"
    # distinct value per plane so tiles from different focal planes differ
    data = np.zeros((3, 600, 700, 3), dtype=np.uint8)
    for z in range(3):
        data[z] = z * 40
    tifffile.imwrite(
        path,
        data,
        photometric="rgb",
        compression=None,  # uncompressed -> strip-stored
        rowsperstrip=64,
        metadata={
            "axes": "ZYXS",
            "PhysicalSizeX": 0.5,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": 0.5,
            "PhysicalSizeYUnit": "µm",
            "PhysicalSizeZ": 2.0,
            "PhysicalSizeZUnit": "µm",
        },
    )
    return path


@pytest.fixture()
def striped_ome_tiler(striped_ome_path: Path):
    with OmeTiffTiler(striped_ome_path) as tiler:
        yield tiler


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
    def test_get_tile(
        self, level: LevelTiffImage, tile_point: tuple[int, int], hash: str
    ):
        # Arrange

        # Act
        tile = level.get_tile(tile_point)

        # Assert
        assert md5(tile).hexdigest() == hash

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

    def test_sumples_per_pixel(self, level: LevelTiffImage):
        # Arrange

        # Act
        samples_per_pixel = level.samples_per_pixel

        # Assert
        assert samples_per_pixel == 3

    def test_compressed_size(self, level: LevelTiffImage):
        # Arrange

        # Act
        compressed_size = level.compressed_size

        # Assert
        assert compressed_size == 104115549

    @pytest.mark.parametrize(
        ["level", "expected_size"],
        [
            (0, SizeMm(0.000499, 0.000499)),
            (1, SizeMm(0.001996, 0.001996)),
            (2, SizeMm(0.007984, 0.007984)),
        ],
    )
    def test_pixel_spacing(
        self, tiler: OmeTiffTiler, level: int, expected_size: SizeMm
    ):
        # Arrange
        base_level = tiler.get_level(level)

        # Act
        base_pixel_spacing = base_level.pixel_spacing

        # Assert
        assert base_pixel_spacing == expected_size


class TestOmeTiffStripedImage:
    """Unit tests for the strip-stored (uncompressed) OME level image, exercised
    without a fixture file by injecting a decoded array."""

    def _make_image(self, decoy: Decoy, image: np.ndarray) -> OmeTiffStripedImage:
        page = decoy.mock(cls=TiffPage)
        decoy.when(page.photometric).then_return(PHOTOMETRIC.RGB)
        decoy.when(page.bitspersample).then_return(8)
        decoy.when(page.compression).then_return(COMPRESSION.NONE)
        striped = OmeTiffStripedImage.__new__(OmeTiffStripedImage)
        striped._page = page
        striped._image_size = Size(image.shape[1], image.shape[0])
        striped._tile_size = Size(8, 8)
        striped._decoded_image = image  # bypass the cached asarray
        return striped

    def test_full_tile_returns_region(self, decoy: Decoy) -> None:
        # Arrange
        image = np.arange(10 * 10 * 3, dtype=np.uint8).reshape(10, 10, 3)
        striped = self._make_image(decoy, image)

        # Act
        tile = striped.get_decoded_tile((0, 0))

        # Assert
        assert tile.shape == (8, 8, 3)
        assert np.array_equal(tile, image[0:8, 0:8])

    def test_edge_tile_is_padded_with_fill_value(self, decoy: Decoy) -> None:
        # Arrange
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        striped = self._make_image(decoy, image)

        # Act
        tile = striped.get_decoded_tile((1, 1))  # only [8:10, 8:10] is real

        # Assert
        assert tile.shape == (8, 8, 3)
        assert np.array_equal(tile[0:2, 0:2], image[8:10, 8:10])
        assert np.all(tile[2:, :] == 255)  # RGB uint8 fill is white
        assert np.all(tile[:, 2:] == 255)

    def test_get_tile_is_raw_bytes_of_decoded(self, decoy: Decoy) -> None:
        # Arrange
        image = np.arange(10 * 10 * 3, dtype=np.uint8).reshape(10, 10, 3)
        striped = self._make_image(decoy, image)

        # Act
        raw = striped.get_tile((1, 1))

        # Assert
        assert raw == striped.get_decoded_tile((1, 1)).tobytes()

    def test_tile_outside_image_raises(self, decoy: Decoy) -> None:
        # Arrange
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        striped = self._make_image(decoy, image)

        # Act, Assert
        with pytest.raises(ValueError):
            striped.get_decoded_tile((2, 0))  # tiled_size is (2, 2)


@pytest.mark.unittest
class TestOmeTiffStripedTiler:
    """Integration tests against a generated strip-stored (uncompressed) OME z-stack."""

    def test_levels_are_striped_images(self, striped_ome_tiler: OmeTiffTiler):
        # Arrange

        # Act
        levels = striped_ome_tiler.levels

        # Assert
        assert all(isinstance(level, OmeTiffStripedImage) for level in levels)

    def test_focal_planes(self, striped_ome_tiler: OmeTiffTiler):
        # Arrange

        # Act
        focal_planes = sorted({level.focal_plane for level in striped_ome_tiler.levels})

        # Assert: one focal plane per z, spaced by the 2.0 um physical z size
        assert focal_planes == [0.0, 2.0, 4.0]

    def test_single_optical_path(self, striped_ome_tiler: OmeTiffTiler):
        # Arrange

        # Act
        optical_paths = {level.optical_path for level in striped_ome_tiler.levels}

        # Assert: rgb is folded into the sample axis, so there is one optical path
        assert optical_paths == {"0"}

    def test_mpp(self, striped_ome_tiler: OmeTiffTiler):
        # Arrange
        level = cast(OmeTiffStripedImage, striped_ome_tiler.get_level(0))

        # Act
        mpp = level.mpp

        # Assert
        assert mpp == SizeMm(0.5, 0.5)

    def test_get_decoded_tile_shape(self, striped_ome_tiler: OmeTiffTiler):
        # Arrange
        level = striped_ome_tiler.get_level(0)

        # Act
        tile = level.get_decoded_tile((0, 0))

        # Assert
        assert tile.shape == (
            level.tile_size.height,
            level.tile_size.width,
            level.samples_per_pixel,
        )

    def test_get_tile_is_raw_bytes_of_decoded(self, striped_ome_tiler: OmeTiffTiler):
        # Arrange
        level = striped_ome_tiler.get_level(0)

        # Act, Assert
        assert level.get_tile((0, 0)) == level.get_decoded_tile((0, 0)).tobytes()
