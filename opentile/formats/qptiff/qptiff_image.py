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

"""Image implementations for qptiff files.

qptiff levels are a plain tiled pyramid (512 x 512 tiles). Levels of 2K x 2K pixels or
smaller are stored as strips instead of tiles, as are the thumbnail, macro and label
images; those are served as a single tile covering the whole image.
"""

from typing import Optional

from tifffile import COMPRESSION, TiffPage

from opentile.file import OpenTileFile
from opentile.geometry import Size, SizeMm
from opentile.jpeg import Jpeg
from opentile.tiff_image import (
    LevelTiffImage,
    NativeTiledTiffImage,
    StripedAssociatedImage,
    StripedThumbnailImage,
    StripedTiffImage,
)


class QptiffLevelImage(NativeTiledTiffImage, LevelTiffImage):
    """Tiled pyramid level."""

    def __init__(
        self,
        page: TiffPage,
        file: OpenTileFile,
        base_size: Size,
        base_mpp: SizeMm,
        optical_path: str = "1",
    ):
        """
        Parameters
        ----------
        page: TiffPage
            TiffPage defining the page.
        file: OpenTileFile
            File to read data from.
        base_size: Size
            Size of base level in pyramid.
        base_mpp: SizeMm
            Mpp (um/pixel) for base level in pyramid.
        optical_path: str = "1"
            Identifier of the band this level holds.
        """
        super().__init__(page, file)
        self._base_size = base_size
        self._base_mpp = base_mpp
        self._scale = self._calculate_scale(base_size)
        self._pyramid_index = self._calculate_pyramidal_index(self._scale)
        self._mpp = self._calculate_mpp(base_mpp, self._scale)
        self._optical_path = optical_path

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._page}, {self._file}, "
            f"{self._base_size}, {self._base_mpp}, {self._optical_path})"
        )

    @property
    def supported_compressions(self) -> Optional[list[COMPRESSION]]:
        return None

    @property
    def mpp(self) -> SizeMm:
        return self._mpp

    @property
    def pixel_spacing(self) -> SizeMm:
        return self.mpp / 1000

    @property
    def scale(self) -> float:
        return self._scale

    @property
    def pyramid_index(self) -> int:
        return self._pyramid_index

    @property
    def optical_path(self) -> str:
        return self._optical_path


class QptiffStripedImage(StripedTiffImage):
    """Non-tiled qptiff image, served as a single tile.

    qptiff stores these as JPEG or LZW strips. JPEG strips are concatenated into one
    scan by `StripedTiffImage`. LZW strips cannot be concatenated, as each strip is an
    independently terminated LZW stream, so they are decoded and served as raw bytes,
    reported as `COMPRESSION.NONE`.
    """

    @property
    def compression(self) -> COMPRESSION:
        if self._page.compression == COMPRESSION.LZW:
            return COMPRESSION.NONE
        return super().compression

    def get_tile(self, tile_position: tuple[int, int]) -> bytes:
        if self._page.compression != COMPRESSION.LZW:
            return super().get_tile(tile_position)
        return self.get_decoded_tile(tile_position).tobytes()


class QptiffStripedLevelImage(QptiffStripedImage, LevelTiffImage):
    """Pyramid level stored as strips (levels of 2K x 2K pixels or smaller)."""

    def __init__(
        self,
        page: TiffPage,
        file: OpenTileFile,
        base_size: Size,
        base_mpp: SizeMm,
        jpeg: Jpeg,
        optical_path: str = "1",
    ):
        """
        Parameters
        ----------
        page: TiffPage
            TiffPage defining the page.
        file: OpenTileFile
            File to read data from.
        base_size: Size
            Size of base level in pyramid.
        base_mpp: SizeMm
            Mpp (um/pixel) for base level in pyramid.
        jpeg: Jpeg
            Jpeg instance to use.
        optical_path: str = "1"
            Identifier of the band this level holds.
        """
        super().__init__(page, file, jpeg)
        self._base_size = base_size
        self._base_mpp = base_mpp
        self._scale = self._calculate_scale(base_size)
        self._pyramid_index = self._calculate_pyramidal_index(self._scale)
        self._mpp = self._calculate_mpp(base_mpp, self._scale)
        self._optical_path = optical_path

    @property
    def mpp(self) -> SizeMm:
        return self._mpp

    @property
    def pixel_spacing(self) -> SizeMm:
        return self.mpp / 1000

    @property
    def scale(self) -> float:
        return self._scale

    @property
    def pyramid_index(self) -> int:
        return self._pyramid_index

    @property
    def optical_path(self) -> str:
        return self._optical_path


class QptiffThumbnailImage(QptiffStripedImage, StripedThumbnailImage):
    """Striped thumbnail image (~500 x 500 RGB)."""


class QptiffAssociatedImage(QptiffStripedImage, StripedAssociatedImage):
    """Striped macro (overview) or label image."""
