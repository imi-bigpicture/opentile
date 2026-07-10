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

"""Associated image implementations for Ventana bif files. Overlapping (raw) level
images use the shared opentile.tiff_image.OverlappingLevelTiffImage; already-stitched
Ventana tiff files use the plain VentanaLevelTiffImage below."""

from typing import Optional

from tifffile import COMPRESSION, TiffPage

from opentile.file import OpenTileFile
from opentile.geometry import Size, SizeMm
from opentile.tiff_image import (
    AssociatedTiffImage,
    LevelTiffImage,
    NativeTiledTiffImage,
    ThumbnailTiffImage,
)


class VentanaLevelTiffImage(NativeTiledTiffImage, LevelTiffImage):
    """Level image for an already-stitched (non-overlapping) Ventana tiff file, whose
    tiles abut like any plain tiled pyramid."""

    def __init__(
        self, page: TiffPage, file: OpenTileFile, base_size: Size, base_mpp: SizeMm
    ):
        super().__init__(page, file)
        self._base_mpp = base_mpp
        self._scale = self._calculate_scale(base_size)
        self._pyramid_index = self._calculate_pyramidal_index(self._scale)
        self._mpp = self._calculate_mpp(base_mpp, self._scale)

    @property
    def pixel_spacing(self) -> SizeMm:
        return self._mpp / 1000

    @property
    def mpp(self) -> SizeMm:
        return self._mpp

    @property
    def scale(self) -> float:
        return self._scale

    @property
    def pyramid_index(self) -> int:
        return self._pyramid_index

    @property
    def supported_compressions(self) -> Optional[list[COMPRESSION]]:
        return None


class VentanaAssociatedTiffImage(NativeTiledTiffImage, AssociatedTiffImage):
    @property
    def pixel_spacing(self) -> Optional[SizeMm]:
        return None

    @property
    def supported_compressions(self) -> Optional[list[COMPRESSION]]:
        return None


class VentanaThumbnailTiffImage(NativeTiledTiffImage, ThumbnailTiffImage):
    def __init__(
        self,
        page: TiffPage,
        file: OpenTileFile,
        base_size: Size,
        base_mpp: SizeMm,
    ):
        super().__init__(page, file)
        self._base_size = base_size
        self._base_mpp = base_mpp
        self._scale = self._calculate_scale(base_size)
        self._mpp = self._calculate_mpp(base_mpp, self._scale)

    @property
    def scale(self) -> float:
        return self._scale

    @property
    def pixel_spacing(self) -> SizeMm:
        return self._mpp / 1000

    @property
    def supported_compressions(self) -> Optional[list[COMPRESSION]]:
        return None
