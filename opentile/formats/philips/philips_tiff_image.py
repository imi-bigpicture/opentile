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

"""Image implementation for Philips tiff files."""

from typing import Optional

from tifffile import COMPRESSION, TiffPage

from opentile.file import OpenTileFile
from opentile.geometry import Size, SizeMm
from opentile.tiff_image import (
    AssociatedTiffImage,
    NativeTiledTiffImage,
    SparseTiledLevelImage,
    ThumbnailTiffImage,
)


class PhilipsAssociatedTiffImage(NativeTiledTiffImage, AssociatedTiffImage):
    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._page}, {self._file}"

    @property
    def pixel_spacing(self) -> Optional[SizeMm]:
        return None

    @property
    def supported_compressions(self) -> Optional[list[COMPRESSION]]:
        return None


class PhilipsThumbnailTiffImage(NativeTiledTiffImage, ThumbnailTiffImage):
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

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._page}, {self._file}, "
            f"{self._base_size}, {self._base_mpp})"
        )

    @property
    def scale(self) -> float:
        return self._scale

    @property
    def pixel_spacing(self) -> SizeMm:
        return self._mpp / 1000

    @property
    def supported_compressions(self) -> Optional[list[COMPRESSION]]:
        return None


class PhilipsLevelTiffImage(SparseTiledLevelImage):
    """Philips level image: a sparse JPEG-tiled pyramid level (see
    `SparseTiledLevelImage`)."""
