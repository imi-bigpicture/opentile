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

"""Image implementation for Motic tiff files.

The associated images (thumbnail, label, macro) use the same Aperio jpeg/lzw striped
layout as svs, so the svs image classes are reused for them by the tiler; only the
natively tiled level needs its own implementation here.
"""

from typing import Optional

from tifffile import COMPRESSION, TiffPage

from opentile.file import OpenTileFile
from opentile.geometry import Size, SizeMm
from opentile.tiff_image import LevelTiffImage, NativeTiledTiffImage


class MoticTiffImage(NativeTiledTiffImage, LevelTiffImage):
    def __init__(
        self, page: TiffPage, file: OpenTileFile, base_size: Size, base_mpp: SizeMm
    ):
        """Level image for a Motic tiff file. The level is natively tiled (jpeg), so
        the tiles are served as-is.

        Parameters
        ----------
        page: TiffPage
            TiffPage defining the level.
        file: OpenTileFile
            File to read data from.
        base_size: Size
            Size of the base level in the pyramid.
        base_mpp: SizeMm
            Pixel spacing (um/pixel) of the base level in the pyramid.
        """
        super().__init__(page, file)
        self._base_size = base_size
        self._base_mpp = base_mpp
        self._scale = self._calculate_scale(base_size)
        self._pyramid_index = self._calculate_pyramidal_index(self._scale)
        self._mpp = self._calculate_mpp(base_mpp, self._scale)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._page}, {self._file}, "
            f"{self._base_size}, {self._base_mpp})"
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
