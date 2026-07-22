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

"""Image implementation for Huron tiff files."""

from typing import Optional

import numpy as np
from tifffile import COMPRESSION, TiffPage

from opentile.file import OpenTileFile
from opentile.geometry import Size, SizeMm
from opentile.tiff_image import (
    AssociatedTiffImage,
    BaseTiffImage,
    LevelTiffImage,
    NativeTiledTiffImage,
    ThumbnailTiffImage,
)


class HuronTiffImage(NativeTiledTiffImage, LevelTiffImage):
    def __init__(
        self, page: TiffPage, file: OpenTileFile, base_size: Size, base_mpp: SizeMm
    ):
        """Level image for a Huron tiff file. The level is natively tiled (JPEG 2000
        or jpeg), so the tiles are served as-is.

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


class HuronSingleFrameImage(BaseTiffImage):
    """A Huron image stored as an uncompressed, multi-strip frame (thumbnail, label,
    macro). The strips are read and assembled into one raw-pixel tile; there is no
    per-tile encoded representation to pass through, so it is served uncompressed."""

    @property
    def supported_compressions(self) -> Optional[list[COMPRESSION]]:
        # Only uncompressed pages: the tile is served as raw pixels, so the reported
        # (inherited) compression of NONE must match the served bytes.
        return [COMPRESSION.NONE]

    def get_decoded_tile(self, tile_position: tuple[int, int]) -> np.ndarray:
        if tile_position != (0, 0):
            raise ValueError("Non-tiled image, expected tile_position (0, 0)")
        return self._page.asarray(squeeze=True)

    def get_tile(self, tile_position: tuple[int, int]) -> bytes:
        return self.get_decoded_tile(tile_position).tobytes()


class HuronAssociatedImage(HuronSingleFrameImage, AssociatedTiffImage):
    """Associated image (label or macro) of a Huron tiff file."""

    @property
    def pixel_spacing(self) -> Optional[SizeMm]:
        return None


class HuronThumbnailImage(HuronSingleFrameImage, ThumbnailTiffImage):
    def __init__(
        self, page: TiffPage, file: OpenTileFile, base_size: Size, base_mpp: SizeMm
    ):
        """Thumbnail image of a Huron tiff file.

        Parameters
        ----------
        page: TiffPage
            TiffPage defining the thumbnail.
        file: OpenTileFile
            File to read data from.
        base_size: Size
            Size of the base level in the pyramid.
        base_mpp: SizeMm
            Pixel spacing (um/pixel) of the base level in the pyramid.
        """
        super().__init__(page, file)
        self._base_size = base_size
        self._scale = self._calculate_scale(base_size)
        self._mpp = self._calculate_mpp(base_mpp, self._scale)

    @property
    def scale(self) -> float:
        return self._scale

    @property
    def pixel_spacing(self) -> SizeMm:
        return self._mpp / 1000
