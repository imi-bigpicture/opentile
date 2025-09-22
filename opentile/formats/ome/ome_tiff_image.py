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

"""Image implementation for OME tiff files."""

from typing import List, Optional, Tuple

import numpy as np
from tifffile import COMPRESSION, TiffPage

from opentile.file import OpenTileFile
from opentile.formats.ndpi.ndpi_tiler import NdpiOneFrameImage
from opentile.geometry import Size, SizeMm
from opentile.jpeg import Jpeg
from opentile.tiff_image import (
    AssociatedTiffImage,
    LevelTiffImage,
    ThumbnailTiffImage,
    NativeTiledTiffImage,
    BaseTiffImage,
)


class OmeTiffImage(BaseTiffImage):
    def __init__(
        self,
        page: TiffPage,
        file: OpenTileFile,
        base_mpp: Optional[SizeMm] = None,
    ):
        super().__init__(page, file)
        self._base_mpp = base_mpp
        self._mpp = base_mpp

    @property
    def supported_compressions(self) -> Optional[List[COMPRESSION]]:
        return None

    def get_tile(self, tile_position: Tuple[int, int]) -> bytes:
        if tile_position != (0, 0):
            raise ValueError("Non-tiled image, expected tile_position (0, 0)")
        return self._read_frame(0)

    def get_decoded_tile(self, tile_position: Tuple[int, int]) -> np.ndarray:
        if tile_position != (0, 0):
            raise ValueError("Non-tiled image, expected tile_position (0, 0)")
        return self._page.asarray(squeeze=True, lock=self._file.lock)


class OmeTiffAssociatedImage(OmeTiffImage, AssociatedTiffImage):
    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._page}, {self._file}, " f"{self._base_mpp})"
        )

    @property
    def mpp(self) -> Optional[SizeMm]:
        return self._mpp

    @property
    def pixel_spacing(self) -> Optional[SizeMm]:
        if self.mpp is None:
            return None
        return self.mpp / 1000


class OmeTiffThumbnailImage(OmeTiffImage, ThumbnailTiffImage):
    def __init__(
        self, page: TiffPage, file: OpenTileFile, base_size: Size, base_mpp: SizeMm
    ):
        super().__init__(
            page,
            file,
        )
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
    def mpp(self) -> SizeMm:
        return self._mpp

    @property
    def pixel_spacing(self) -> SizeMm:
        return self.mpp / 1000

    @property
    def scale(self) -> float:
        return self._scale


class OmeTiffOneFrameImage(NdpiOneFrameImage, LevelTiffImage):
    """Some ome tiff files have levels that are not tiled, similar to ndpi.
    Not sure if this is something worth supporting yet, and if so should either
    refactor the ndpi-classes to separate out the ndpi-specific metadata
    processing or make a new metadata processing class."""

    def __init__(
        self,
        page: TiffPage,
        file: OpenTileFile,
        base_size: Size,
        tile_size: Size,
        base_mpp: SizeMm,
        jpeg: Jpeg,
    ):
        super().__init__(page, file, base_size, tile_size, jpeg)
        self._base_mpp = base_mpp
        self._scale = self._calculate_scale(base_size)
        self._pyramid_index = self._calculate_pyramidal_index(self._scale)
        self._mpp = self._calculate_mpp(base_mpp, self._scale)
        self._jpeg = jpeg

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._page}, {self._file}, "
            f"{self._base_size}, {self._tile_size}, {self._base_mpp}, {self._jpeg})"
        )

    @property
    def scale(self) -> float:
        return self._scale

    @property
    def pyramid_index(self) -> int:
        return self._pyramid_index

    @property
    def pixel_spacing(self) -> SizeMm:
        return self.mpp / 1000


class OmeTiffTiledImage(NativeTiledTiffImage, LevelTiffImage):
    def __init__(
        self,
        page: TiffPage,
        file: OpenTileFile,
        base_size: Size,
        base_mpp: SizeMm,
    ):
        super().__init__(page, file)
        self._image_size = Size(self._page.imagewidth, self._page.imagelength)
        self._base_size = base_size
        self._base_mpp = base_mpp
        self._scale = self._calculate_scale(base_size)
        self._pyramid_index = self._calculate_pyramidal_index(self.scale)
        self._mpp = self._calculate_mpp(base_mpp, self._scale)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._page}, {self._file}, "
            f"{self._base_size}, {self._base_mpp})"
        )

    @property
    def mpp(self) -> SizeMm:
        return self._mpp

    @property
    def pixel_spacing(self) -> SizeMm:
        return self.mpp / 1000

    @property
    def supported_compressions(self) -> Optional[List[COMPRESSION]]:
        return None

    @property
    def scale(self) -> float:
        return self._scale

    @property
    def pyramid_index(self) -> int:
        return self._pyramid_index
