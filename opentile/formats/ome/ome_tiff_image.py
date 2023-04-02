#    Copyright 2022 SECTRA AB
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
from tifffile.tifffile import COMPRESSION, TiffPage

from opentile.formats.ndpi.ndpi_tiler import NdpiOneFrameImage
from opentile.geometry import Size, SizeMm
from opentile.jpeg import Jpeg
from opentile.tiff_image import LockableFileHandle, NativeTiledTiffImage, TiffImage


class OmeTiffImage(TiffImage):
    def __init__(
        self,
        page: TiffPage,
        fh: LockableFileHandle,
        base_mpp: Optional[SizeMm] = None,
    ):
        super().__init__(page, fh)
        self._base_mpp = base_mpp
        self._mpp = base_mpp

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._page}, {self._fh}, " f"{self._base_mpp})"

    @property
    def mpp(self) -> Optional[SizeMm]:
        return self._mpp

    @property
    def pixel_spacing(self) -> Optional[SizeMm]:
        """Return pixel spacing in mm per pixel."""
        if self.mpp is None:
            return None
        return self.mpp / 1000.0

    @property
    def supported_compressions(self) -> Optional[List[COMPRESSION]]:
        return None

    def get_tile(self, tile_position: Tuple[int, int]) -> bytes:
        if tile_position != (0, 0):
            raise ValueError("Non-tiled image, expected tile_position (0, 0)")
        return self._read_frame(0)

    def get_decoded_tile(self, tile_position: Tuple[int, int]) -> np.ndarray:
        frame = self.get_tile(tile_position)
        data, _, shape = self.page.decode(frame, 0)
        assert isinstance(data, np.ndarray)
        data.shape = shape[1:]
        return data


class OmeTiffOneFrameImage(NdpiOneFrameImage):
    """Some ome tiff files have levels that are not tiled, similar to ndpi.
    Not sure if this is something worth supporting yet, and if so should either
    refactor the ndpi-classes to separate out the ndpi-specific metadata
    processing or make a new metadata processing class."""

    def __init__(
        self,
        page: TiffPage,
        fh: LockableFileHandle,
        base_size: Size,
        tile_size: Size,
        base_mpp: SizeMm,
        jpeg: Jpeg,
    ):
        super().__init__(page, fh, base_size, tile_size, jpeg)
        self._pyramid_index = self._calculate_pyramidal_index(base_size)
        self._mpp = self._calculate_mpp(base_mpp)
        self._jpeg = jpeg

    @property
    def supported_compressions(self) -> Optional[List[COMPRESSION]]:
        return [COMPRESSION.JPEG]

    @property
    def pyramid_index(self) -> int:
        return self._pyramid_index

    @property
    def pixel_spacing(self) -> SizeMm:
        return self.mpp / 1000.0


class OmeTiffTiledImage(NativeTiledTiffImage):
    def __init__(
        self,
        page: TiffPage,
        fh: LockableFileHandle,
        base_size: Size,
        base_mpp: SizeMm,
    ):
        super().__init__(page, fh)
        self._image_size = Size(self._page.imagewidth, self._page.imagelength)
        self._base_size = base_size
        self._base_mpp = base_mpp
        self._pyramid_index = self._calculate_pyramidal_index(base_size)
        self._mpp = self._calculate_mpp(base_mpp)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._page}, {self._fh}, "
            f"{self._base_size}, {self._base_mpp})"
        )

    @property
    def mpp(self) -> SizeMm:
        """Return pixel spacing in um per pixel."""
        return self._mpp

    @property
    def pixel_spacing(self) -> SizeMm:
        """Return pixel spacing in mm per pixel."""
        return self.mpp / 1000.0

    @property
    def supported_compressions(self) -> Optional[List[COMPRESSION]]:
        return None

    @property
    def pyramid_index(self) -> int:
        return self._pyramid_index
