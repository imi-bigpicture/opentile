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

from functools import cached_property
from typing import Optional

import numpy as np
from tifffile import COMPRESSION, TiffPage

from opentile.file import OpenTileFile
from opentile.formats.ndpi.ndpi_tiler import NdpiOneFrameImage
from opentile.geometry import Point, Region, Size, SizeMm
from opentile.jpeg import Jpeg
from opentile.tiff_image import (
    AssociatedTiffImage,
    BaseTiffImage,
    LevelTiffImage,
    NativeTiledTiffImage,
    ThumbnailTiffImage,
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
    def supported_compressions(self) -> Optional[list[COMPRESSION]]:
        return None

    def get_tile(self, tile_position: tuple[int, int]) -> bytes:
        if tile_position != (0, 0):
            raise ValueError("Non-tiled image, expected tile_position (0, 0)")
        return self._read_frame(0)

    def get_decoded_tile(self, tile_position: tuple[int, int]) -> np.ndarray:
        if tile_position != (0, 0):
            raise ValueError("Non-tiled image, expected tile_position (0, 0)")
        return self._page.asarray(squeeze=True)


class OmeTiffAssociatedImage(OmeTiffImage, AssociatedTiffImage):
    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._page}, {self._file}, {self._base_mpp})"

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
        focal_plane: float,
        optical_path: str,
        jpeg: Jpeg,
    ):
        super().__init__(page, file, base_size, tile_size, jpeg)
        self._base_mpp = base_mpp
        self._scale = self._calculate_scale(base_size)
        self._pyramid_index = self._calculate_pyramidal_index(self._scale)
        self._mpp = self._calculate_mpp(base_mpp, self._scale)
        self._jpeg = jpeg
        self._focal_plane = focal_plane
        self._optical_path = optical_path

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._page}, {self._file}, {self._base_size}, "
            f"{self._tile_size}, {self._base_mpp}, {self._focal_plane}, "
            f"{self._optical_path}, {self._jpeg})"
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

    @property
    def focal_plane(self) -> float:
        return self._focal_plane

    @property
    def optical_path(self) -> str:
        return self._optical_path


class OmeTiffStripedImage(BaseTiffImage, LevelTiffImage):
    """OME tiff level stored as strips rather than tiles (e.g. uncompressed exports).
    The page is decoded once and served as a tile grid. Since the data is not natively
    tiled and (for uncompressed pages) has no per-tile encoded representation,
    `get_tile` returns the raw pixel bytes of the cropped region."""

    def __init__(
        self,
        page: TiffPage,
        file: OpenTileFile,
        base_size: Size,
        tile_size: Size,
        base_mpp: SizeMm,
        focal_plane: float,
        optical_path: str,
    ):
        super().__init__(page, file)
        # Override the untiled default (tile size == image size) with a real grid.
        self._tile_size = tile_size
        self._tiled_region = Region(position=Point(0, 0), size=self.tiled_size)
        self._image_size = Size(self._page.imagewidth, self._page.imagelength)
        self._base_size = base_size
        self._base_mpp = base_mpp
        self._scale = self._calculate_scale(base_size)
        self._pyramid_index = self._calculate_pyramidal_index(self._scale)
        self._mpp = self._calculate_mpp(base_mpp, self._scale)
        self._focal_plane = focal_plane
        self._optical_path = optical_path

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._page}, {self._file}, {self._base_size}, "
            f"{self._tile_size}, {self._base_mpp}, {self._focal_plane}, "
            f"{self._optical_path})"
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
    def focal_plane(self) -> float:
        return self._focal_plane

    @property
    def optical_path(self) -> str:
        return self._optical_path

    @cached_property
    def _decoded_image(self) -> np.ndarray:
        """The whole page decoded once (tifffile reassembles the strips)."""
        return self._page.asarray(squeeze=True)

    def get_decoded_tile(self, tile_position: tuple[int, int]) -> np.ndarray:
        point = Point.from_tuple(tile_position)
        if not self._check_if_tile_inside_image(point):
            raise ValueError(f"Tile {tile_position} is outside {self.tiled_size}.")
        left = point.x * self.tile_size.width
        top = point.y * self.tile_size.height
        region = self._decoded_image[
            top : top + self.tile_size.height, left : left + self.tile_size.width
        ]
        pad_height = self.tile_size.height - region.shape[0]
        pad_width = self.tile_size.width - region.shape[1]
        if pad_height or pad_width:
            padding = [(0, pad_height), (0, pad_width)] + [(0, 0)] * (region.ndim - 2)
            region = np.pad(region, padding, constant_values=self.fill_value)
        return region

    def get_tile(self, tile_position: tuple[int, int]) -> bytes:
        # Compression is NONE, so the "encoded" tile is just the raw pixel bytes.
        return self.get_decoded_tile(tile_position).tobytes()


class OmeTiffTiledImage(NativeTiledTiffImage, LevelTiffImage):
    def __init__(
        self,
        page: TiffPage,
        file: OpenTileFile,
        base_size: Size,
        base_mpp: SizeMm,
        focal_plane: float,
        optical_path: str,
    ):
        super().__init__(page, file)
        self._image_size = Size(self._page.imagewidth, self._page.imagelength)
        self._base_size = base_size
        self._base_mpp = base_mpp
        self._scale = self._calculate_scale(base_size)
        self._pyramid_index = self._calculate_pyramidal_index(self.scale)
        self._mpp = self._calculate_mpp(base_mpp, self._scale)
        self._focal_plane = focal_plane
        self._optical_path = optical_path

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._page}, {self._file}, "
            f"{self._base_size}, {self._base_mpp}, {self._focal_plane}, "
            f"{self._optical_path})"
        )

    @property
    def mpp(self) -> SizeMm:
        return self._mpp

    @property
    def pixel_spacing(self) -> SizeMm:
        return self.mpp / 1000

    @property
    def supported_compressions(self) -> Optional[list[COMPRESSION]]:
        return None

    @property
    def scale(self) -> float:
        return self._scale

    @property
    def pyramid_index(self) -> int:
        return self._pyramid_index

    @property
    def focal_plane(self) -> float:
        return self._focal_plane

    @property
    def optical_path(self) -> str:
        return self._optical_path
