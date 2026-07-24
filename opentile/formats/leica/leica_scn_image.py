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

"""Image implementations for Leica SCN files: a plain tiled JPEG pyramid level and the
tiled macro image served as an associated overview."""

from typing import Optional

from tifffile import COMPRESSION, TiffPage

from opentile.file import OpenTileFile
from opentile.geometry import Point, Region, Size, SizeMm
from opentile.tiff_image import (
    AssociatedTiffImage,
    LevelTiffImage,
    NativeTiledTiffImage,
)


class LeicaScnLevelTiffImage(NativeTiledTiffImage, LevelTiffImage):
    """Level image for a Leica SCN main image: a plain tiled JPEG pyramid whose mpp is
    scaled from the base level parsed from the SCN XML."""

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


class LeicaScnAssociatedImage(NativeTiledTiffImage, AssociatedTiffImage):
    """The Leica SCN macro image (highest-resolution dimension), tiled JPEG."""

    @property
    def pixel_spacing(self) -> Optional[SizeMm]:
        return None

    @property
    def supported_compressions(self) -> Optional[list[COMPRESSION]]:
        return None


class LeicaScnLabelImage(NativeTiledTiffImage, AssociatedTiffImage):
    """Label cropped from the lower end of the tiled macro image.

    Leica SCN has no dedicated label image, but the macro spans the whole slide and so
    images the label at its far end. Unlike the single-frame macros of ndpi and argos,
    which can be cropped losslessly to an exact pixel column, this macro is tiled: the
    crop is therefore aligned down to a whole tile row and the tiles are served
    unchanged. Aligning down means the label is never clipped, at the cost of some
    slide margin above it.
    """

    def __init__(self, page: TiffPage, file: OpenTileFile, crop_position: float):
        """
        Parameters
        ----------
        page: TiffPage
            The macro page to crop the label from.
        file: OpenTileFile
            File to read data from.
        crop_position: float
            Top edge of the label as a fraction of the macro height; the label spans
            from here to the bottom. Rounded down to a whole tile row.
        """
        super().__init__(page, file)
        self._tile_row_offset = int(page.imagelength * crop_position) // page.tilelength
        self._image_size = Size(
            page.imagewidth,
            page.imagelength - self._tile_row_offset * page.tilelength,
        )
        # The tiled region is computed from the image size in the base class, so it has
        # to be rebuilt for the cropped size.
        self._tiled_region = Region(position=Point(0, 0), size=self.tiled_size)

    @property
    def pixel_spacing(self) -> Optional[SizeMm]:
        return None

    @property
    def supported_compressions(self) -> Optional[list[COMPRESSION]]:
        return None

    def _tile_point_to_frame_index(self, tile_point: Point) -> int:
        """Return the frame index in the uncropped macro for a tile of the label."""
        return (
            tile_point.y + self._tile_row_offset
        ) * self.tiled_size.width + tile_point.x
