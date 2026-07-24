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

"""Image implementation for Leica SCN files: the label cropped from the macro."""

from typing import Optional

from tifffile import COMPRESSION, TiffPage

from opentile.file import OpenTileFile
from opentile.geometry import Point, Region, Size, SizeMm
from opentile.tiff_image import AssociatedTiffImage
from opentile.tiff_image_bases import NativeTiledTiffImage


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
