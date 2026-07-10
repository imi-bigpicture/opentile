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

"""Placement of overlapping source tiles onto a de-overlapped canvas."""

import math
from dataclasses import dataclass

from opentile.geometry import Point, PointF, Region, Size


@dataclass(frozen=True)
class TilePlacement:
    """One piece a stored source tile contributes to the de-overlapped canvas."""

    source: Region
    """The region within the decoded source tile that is placed."""
    position: PointF
    """The (sub-pixel) top-left the region occupies on the de-overlapped canvas."""


@dataclass(frozen=True)
class TileOverlap:
    """Placement of overlapping source tiles onto a de-overlapped canvas.

    Some formats (Trestle, Ventana) store tiles that overlap their neighbours.

    A stored source tile may contribute more than one piece: a plain tile contributes
    its whole self, while a tile that packs several downsampled tiles contributes one
    piece per packed sub-tile.
    """

    image_size: Size
    """The composed (de-overlapped) level size."""
    placements: dict[Point, list[TilePlacement]]
    """Maps each stored source-tile grid position to the pieces it contributes."""

    @classmethod
    def from_regular_grid(
        cls, mosaic_size: Size, tile_size: Size, overlap: Size
    ) -> "TileOverlap":
        """Build placement for a regular tile grid where every tile overlaps its
        neighbour by a constant amount, contributing its whole self.

        Parameters
        ----------
        mosaic_size: Size
            Size of the overlapping tile mosaic as stored (the tiff page size),
            larger than the resulting de-overlapped `image_size`.
        tile_size: Size
            Size of a source tile.
        overlap: Size
            Pixels each tile overlaps its right/bottom neighbour.
        """
        tiles_across = math.ceil(mosaic_size.width / tile_size.width)
        tiles_down = math.ceil(mosaic_size.height / tile_size.height)
        step_x = tile_size.width - overlap.width
        step_y = tile_size.height - overlap.height
        image_size = Size(
            mosaic_size.width - (tiles_across - 1) * overlap.width,
            mosaic_size.height - (tiles_down - 1) * overlap.height,
        )
        whole_tile = Region(Point(0, 0), tile_size)
        placements = {
            Point(x, y): [TilePlacement(whole_tile, PointF(x * step_x, y * step_y))]
            for y in range(tiles_down)
            for x in range(tiles_across)
        }
        return cls(image_size, placements)
