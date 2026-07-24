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

"""Tiler for reading tiles from Ventana bif files.

Ventana stores a single-file pyramidal tiled BigTIFF whose tiles overlap. The base
level is de-overlapped by accumulating the per-boundary overlaps that `VentanaMetadata`
parses from the ``EncodeInfo`` XMP into absolute tile positions.

Reduced levels are downsamples of the raw overlapping mosaic and record no overlap of
their own: each stored tile packs a ``downsample x downsample`` block of downsampled
level-0 tiles, so `_packed_overlap` places each level-0 tile at its stitch position
scaled by ``1 / downsample``, cut from the sub-tile it occupies (see `TilePlacement`).
"""

import math
from pathlib import Path
from typing import Any, Optional, Union

from tifffile import TiffFile, TiffPage, TiffPageSeries
from upath import UPath

from opentile.exceptions import MissingAssociatedImageError
from opentile.file import OpenTileFile
from opentile.formats.ventana.ventana_tiff_image import (
    VentanaAssociatedTiffImage,
    VentanaLevelTiffImage,
    VentanaThumbnailTiffImage,
)
from opentile.formats.ventana.ventana_tiff_metadata import VentanaMetadata
from opentile.geometry import Point, PointF, Region, Size
from opentile.metadata import Metadata
from opentile.tiff_format import TiffFormat
from opentile.tiff_image import (
    AssociatedTiffImage,
    LevelTiffImage,
    OverlappingLevelTiffImage,
    ThumbnailTiffImage,
)
from opentile.tile_overlap import TileOverlap, TilePlacement
from opentile.tiler import Tiler


class VentanaTiffTiler(Tiler):
    def __init__(
        self,
        file: Union[str, Path, UPath, OpenTileFile],
        file_options: Optional[dict[str, Any]] = None,
    ):
        """Tiler for Ventana bif file.

        Parameters
        ----------
        file: Union[str, Path, UPath, OpenTileFile]
            Filepath to a Ventana bif file or an opened Ventana OpenTileFile.
        file_options: Optional[Dict[str, Any]] = None
            Options to pass to filesystem when opening file.
        """
        super().__init__(file, file_options)
        # An already-stitched Ventana tiff has no EncodeInfo/XMP on its level page; its
        # tiles abut and are served as a plain pyramid. A raw bif has the stitch XMP.
        self._stitched = "XMP" not in self._base_page.tags
        self._metadata = VentanaMetadata(self._metadata_page())
        self._base_mpp = self._metadata.mpp
        if self._stitched:
            self._base_composed_size = self._base_size
        else:
            self._base_overlap = self._create_base_overlap(
                self._base_page.tilewidth, self._base_page.tilelength
            )
            self._base_composed_size = self._base_overlap.image_size

    def _metadata_page(self) -> TiffPage:
        """The page carrying the iScan XMP: the level base page for a raw bif, or an
        associated page (the iScan travels on the label) for a stitched tiff."""
        if "XMP" in self._base_page.tags:
            return self._base_page
        for series in self._file.series:
            for page in series.pages:
                if isinstance(page, TiffPage) and "XMP" in page.tags:
                    return page
        return self._base_page

    @property
    def metadata(self) -> Metadata:
        return self._metadata

    @property
    def format(self) -> TiffFormat:
        return TiffFormat.VENTANA

    @classmethod
    def supported(cls, tiff_file: TiffFile) -> bool:
        return tiff_file.is_bif

    def _create_level(self, level: int, page: int = 0) -> LevelTiffImage:
        tiff_page = self._get_tiff_page(self._level_series_index, level, page)
        if self._stitched:
            return VentanaLevelTiffImage(
                tiff_page, self._file, self._base_size, self._base_mpp
            )
        if level == 0:
            return OverlappingLevelTiffImage(
                tiff_page, self._file, self._base_mpp, 1.0, self._base_overlap
            )
        downsample = round(self._base_page.imagewidth / tiff_page.imagewidth)
        overlap = self._create_downsampled_overlap(
            downsample, Size(tiff_page.tilewidth, tiff_page.tilelength)
        )
        return OverlappingLevelTiffImage(
            tiff_page, self._file, self._base_mpp, float(downsample), overlap
        )

    def _create_label(self, page: int = 0) -> AssociatedTiffImage:
        if self._label_series_index is None:
            raise MissingAssociatedImageError("No label series found in this file.")
        return VentanaAssociatedTiffImage(
            self._get_tiff_page(self._label_series_index, 0, page), self._file
        )

    def _create_overview(self, page: int = 0) -> AssociatedTiffImage:
        raise NotImplementedError("Ventana bif files have no overview image.")

    def _create_thumbnail(self, page: int = 0) -> ThumbnailTiffImage:
        if self._thumbnail_series_index is None:
            raise MissingAssociatedImageError(
                "No thumbnail series found in this file."
            )
        return VentanaThumbnailTiffImage(
            self._get_tiff_page(self._thumbnail_series_index, 0, page),
            self._file,
            self._base_composed_size,
            self._base_mpp,
        )

    def _is_level_series(self, series: TiffPageSeries) -> bool:
        return series.name == "Baseline"

    def _is_overview_series(self, series: TiffPageSeries) -> bool:
        return False

    def _is_label_series(self, series: TiffPageSeries) -> bool:
        return series.name == "Label"

    def _is_thumbnail_series(self, series: TiffPageSeries) -> bool:
        return series.name == "Thumbnail"

    def _create_base_overlap(self, tile_width: int, tile_height: int) -> TileOverlap:
        """Return the base-level placement: each stored tile placed whole at its
        de-overlapped position. Every scanned area is laid out on a shared canvas at its
        own origin, then all positions are shifted so the canvas starts at (0, 0)."""
        whole_tile = Region(Point(0, 0), Size(tile_width, tile_height))
        placements: dict[Point, list[TilePlacement]] = {}
        for area in self._metadata.areas:
            # Accumulate per-boundary pitches into separable positions within the area;
            # each axis has one more tile than boundary, so col_x/row_y span its grid.
            col_x = [0.0]
            for overlap in area.col_overlaps:
                col_x.append(col_x[-1] + (tile_width - overlap))
            row_y = [0.0]
            for overlap in area.row_overlaps:
                row_y.append(row_y[-1] + (tile_height - overlap))

            anchor_x = area.origin_col * tile_width
            anchor_y = area.origin_row * tile_height
            for row in range(area.num_rows):
                for col in range(area.num_cols):
                    grid = Point(area.origin_col + col, area.origin_row + row)
                    position = PointF(anchor_x + col_x[col], anchor_y + row_y[row])
                    placements[grid] = [TilePlacement(whole_tile, position)]

        # Shift every position so the composed canvas starts at the origin.
        min_x = min(p[0].position.x for p in placements.values())
        min_y = min(p[0].position.y for p in placements.values())
        max_x = max(p[0].position.x for p in placements.values())
        max_y = max(p[0].position.y for p in placements.values())
        placements = {
            grid: [
                TilePlacement(
                    piece.source,
                    PointF(piece.position.x - min_x, piece.position.y - min_y),
                )
                for piece in pieces
            ]
            for grid, pieces in placements.items()
        }
        image_size = Size(
            math.ceil(max_x - min_x) + tile_width,
            math.ceil(max_y - min_y) + tile_height,
        )
        return TileOverlap(image_size, placements)

    def _create_downsampled_overlap(
        self, downsample: int, tile_size: Size
    ) -> TileOverlap:
        """Reduced-level placement. Each stored tile of `tile_size` packs a
        ``downsample x downsample`` block of downsampled level-0 tiles; place each at
        its level-0 stitch position scaled by ``1 / downsample``, cut from the sub-tile
        it occupies within its stored tile."""
        subtile = Size(tile_size.width // downsample, tile_size.height // downsample)
        placements: dict[Point, list[TilePlacement]] = {}
        for level0_point, level0_placements in self._base_overlap.placements.items():
            base_position = level0_placements[0].position
            stored_tile = Point(
                level0_point.x // downsample, level0_point.y // downsample
            )
            source = Region(
                Point(
                    (level0_point.x % downsample) * subtile.width,
                    (level0_point.y % downsample) * subtile.height,
                ),
                subtile,
            )
            position = PointF(
                base_position.x / downsample, base_position.y / downsample
            )
            placements.setdefault(stored_tile, []).append(
                TilePlacement(source, position)
            )
        image_size = Size(
            round(self._base_overlap.image_size.width / downsample),
            round(self._base_overlap.image_size.height / downsample),
        )
        return TileOverlap(image_size, placements)
