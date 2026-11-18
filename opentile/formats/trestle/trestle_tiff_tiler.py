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

"""Tiler for reading tiles from Trestle tiff files.

Trestle stores a single-file pyramidal tiled TIFF where each level's tiles overlap
their neighbours. The per-level overlap is listed in the base page's
``OverlapsXY`` description field as ``x0 y0 x1 y1 ...`` (further levels 0).
"""

import math
from functools import cached_property
from pathlib import Path
from typing import Any, Optional, Union

from tifffile import TiffFile, TiffPage, TiffPageSeries
from upath import UPath

from opentile.exceptions import MissingAssociatedImageError
from opentile.file import OpenTileFile
from opentile.formats.trestle.trestle_image import (
    TrestleOverlappingImage,
    TrestleTiledImage,
)
from opentile.formats.trestle.trestle_tiff_metadata import TrestleMetadata
from opentile.geometry import PointF, Size
from opentile.metadata import Metadata
from opentile.tiff_format import TiffFormat
from opentile.tiff_image import AssociatedTiffImage, LevelTiffImage, ThumbnailTiffImage
from opentile.tile_overlap import TileOverlap
from opentile.tiler import Tiler


class TrestleTiffTiler(Tiler):
    def __init__(
        self,
        file: Union[str, Path, UPath, OpenTileFile],
        tile_size: int = 512,
        file_options: Optional[dict[str, Any]] = None,
    ):
        """Tiler for Trestle tiff file.

        A level stored without overlap at `tile_size` is served as stored. Other
        levels are re-gridded into non-overlapping `tile_size` tiles when that is
        lossless (baseline 8-bit JPEG with the tile step and `tile_size` multiples of
        the MCU), and otherwise served as stored with an `overlap` placement for the
        consumer to compose.

        Parameters
        ----------
        file: Union[str, Path, UPath, OpenTileFile]
            Filepath to a Trestle TiffFile or an opened Trestle OpenTileFile.
        tile_size: int = 512
            Size of the served tiles. Must be a multiple of the JPEG MCU (8 or 16)
            for levels to be re-gridded.
        file_options: Optional[Dict[str, Any]] = None
            Options to pass to filesystem when opening file.
        """
        super().__init__(file, file_options)
        self._tile_size = Size(tile_size, tile_size)
        self._metadata = TrestleMetadata(self._base_page)
        self._level_pages = [
            page
            for page in self._file.tiff.pages
            if isinstance(page, TiffPage) and page.is_tiled
        ]
        self._base_mpp = self._metadata.mpp
        self._base_composed_size = self._overlap_for(
            self._level_pages[0], self._metadata.level_overlap(0)
        ).image_size

    @cached_property
    def levels(self) -> list[LevelTiffImage]:
        return [self.get_level(level) for level in range(len(self._level_pages))]

    @property
    def metadata(self) -> Metadata:
        return self._metadata

    @property
    def format(self) -> TiffFormat:
        return TiffFormat.TRESTLE

    @classmethod
    def supported(cls, tiff_file: TiffFile) -> bool:
        software = tiff_file.pages.first.software
        return software.startswith("MedScan")

    def _create_level(self, level: int, page: int = 0) -> LevelTiffImage:
        tiff_page = self._level_pages[level]
        level_overlap = self._metadata.level_overlap(level)
        stored_tile_size = Size(tiff_page.tilewidth, tiff_page.tilelength)
        overlap = self._overlap_for(tiff_page, level_overlap)
        scale = self._level_scale(overlap.image_size)
        origin_offset = self._origin_offsets[level]
        if level_overlap == (0, 0) and stored_tile_size == self._tile_size:
            return TrestleTiledImage(
                tiff_page, self._file, self._base_mpp, scale, origin_offset
            )
        stored = TrestleOverlappingImage(
            tiff_page,
            self._file,
            self._base_mpp,
            scale,
            overlap,
            origin_offset,
        )
        # A level without overlap but at another tile size is also re-gridded (its
        # step is the stored tile size), or else served with a zero-overlap placement.
        return stored.regrid_if_possible(self._tile_size)

    def _level_scale(self, composed_size: Size) -> float:
        """The level's downsample: the power of two nearest the ratio of the base and
        level sizes. The sizes themselves do not give it exactly, as coarser levels
        are cropped (see `_origin_offsets`)."""
        ratio = self._base_composed_size.width / composed_size.width
        return float(2 ** round(math.log2(ratio)))

    @cached_property
    def _origin_offsets(self) -> list[PointF]:
        """Position of each level's first pixel on the base level, in base-level
        pixels.

        A level at downsample `scale` covers ``size * scale`` base-level pixels. Where
        that falls short of the base level by `scale` or more, the shortfall can not
        be size rounding: the scanner built the level from the base image cropped
        equally on both sides, so the level starts half the shortfall in (e.g. 64
        base-level pixels short, the base level's overlap, starts 32 in).

        A shortfall smaller than `scale` is ambiguous. It is taken as the crop of the
        finer levels when the level's size matches that crop (64 base pixels is only
        half a pixel at downsample 128), and otherwise as size rounding: a level
        sized by rounding ``base / scale`` starts at the base level's first pixel, so
        its offset is 0 rather than half the rounding.
        """
        base = self._base_composed_size
        sizes = [
            self._overlap_for(page, self._metadata.level_overlap(level)).image_size
            for level, page in enumerate(self._level_pages)
        ]
        scales = [self._level_scale(size) for size in sizes]
        offset_x = self._axis_origin_offsets(
            base.width, [size.width for size in sizes], scales
        )
        offset_y = self._axis_origin_offsets(
            base.height, [size.height for size in sizes], scales
        )
        return [PointF(x, y) for x, y in zip(offset_x, offset_y)]

    @staticmethod
    def _axis_origin_offsets(
        base: int, sizes: list[int], scales: list[float]
    ) -> list[float]:
        """Origin offsets along one axis for levels ordered from fine to coarse (see
        `_origin_offsets`)."""
        offsets: list[float] = []
        crop = 0.0
        for size, scale in zip(sizes, scales):
            shortfall = base - size * scale
            if abs(shortfall) >= scale:
                crop = shortfall
                offsets.append(crop / 2)
            elif crop != 0 and abs(base - crop - size * scale) < scale:
                offsets.append(crop / 2)
            else:
                offsets.append(0.0)
        return offsets

    def _create_label(self, page: int = 0) -> AssociatedTiffImage:
        raise MissingAssociatedImageError(
            "Trestle stores associated images in separate files."
        )

    def _create_overview(self, page: int = 0) -> AssociatedTiffImage:
        raise MissingAssociatedImageError(
            "Trestle stores associated images in separate files."
        )

    def _create_thumbnail(self, page: int = 0) -> ThumbnailTiffImage:
        raise MissingAssociatedImageError(
            "Trestle stores associated images in separate files."
        )

    def _is_level_series(self, series: TiffPageSeries) -> bool:
        page = series.pages[0]
        return isinstance(page, TiffPage) and page.subfiletype == 0

    def _is_overview_series(self, series: TiffPageSeries) -> bool:
        return False

    def _is_label_series(self, series: TiffPageSeries) -> bool:
        return False

    def _is_thumbnail_series(self, series: TiffPageSeries) -> bool:
        return False

    @staticmethod
    def _overlap_for(page: TiffPage, overlap: tuple[int, int]) -> TileOverlap:
        """De-overlap placement for a level: each tile keeps its (tile - overlap)
        top-left footprint on the composed canvas."""
        return TileOverlap.from_regular_grid(
            Size(page.imagewidth, page.imagelength),
            Size(page.tilewidth, page.tilelength),
            Size(overlap[0], overlap[1]),
        )
