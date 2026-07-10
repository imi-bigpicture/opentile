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

from functools import cached_property
from pathlib import Path
from typing import Any, Optional, Union

from tifffile import TiffFile, TiffPage, TiffPageSeries
from upath import UPath

from opentile.file import OpenTileFile
from opentile.formats.trestle.trestle_tiff_metadata import TrestleMetadata
from opentile.geometry import Size
from opentile.metadata import Metadata
from opentile.tiff_format import TiffFormat
from opentile.tiff_image import (
    AssociatedTiffImage,
    LevelTiffImage,
    OverlappingLevelTiffImage,
    ThumbnailTiffImage,
)
from opentile.tile_overlap import TileOverlap
from opentile.tiler import Tiler


class TrestleTiffTiler(Tiler):
    def __init__(
        self,
        file: Union[str, Path, UPath, OpenTileFile],
        file_options: Optional[dict[str, Any]] = None,
    ):
        """Tiler for Trestle tiff file.

        Parameters
        ----------
        file: Union[str, Path, UPath, OpenTileFile]
            Filepath to a Trestle TiffFile or an opened Trestle OpenTileFile.
        file_options: Optional[Dict[str, Any]] = None
            Options to pass to filesystem when opening file.
        """
        super().__init__(file, file_options)
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
        overlap = self._overlap_for(tiff_page, self._metadata.level_overlap(level))
        scale = self._base_composed_size.width / overlap.image_size.width
        return OverlappingLevelTiffImage(
            tiff_page,
            self._file,
            self._base_mpp,
            scale,
            overlap,
        )

    def _create_label(self, page: int = 0) -> AssociatedTiffImage:
        raise NotImplementedError("Trestle stores associated images in separate files.")

    def _create_overview(self, page: int = 0) -> AssociatedTiffImage:
        raise NotImplementedError("Trestle stores associated images in separate files.")

    def _create_thumbnail(self, page: int = 0) -> ThumbnailTiffImage:
        raise NotImplementedError("Trestle stores associated images in separate files.")

    @staticmethod
    def _is_level_series(series: TiffPageSeries) -> bool:
        page = series.pages[0]
        return isinstance(page, TiffPage) and page.subfiletype == 0

    @staticmethod
    def _is_overview_series(series: TiffPageSeries) -> bool:
        return False

    @staticmethod
    def _is_label_series(series: TiffPageSeries) -> bool:
        return False

    @staticmethod
    def _is_thumbnail_series(series: TiffPageSeries) -> bool:
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
