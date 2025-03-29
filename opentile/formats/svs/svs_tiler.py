#    Copyright 2021-2024 SECTRA AB
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

"""Tiler for reading tiles from svs files."""

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Union

from tifffile import TiffFile, TiffPageSeries
from upath import UPath

from opentile.file import OpenTileFile
from opentile.formats.svs.svs_image import (
    SvsLabelImage,
    SvsOverviewImage,
    SvsThumbnailImage,
    SvsTiledImage,
)
from opentile.formats.svs.svs_metadata import SvsMetadata
from opentile.geometry import SizeMm
from opentile.jpeg import Jpeg
from opentile.metadata import Metadata
from opentile.tiff_image import AssociatedTiffImage, LevelTiffImage, ThumbnailTiffImage
from opentile.tiler import Tiler


class SvsTiler(Tiler):
    def __init__(
        self,
        file: Union[str, Path, UPath, OpenTileFile],
        turbo_path: Optional[Union[str, Path]] = None,
        file_options: Optional[Dict[str, Any]] = None,
    ):
        """Tiler for svs file.

        Parameters
        ----------
        file: Union[str, Path, UPath, OpenTileFile]
            Filepath to a svs TiffFile or an opened svs OpenTileFile.
        turbo_path: Optional[Union[str, Path]] = None
            Path to turbojpeg (dll or so).
        file_options: Optional[Dict[str, Any]] = None
            Options to pass to filesystem when opening file.
        """
        super().__init__(file, file_options)
        self._jpeg = Jpeg(turbo_path)
        if "InterColorProfile" in self._file.pages.first.tags:
            icc_profile = self._file.pages.first.tags["InterColorProfile"].value
            assert isinstance(icc_profile, bytes) or icc_profile is None
            self._icc_profile = icc_profile
        self._metadata = SvsMetadata(self._base_page)
        self._base_mpp = SizeMm(self._metadata.mpp, self._metadata.mpp)

    @property
    def metadata(self) -> Metadata:
        return self._metadata

    @classmethod
    def supported(cls, tiff_file: TiffFile) -> bool:
        return tiff_file.is_svs

    @staticmethod
    def _is_level_series(series: TiffPageSeries) -> bool:
        return series.name == "Baseline"

    @staticmethod
    def _is_label_series(series: TiffPageSeries) -> bool:
        return series.name == "Label"

    @staticmethod
    def _is_overview_series(series: TiffPageSeries) -> bool:
        return series.name == "Macro"

    @staticmethod
    def _is_thumbnail_series(series: TiffPageSeries) -> bool:
        return series.name == "Thumbnail"

    @lru_cache(None)
    def get_level(self, level: int, page: int = 0) -> LevelTiffImage:
        return self._get_level(level, page)

    def _get_level(self, level: int, page: int = 0) -> SvsTiledImage:
        series = self._level_series_index
        if level > 0:
            parent = self._get_level(level - 1, page)
        else:
            parent = None
        svs_page = SvsTiledImage(
            self._get_tiff_page(series, level, page),
            self._file,
            self._base_size,
            self._base_mpp,
            parent,
        )
        return svs_page

    @lru_cache(None)
    def get_label(self, page: int = 0) -> AssociatedTiffImage:
        if self._label_series_index is None:
            raise ValueError("No label detected in file")
        return SvsLabelImage(
            self._get_tiff_page(self._label_series_index, 0, page),
            self._file,
            self._jpeg,
        )

    @lru_cache(None)
    def get_overview(self, page: int = 0) -> AssociatedTiffImage:
        if self._overview_series_index is None:
            raise ValueError("No overview detected in file")
        return SvsOverviewImage(
            self._get_tiff_page(self._overview_series_index, 0, page),
            self._file,
            self._jpeg,
        )

    @lru_cache(None)
    def get_thumbnail(self, page: int = 0) -> ThumbnailTiffImage:
        if self._thumbnail_series_index is None:
            raise ValueError("No overview detected in file")
        return SvsThumbnailImage(
            self._get_tiff_page(self._thumbnail_series_index, 0, page),
            self._file,
            self._base_size,
            self._base_mpp,
            self._jpeg,
        )
