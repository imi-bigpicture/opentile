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

"""Tiler for reading tiles from Philips tiff files."""

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Union

from tifffile import TiffFile, TiffPage, TiffPageSeries, TiffFrame
from upath import UPath

from opentile.file import OpenTileFile
from opentile.formats.philips.philips_tiff_image import (
    PhilipsAssociatedTiffImage,
    PhilipsLevelTiffImage,
    PhilipsThumbnailTiffImage,
)
from opentile.formats.philips.philips_tiff_metadata import PhilipsTiffMetadata
from opentile.geometry import SizeMm
from opentile.jpeg import Jpeg
from opentile.metadata import Metadata
from opentile.tiff_image import (
    AssociatedTiffImage,
    LevelTiffImage,
    ThumbnailTiffImage,
)
from opentile.tiler import Tiler


class PhilipsTiffTiler(Tiler):
    def __init__(
        self,
        file: Union[str, Path, UPath, OpenTileFile],
        turbo_path: Optional[Union[str, Path]] = None,
        file_options: Optional[Dict[str, Any]] = None,
    ):
        """Tiler for Philips tiff file.

        Parameters
        ----------
        file: Union[str, Path, UPath, OpenTileFile]
            Filepath to a Philips TiffFile or an opened Philips OpenTileFile.
        turbo_path: Optional[Union[str, Path]] = None
            Path to turbojpeg (dll or so).
        file_options: Optional[Dict[str, Any]] = None
            Options to pass to filesystem when opening file.
        """
        super().__init__(file, file_options)
        self._jpeg = Jpeg(turbo_path)
        self._metadata = PhilipsTiffMetadata(self._file.tiff)
        assert self._metadata.pixel_spacing is not None
        self._base_mpp = SizeMm.from_tuple(self._metadata.pixel_spacing) * 1000.0

    @property
    def metadata(self) -> Metadata:
        return self._metadata

    @classmethod
    def supported(cls, tiff_file: TiffFile) -> bool:
        return tiff_file.is_philips

    @lru_cache(None)
    def get_level(self, level: int, page: int = 0) -> LevelTiffImage:
        return PhilipsLevelTiffImage(
            self._get_tiff_page(self._level_series_index, level, page),
            self._file,
            self._base_size,
            self._base_mpp,
            self._jpeg,
        )

    @lru_cache(None)
    def get_label(self, page: int = 0) -> AssociatedTiffImage:
        if self._label_series_index is None:
            raise ValueError("No label series found in this file.")
        return PhilipsAssociatedTiffImage(
            self._get_tiff_page(self._label_series_index, 0, page), self._file
        )

    @lru_cache(None)
    def get_overview(self, page: int = 0) -> AssociatedTiffImage:
        if self._overview_series_index is None:
            raise ValueError("No overview series found in this file.")
        return PhilipsAssociatedTiffImage(
            self._get_tiff_page(self._overview_series_index, 0, page), self._file
        )

    @lru_cache(None)
    def get_thumbnail(self, page: int = 0) -> ThumbnailTiffImage:
        if self._thumbnail_series_index is None:
            raise ValueError("No thumbnail series found in this file.")
        return PhilipsThumbnailTiffImage(
            self._get_tiff_page(self._thumbnail_series_index, 0, page),
            self._file,
            self._base_size,
            self._base_mpp,
        )

    @staticmethod
    def _is_level_series(series: TiffPageSeries) -> bool:
        return series.index == 0

    @staticmethod
    def _is_overview_series(series: TiffPageSeries) -> bool:
        page = series.pages[0]
        if isinstance(page, TiffFrame):
            page = page.aspage()
        assert isinstance(page, TiffPage)
        return page.description.find("Macro") > -1

    @staticmethod
    def _is_label_series(series: TiffPageSeries) -> bool:
        page = series.pages[0]
        if isinstance(page, TiffFrame):
            page = page.aspage()
        assert isinstance(page, TiffPage)
        return page.description.find("Label") > -1

    @staticmethod
    def _is_thumbnail_series(series: TiffPageSeries) -> bool:
        page = series.pages[0]
        if isinstance(page, TiffFrame):
            page = page.aspage()
        assert isinstance(page, TiffPage)
        return page.description.find("Thumbnail") > -1

    @staticmethod
    def _get_associated_mpp_from_page(page: TiffPage):
        """Return mpp (um/pixel) for associated image (label or
        macro) from page."""
        pixel_size_start_string = "pixelsize=("
        pixel_size_start = page.description.find(pixel_size_start_string)
        pixel_size_end = page.description.find(")", pixel_size_start)
        pixel_size_string = page.description[
            pixel_size_start + len(pixel_size_start_string) : pixel_size_end
        ]
        pixel_spacing = SizeMm.from_tuple(
            [float(v) for v in pixel_size_string.replace('"', "").split(",")]
        )
        return pixel_spacing / 1000.0
