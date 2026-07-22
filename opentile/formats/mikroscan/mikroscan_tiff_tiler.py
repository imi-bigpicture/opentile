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

"""Tiler for reading tiles from Mikroscan tiff files."""

from pathlib import Path
from typing import Any, Optional, Union

from tifffile import TiffFile, TiffPageSeries
from upath import UPath

from opentile.file import OpenTileFile
from opentile.formats.mikroscan.mikroscan_tiff_image import MikroscanTiffImage
from opentile.formats.mikroscan.mikroscan_tiff_metadata import MikroscanTiffMetadata
from opentile.geometry import SizeMm
from opentile.metadata import Metadata
from opentile.tiff_format import TiffFormat
from opentile.tiff_image import (
    AssociatedTiffImage,
    LevelTiffImage,
    ThumbnailTiffImage,
)
from opentile.tiler import Tiler


class MikroscanTiffTiler(Tiler):
    def __init__(
        self,
        file: Union[str, Path, UPath, OpenTileFile],
        file_options: Optional[dict[str, Any]] = None,
    ):
        """Tiler for Mikroscan tiff file.

        Parameters
        ----------
        file: Union[str, Path, UPath, OpenTileFile]
            Filepath to a Mikroscan TiffFile or an opened Mikroscan OpenTileFile.
        file_options: Optional[Dict[str, Any]] = None
            Options to pass to filesystem when opening file.
        """
        super().__init__(file, file_options)
        self._metadata = MikroscanTiffMetadata(self._base_page)
        self._base_mpp = SizeMm(self._metadata.mpp, self._metadata.mpp)

    @property
    def metadata(self) -> Metadata:
        return self._metadata

    @property
    def format(self) -> TiffFormat:
        return TiffFormat.MIKROSCAN

    @classmethod
    def supported(cls, tiff_file: TiffFile) -> bool:
        # Mikroscan (SL5) files use the Aperio pipe-separated description but with a
        # "Mikroscan Image Structure" header instead of the "Aperio " prefix.
        return "Mikroscan Image Structure" in tiff_file.pages.first.description

    def _create_level(self, level: int, page: int = 0) -> LevelTiffImage:
        return MikroscanTiffImage(
            self._get_tiff_page(self._level_series_index, level, page),
            self._file,
            self._base_size,
            self._base_mpp,
        )

    def _create_label(self, page: int = 0) -> AssociatedTiffImage:
        raise NotImplementedError()

    def _create_overview(self, page: int = 0) -> AssociatedTiffImage:
        raise NotImplementedError()

    def _create_thumbnail(self, page: int = 0) -> ThumbnailTiffImage:
        raise NotImplementedError()

    @staticmethod
    def _is_level_series(series: TiffPageSeries) -> bool:
        return series.index == 0

    @staticmethod
    def _is_overview_series(series: TiffPageSeries) -> bool:
        return False

    @staticmethod
    def _is_label_series(series: TiffPageSeries) -> bool:
        return False

    @staticmethod
    def _is_thumbnail_series(series: TiffPageSeries) -> bool:
        return False
