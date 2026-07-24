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

"""Tiler for reading tiles from Huron tiff files."""

from pathlib import Path
from typing import Any, Optional, Union

from tifffile import TiffFile, TiffPageSeries
from upath import UPath

from opentile.file import OpenTileFile
from opentile.formats.huron.huron_tiff_metadata import HuronTiffMetadata
from opentile.geometry import SizeMm
from opentile.metadata import Metadata
from opentile.tiff_format import TiffFormat
from opentile.tiff_image import (
    AssociatedTiffImage,
    DecodedAssociatedImage,
    DecodedThumbnailImage,
    LevelTiffImage,
    NativeTiledLevelImage,
    ThumbnailTiffImage,
)
from opentile.tiler import Tiler


class HuronTiffTiler(Tiler):
    def __init__(
        self,
        file: Union[str, Path, UPath, OpenTileFile],
        file_options: Optional[dict[str, Any]] = None,
    ):
        """Tiler for Huron tiff file.

        Parameters
        ----------
        file: Union[str, Path, UPath, OpenTileFile]
            Filepath to a Huron TiffFile or an opened Huron OpenTileFile.
        file_options: Optional[Dict[str, Any]] = None
            Options to pass to filesystem when opening file.
        """
        super().__init__(file, file_options)
        self._metadata = HuronTiffMetadata(self._base_page)
        self._base_mpp = SizeMm(self._metadata.mpp, self._metadata.mpp)

    @property
    def metadata(self) -> Metadata:
        return self._metadata

    @property
    def format(self) -> TiffFormat:
        return TiffFormat.HURON

    @classmethod
    def supported(cls, tiff_file: TiffFile) -> bool:
        # Huron (MACROscan) files carry an Aperio-like but non-Aperio description with
        # newline-separated fields. "Image Dimensions =" identifies the format (it is
        # not written by the other supported formats); "Resolution =" is also required
        # since the base mpp is read from it.
        description = tiff_file.pages.first.description
        return "Image Dimensions =" in description and "Resolution =" in description

    def _create_level(self, level: int, page: int = 0) -> LevelTiffImage:
        return NativeTiledLevelImage(
            self._get_tiff_page(self._level_series_index, level, page),
            self._file,
            self._base_size,
            self._base_mpp,
        )

    def _create_label(self, page: int = 0) -> AssociatedTiffImage:
        assert self._label_series_index is not None
        return DecodedAssociatedImage(
            self._get_tiff_page(self._label_series_index, 0, page), self._file
        )

    def _create_overview(self, page: int = 0) -> AssociatedTiffImage:
        assert self._overview_series_index is not None
        return DecodedAssociatedImage(
            self._get_tiff_page(self._overview_series_index, 0, page), self._file
        )

    def _create_thumbnail(self, page: int = 0) -> ThumbnailTiffImage:
        assert self._thumbnail_series_index is not None
        return DecodedThumbnailImage(
            self._get_tiff_page(self._thumbnail_series_index, 0, page),
            self._file,
            self._base_size,
            self._base_mpp,
        )

    def _is_level_series(self, series: TiffPageSeries) -> bool:
        return series.index == 0

    def _is_overview_series(self, series: TiffPageSeries) -> bool:
        return series.name == "Macro"

    def _is_label_series(self, series: TiffPageSeries) -> bool:
        return series.name == "Label"

    def _is_thumbnail_series(self, series: TiffPageSeries) -> bool:
        return series.name == "Thumbnail"
