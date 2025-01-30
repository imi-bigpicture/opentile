#    Copyright 2022-2024 SECTRA AB
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

"""Tiler for reading tiles from 3Dhistech tiff files."""

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Union

from tifffile import TiffFile, TiffPageSeries
from upath import UPath

from opentile.file import OpenTileFile
from opentile.formats.histech.histech_tiff_image import HistechTiffImage
from opentile.jpeg import Jpeg
from opentile.metadata import Metadata
from opentile.tiff_image import TiffImage
from opentile.tiler import Tiler


class HistechTiffTiler(Tiler):
    def __init__(
        self,
        file: Union[str, Path, UPath, OpenTileFile],
        turbo_path: Optional[Union[str, Path]] = None,
        file_options: Optional[Dict[str, Any]] = None,
    ):
        """Tiler for 3DHistech tiff file.

        Parameters
        ----------
        file: Union[str, Path, UPath, OpenTileFile]
            Filepath to a 3DHistech TiffFile or an opened 3DHistech OpenTileFile.
        turbo_path: Optional[Union[str, Path]] = None
            Path to turbojpeg (dll or so).
        file_options: Optional[Dict[str, Any]] = None
            Options to pass to filesystem when opening file.
        """
        super().__init__(file, file_options)
        self._jpeg = Jpeg(turbo_path)

    @property
    def metadata(self) -> Metadata:
        """No known metadata for 3DHistech tiff files."""
        return Metadata()

    @classmethod
    def supported(cls, tiff_file: TiffFile) -> bool:
        return "3dh_PixelSizeX" in tiff_file.pages.first.description

    @lru_cache(None)
    def get_level(self, level: int, page: int = 0) -> TiffImage:
        return HistechTiffImage(
            self._get_tiff_page(self._level_series_index, level, page),
            self._file,
            self.base_size,
        )

    def get_label(self, page: int = 0) -> TiffImage:
        raise NotImplementedError()

    def get_overview(self, page: int = 0) -> TiffImage:
        raise NotImplementedError()

    def get_thumbnail(self, page: int = 0) -> TiffImage:
        raise NotImplementedError()

    @staticmethod
    def _is_level_series(series: TiffPageSeries) -> bool:
        return series.index == 0

    @staticmethod
    def _is_overview_series(series: TiffPageSeries) -> bool:
        """Return true if series is a overview series."""
        return False

    @staticmethod
    def _is_label_series(series: TiffPageSeries) -> bool:
        """Return true if series is a label series."""
        return False

    @staticmethod
    def _is_thumbnail_series(series: TiffPageSeries) -> bool:
        """Return true if series is a thumbnail series."""
        return False
