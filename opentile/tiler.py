#    Copyright 2021 SECTRA AB
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


from abc import ABCMeta, abstractclassmethod, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple

from tifffile.tifffile import (
    TiffFile,
    TiffPage,
    TiffPageSeries,
)

from opentile.geometry import Size
from opentile.metadata import Metadata
from opentile.page import OpenTilePage


class Tiler(metaclass=ABCMeta):
    """Abstract class for reading pages from TiffFile."""

    _level_series_index: int = 0
    _overview_series_index: Optional[int] = None
    _label_series_index: Optional[int] = None
    _icc_profile: Optional[bytes] = None

    def __init__(self, filepath: Path):
        """Abstract class for reading pages from TiffFile.

        Parameters
        ----------
        filepath: Path
            Filepath to a TiffFile.
        """
        self._tiff_file = TiffFile(filepath)
        base_page = self.series[self._level_series_index].pages[0]
        assert isinstance(base_page, TiffPage)
        self._base_page = base_page
        self._base_size = Size(self.base_page.imagewidth, self.base_page.imagelength)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def metadata(self) -> Metadata:
        raise NotImplementedError()

    @property
    def base_page(self) -> TiffPage:
        """Return base pyramid level in pyramid series."""
        return self._base_page

    @property
    def base_size(self) -> Size:
        """Return size of base pyramid level in pyramid series."""
        return self._base_size

    @property
    def series(self) -> List[TiffPageSeries]:
        """Return contained TiffPageSeries."""
        return self._tiff_file.series

    @property
    def levels(self) -> List[OpenTilePage]:
        """Return list of pyramid level OpenTilePages."""
        if self._level_series_index is None:
            return []
        return [
            self.get_level(level_index, page_index)
            for level_index, level in enumerate(
                self.series[self._level_series_index].levels
            )
            for page_index, page in enumerate(level.pages)
        ]

    @property
    def labels(self) -> List[OpenTilePage]:
        """Return list of label OpenTilePage."""
        if self._label_series_index is None:
            return []
        return [
            self.get_label(page_index)
            for page_index, page in enumerate(
                self.series[self._label_series_index].pages
            )
        ]

    @property
    def overviews(self) -> List[OpenTilePage]:
        """Return list of overview OpenTilePage."""
        if self._overview_series_index is None:
            return []
        return [
            self.get_overview(page_index)
            for page_index, page in enumerate(
                self.series[self._overview_series_index].pages
            )
        ]

    @property
    def icc_profile(self) -> Optional[bytes]:
        """Return icc profile if found in file."""
        return self._icc_profile

    @abstractmethod
    def get_page(self, series: int, level: int, page: int) -> OpenTilePage:
        """Should return a OpenTilePage for series, level, page in file."""
        raise NotImplementedError()

    @abstractclassmethod
    def supported(cls, tiff_file: TiffFile) -> bool:
        raise NotImplementedError()

    def close(self) -> None:
        """CLose tiff-file."""
        self._tiff_file.close()

    def get_tile(
        self, series: int, level: int, page: int, tile_position: Tuple[int, int]
    ) -> bytes:
        """Return list of image bytes for tiles at tile positions.

        Parameters
        ----------
        series: int
            Series of page to get tile from.
        level: int
            Level of page to get tile from.
        page: int
            Page to get tile from.
        tile_position: Tuple[int, int]
            Position of tile to get.

        Returns
        ----------
        bytes
            Tile at position.
        """
        tiled_page = self.get_page(series, level, page)
        return tiled_page.get_tile(tile_position)

    def get_level(self, level: int, page: int = 0) -> OpenTilePage:
        """Return OpenTilePage for level in pyramid series.

        Parameters
        ----------
        level: int
            Level to get.
        page: int
            Index of page to get.

        Returns
        ----------
        OpenTilePage
            Level OpenTilePage.
        """
        return self.get_page(self._level_series_index, level, page)

    def get_label(self, page: int = 0) -> OpenTilePage:
        """Return OpenTilePage for label in label series.

        Parameters
        ----------
        page: int
            Index of page to get.

        Returns
        ----------
        OpenTilePage
            Label OpenTilePage.
        """
        if self._label_series_index is None:
            raise ValueError("No label detected in file")
        return self.get_page(self._label_series_index, 0, page)

    def get_overview(self, page: int = 0) -> OpenTilePage:
        """Return OpenTilePage for overview in overview series.

        Parameters
        ----------
        page: int
            Index of page to get.

        Returns
        ----------
        OpenTilePage
            Overview OpenTilePage.
        """
        if self._overview_series_index is None:
            raise ValueError("No overview detected in file")
        return self.get_page(self._overview_series_index, 0, page)

    def _get_tiff_page(self, series: int, level: int, page: int) -> TiffPage:
        """Return TiffPage for series, level, page."""
        tiff_page = self.series[series].levels[level].pages[page]
        assert isinstance(tiff_page, TiffPage)
        return tiff_page
