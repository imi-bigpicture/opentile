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

"""Base tiler class."""

from abc import ABCMeta, abstractmethod
from functools import cached_property
from pathlib import Path
from typing import Any, Optional, Union

from tifffile import TiffFile, TiffFrame, TiffPage, TiffPageSeries
from upath import UPath

from opentile.cache import lru_cached_method
from opentile.file import OpenTileFile
from opentile.geometry import Size
from opentile.metadata import Metadata
from opentile.tiff_format import TiffFormat
from opentile.tiff_image import (
    AssociatedTiffImage,
    LevelTiffImage,
    NonDyadicPyramidLevelError,
    ThumbnailTiffImage,
)


class Tiler(metaclass=ABCMeta):
    """Base class for reading images from TiffFile."""

    def __init__(
        self,
        file: Union[str, Path, UPath, OpenTileFile],
        file_options: Optional[dict[str, Any]],
    ):
        """Base class for reading images from TiffFile.

        Parameters
        ----------
        file: Union[str, Path, UPath, OpenTileFile]
            Filepath to a TiffFile or a TiffFile.
        file_options: Optional[Dict[str, Any]]
            Options to pass to filesystem when opening file.
        """
        if isinstance(file, OpenTileFile):
            if not self.supported(file.tiff):
                raise ValueError("Unsupported file.")
            self._file = file
        else:
            self._file = OpenTileFile(file, file_options)
        self._level_series_index = 0
        self._overview_series_index: Optional[int] = None
        self._label_series_index: Optional[int] = None
        self._thumbnail_series_index: Optional[int] = None
        for series_index, series in enumerate(self._file.series):
            if self._is_level_series(series):
                self._level_series_index = series_index
            if self._is_label_series(series):
                self._label_series_index = series_index
            elif self._is_overview_series(series):
                self._overview_series_index = series_index
            elif self._is_thumbnail_series(series):
                self._thumbnail_series_index = series_index
        self._icc_profile: Optional[bytes] = None
        base_page = self._file.series[self._level_series_index].pages[0]
        assert isinstance(base_page, TiffPage)
        self._base_page = base_page
        self._base_size = Size(self._base_page.imagewidth, self._base_page.imagelength)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @cached_property
    def levels(self) -> list[LevelTiffImage]:
        """Return list of pyramid level TiffImages.

        Trailing coarse levels whose downsample is not a clean power of two (e.g.
        Ventana's coarsest overview levels) cannot be placed in the pyramid and are
        dropped along with any coarser levels."""
        if self._level_series_index is None:
            return []
        levels: list[LevelTiffImage] = []
        for level_index, level in enumerate(
            self._file.series[self._level_series_index].levels
        ):
            try:
                level_images = [
                    self.get_level(level_index, page_index)
                    for page_index, _ in enumerate(level.pages)
                ]
            except NonDyadicPyramidLevelError:
                break
            levels.extend(level_images)
        return levels

    @cached_property
    def labels(self) -> list[AssociatedTiffImage]:
        """Return list of label TiffImage."""
        if self._label_series_index is None:
            return []
        return [
            self.get_label(page_index)
            for page_index, page in enumerate(
                self._file.series[self._label_series_index].pages
            )
        ]

    @cached_property
    def overviews(self) -> list[AssociatedTiffImage]:
        """Return list of overview TiffImage."""
        if self._overview_series_index is None:
            return []
        return [
            self.get_overview(page_index)
            for page_index, page in enumerate(
                self._file.series[self._overview_series_index].pages
            )
        ]

    @cached_property
    def thumbnails(self) -> list[ThumbnailTiffImage]:
        """Return list of thumbnail TiffImage."""
        if self._thumbnail_series_index is None:
            return []
        return [
            self.get_thumbnail(page_index)
            for page_index, page in enumerate(
                self._file.series[self._thumbnail_series_index].pages
            )
        ]

    @property
    def icc_profile(self) -> Optional[bytes]:
        """Return icc profile if found in file."""
        return self._icc_profile

    @property
    @abstractmethod
    def metadata(self) -> Metadata:
        """Return metadata parsed from file."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def format(self) -> TiffFormat:
        """Return the format of the tiff file."""
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def supported(cls, tiff_file: TiffFile) -> bool:
        """Return true if file is supported by tiler."""
        raise NotImplementedError()

    @lru_cached_method(maxsize=None)
    def get_level(self, level: int, page: int = 0) -> LevelTiffImage:
        """Return TiffImage for level in pyramid series.

        Parameters
        ----------
        level: int
            Level to get.
        page: int = 0
            Index of page to get.

        Returns
        ----------
        TiffImage
            Level TiffImage.
        """
        return self._create_level(level, page)

    @lru_cached_method(maxsize=None)
    def get_label(self, page: int = 0) -> AssociatedTiffImage:
        """Return label TiffImage.

        Parameters
        ----------
        page: int = 0
            Index of page to get.

        Returns
        ----------
        TiffImage
            Label TiffImage.
        """
        return self._create_label(page)

    @lru_cached_method(maxsize=None)
    def get_overview(self, page: int = 0) -> AssociatedTiffImage:
        """Return overview TiffImage.

        Parameters
        ----------
        page: int = 0
            Index of page to get.

        Returns
        ----------
        TiffImage
            Overview TiffImage.
        """
        return self._create_overview(page)

    @lru_cached_method(maxsize=None)
    def get_thumbnail(self, page: int = 0) -> ThumbnailTiffImage:
        """Return thumbnail TiffImage.

        Parameters
        ----------
        page: int = 0
            Index of page to get.

        Returns
        ----------
        TiffImage
            Thumbnail TiffImage.
        """
        return self._create_thumbnail(page)

    @abstractmethod
    def _create_level(self, level: int, page: int = 0) -> LevelTiffImage:
        """Create the TiffImage for a pyramid level."""
        raise NotImplementedError()

    @abstractmethod
    def _create_label(self, page: int = 0) -> AssociatedTiffImage:
        """Create the label TiffImage."""
        raise NotImplementedError()

    @abstractmethod
    def _create_overview(self, page: int = 0) -> AssociatedTiffImage:
        """Create the overview TiffImage."""
        raise NotImplementedError()

    @abstractmethod
    def _create_thumbnail(self, page: int = 0) -> ThumbnailTiffImage:
        """Create the thumbnail TiffImage."""
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def _is_level_series(series: TiffPageSeries) -> bool:
        """Return true if series is a level series."""
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def _is_overview_series(series: TiffPageSeries) -> bool:
        """Return true if series is a overview series."""
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def _is_label_series(series: TiffPageSeries) -> bool:
        """Return true if series is a label series."""
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def _is_thumbnail_series(series: TiffPageSeries) -> bool:
        """Return true if series is a thumbnail series."""
        raise NotImplementedError()

    def close(self) -> None:
        """Close tiff file."""
        self._file.close()

    def get_tile(
        self, series: int, level: int, page: int, tile_position: tuple[int, int]
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
        if series == self._level_series_index:
            image = self.get_level(level, page)
        elif series == self._overview_series_index:
            image = self.get_overview(page)
        elif series == self._label_series_index:
            image = self.get_label(page)
        else:
            raise ValueError("Unknown series.")
        return image.get_tile(tile_position)

    def _get_tiff_page(self, series: int, level: int, page: int) -> TiffPage:
        """Return TiffPage for series, level, page."""
        tiff_page = self._file.series[series].levels[level].pages[page]
        if isinstance(tiff_page, TiffFrame):
            tiff_page = tiff_page.aspage()
        assert isinstance(tiff_page, TiffPage)
        return tiff_page
