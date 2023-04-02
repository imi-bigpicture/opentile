#    Copyright 2021, 2022, 2023 SECTRA AB
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


import math
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from tifffile.tifffile import TiffFile, TiffPage

from opentile.tiler import Tiler
from opentile.geometry import Size
from opentile.jpeg import Jpeg
from opentile.metadata import Metadata
from opentile.ndpi.ndpi_metadata import NdpiMetadata
from opentile.ndpi.ndpi_page import (
    CroppedNdpiPage,
    NdpiOneFramePage,
    NdpiPage,
    NdpiStripedPage,
)


class NdpiTiler(Tiler):
    # The label and overview is cropped out of the macro image.Use a faked
    # label series to avoid clashing with series in the file.
    FAKED_LABEL_SERIES_INDEX = -1

    def __init__(
        self,
        filepath: Union[str, Path],
        tile_size: int,
        turbo_path: Optional[Union[str, Path]] = None,
        label_crop_position: float = 0.3,
    ):
        """Tiler for ndpi file, with functions to produce tiles of specified
        size.

        Parameters
        ----------
        filepath: Union[str, Path]
            Filepath to a ndpi TiffFile.
        tile_size: int
            Tile size to cache and produce. Must be multiple of 8 and will be
            adjusted to be an even multipler or divider of the smallest strip
            width in the file.
        turbo_path: Optional[Union[str, Path]] = None
            Path to turbojpeg (dll or so).
        label_crop_position: float = 0.3
            The position (relative to the image width) to use for cropping out
            the label and overview image from the macro image.

        """
        super().__init__(Path(filepath))

        self._fh = self._tiff_file.filehandle
        self._tile_size = Size(tile_size, tile_size)
        self._tile_size = self._adjust_tile_size(
            tile_size, self._get_smallest_stripe_width()
        )
        if self.tile_size.width % 8 != 0 or self.tile_size.height % 8 != 0:
            raise ValueError(f"Tile size {self.tile_size} not divisable by 8")
        self._turbo_path = turbo_path
        self._jpeg = Jpeg(self._turbo_path)

        self._level_series_index = 0
        for series_index, series in enumerate(self.series):
            if series.name == "Macro":
                self._overview_series_index = series_index
                self._label_series_index = self.FAKED_LABEL_SERIES_INDEX
        self._pages: Dict[Tuple[int, int, int], NdpiPage] = {}
        self._metadata = NdpiMetadata(self.base_page)
        self._label_crop_position = label_crop_position

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._tiff_file.filename}, "
            f"{self.tile_size.to_tuple}, "
            f"{self._turbo_path})"
        )

    def __str__(self) -> str:
        return f"{type(self).__name__} of Tifffile {self._tiff_file}"

    @property
    def tile_size(self) -> Size:
        """The size of the tiles to generate."""
        return self._tile_size

    @property
    def metadata(self) -> Metadata:
        return self._metadata

    @classmethod
    def supported(cls, tiff_file: TiffFile) -> bool:
        return tiff_file.is_ndpi

    def get_page(self, series: int, level: int, page: int) -> NdpiPage:
        """Return NdpiPage for series, level, page. NdpiPages holds a cache, so
        store created pages.
        """
        if not (series, level, page) in self._pages:
            if series == self._level_series_index:
                ndpi_page = self._create_level_page(level, page)
            elif series == self._overview_series_index:
                ndpi_page = self._create_overview_page()
            elif series == self._label_series_index:
                ndpi_page = self._create_label_page()
            else:
                raise ValueError("Unknown series {series}.")
            self._pages[series, level, page] = ndpi_page
        return self._pages[series, level, page]

    @staticmethod
    def _adjust_tile_size(
        requested_tile_width: int, smallest_stripe_width: Optional[int] = None
    ) -> Size:
        """Return adjusted tile size. If file contains striped pages the
        tile size must be an n * smallest stripe width in the file, where n
        is the closest square factor of the ratio between requested tile width
        and smallest stripe width.

        Parameters
        ----------
        requested_tile_width: int
            Requested tile width.
        smallest_stripe_width: Optional[int] = None
            Smallest stripe width in file.

        Returns
        ----------
        Size
            Adjusted tile size.
        """
        if (
            smallest_stripe_width is None
            or smallest_stripe_width == requested_tile_width
        ):
            # No striped pages or requested is equald to smallest
            return Size(requested_tile_width, requested_tile_width)

        if requested_tile_width > smallest_stripe_width:
            factor = requested_tile_width / smallest_stripe_width
        else:
            factor = smallest_stripe_width / requested_tile_width
        # Factor should be a square number (in the series 2^n)
        factor_2 = pow(2, round(math.log2(factor)))
        adjusted_width = factor_2 * smallest_stripe_width
        return Size(adjusted_width, adjusted_width)

    def _get_smallest_stripe_width(self) -> Optional[int]:
        """Return smallest stripe width in file, or None if no page in the
        file is striped.

        Returns
        ----------
        Optional[int]
            The smallest stripe width in the file, or None if no page in the
            file is striped.
        """
        smallest_stripe_width: Optional[int] = None
        for page in self._tiff_file.pages:
            assert isinstance(page, TiffPage)
            stripe_width = page.chunks[1]
            if page.is_tiled and (
                smallest_stripe_width is None or smallest_stripe_width > stripe_width
            ):
                smallest_stripe_width = stripe_width
        return smallest_stripe_width

    def _create_level_page(
        self,
        level: int,
        page: int,
    ) -> Union[NdpiStripedPage, NdpiOneFramePage]:
        """Create a new level page from TiffPage.

        Parameters
        ----------
        level: int
            Level of page.
        page: int
            Page to use.

        Returns
        ----------
        NdpiPage
            Created page.
        """
        tiff_page = (
            self._tiff_file.series[self._level_series_index].levels[level].pages[page]
        )
        assert isinstance(tiff_page, TiffPage)
        if tiff_page.is_tiled:  # Striped ndpi page
            return NdpiStripedPage(
                tiff_page, self._fh, self.base_size, self.tile_size, self._jpeg
            )
        # Single frame, force tiling
        return NdpiOneFramePage(
            tiff_page, self._fh, self.base_size, self.tile_size, self._jpeg
        )

    def _create_label_page(self) -> CroppedNdpiPage:
        """Create a new label page from TiffPage.

        Returns
        ----------
        CroppedNdpiPage
            Created page.
        """
        assert self._overview_series_index is not None
        tiff_page = self._tiff_file.series[self._overview_series_index].pages.pages[0]
        assert isinstance(tiff_page, TiffPage)
        return CroppedNdpiPage(
            tiff_page, self._fh, self._jpeg, (0.0, self._label_crop_position)
        )

    def _create_overview_page(self) -> NdpiPage:
        """Create a new overview page from TiffPage.

        Returns
        ----------
        CroppedNdpiPage
            Created page.
        """
        assert self._overview_series_index is not None
        tiff_page = self._tiff_file.series[self._overview_series_index].pages.pages[0]
        assert isinstance(tiff_page, TiffPage)
        return CroppedNdpiPage(
            tiff_page, self._fh, self._jpeg, (self._label_crop_position, 1.0)
        )
