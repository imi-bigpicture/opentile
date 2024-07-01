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

"""Tiler for reading tiles from ndpi files."""

import math
from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

from tifffile.tifffile import TiffFile, TiffPage, TiffPageSeries
from upath import UPath

from opentile.formats.ndpi.ndpi_image import (
    NdpiCroppedImage,
    NdpiOneFrameImage,
    NdpiStripedImage,
)
from opentile.formats.ndpi.ndpi_metadata import NdpiMetadata
from opentile.geometry import Size
from opentile.jpeg import Jpeg
from opentile.metadata import Metadata
from opentile.tiff_image import TiffImage
from opentile.tiler import Tiler


class NdpiTiler(Tiler):
    def __init__(
        self,
        file: Union[str, Path, UPath, TiffFile],
        tile_size: int,
        turbo_path: Optional[Union[str, Path]] = None,
        label_crop_position: float = 0.3,
    ):
        """Tiler for ndpi file, with functions to produce tiles of specified
        size.

        Parameters
        ----------
        file: Union[str, Path, UPath, TiffFile]
            Filepath to a ndpi TiffFile or a ndpi TiffFile.
        tile_size: int
            Tile size to cache and produce. Must be multiple of 8 and will be
            adjusted to be an even multiplier or divider of the smallest strip
            width in the file.
        turbo_path: Optional[Union[str, Path]] = None
            Path to turbojpeg (dll or so).
        label_crop_position: float = 0.3
            The position (relative to the image width) to use for cropping out
            the label and overview image from the macro image.

        """
        super().__init__(file)
        self._tile_size = Size(tile_size, tile_size)
        self._tile_size = self._adjust_tile_size(
            tile_size, self._get_smallest_stripe_width()
        )
        if self.tile_size.width % 8 != 0 or self.tile_size.height % 8 != 0:
            raise ValueError(f"Tile size {self.tile_size} not divisible by 8")
        self._jpeg = Jpeg(turbo_path)
        self._metadata = NdpiMetadata(self.base_page)
        self._label_crop_position = label_crop_position

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

    @staticmethod
    def _is_level_series(series: TiffPageSeries) -> bool:
        return series.index == 0

    @staticmethod
    def _is_overview_series(series: TiffPageSeries) -> bool:
        """Return true if series is a overview series."""
        return series.name == "Macro"

    @staticmethod
    def _is_label_series(series: TiffPageSeries) -> bool:
        """Return true if series is a label series."""
        return False

    @staticmethod
    def _adjust_tile_size(
        requested_tile_width: int, smallest_stripe_width: Optional[int] = None
    ) -> Size:
        """Return adjusted tile size. If file contains striped images the
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
            # No striped pages or requested is equal to smallest
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
        """Return smallest stripe width in file, or None if no image in the
        file is striped.

        Returns
        ----------
        Optional[int]
            The smallest stripe width in the file, or None if no image in the
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

    @lru_cache(None)
    def get_level(
        self,
        level: int,
        page: int = 0,
    ) -> TiffImage:
        """Create a level image.

        Parameters
        ----------
        level: int
            Level of page.
        page: int
            Page to use.

        Returns
        ----------
        TiffImage
            Created image.
        """
        tiff_page = (
            self._tiff_file.series[self._level_series_index].levels[level].pages[page]
        )
        assert isinstance(tiff_page, TiffPage)
        if tiff_page.is_tiled:  # Striped ndpi page
            return NdpiStripedImage(
                tiff_page, self._fh, self.base_size, self.tile_size, self._jpeg
            )
        # Single frame, force tiling
        return NdpiOneFrameImage(
            tiff_page, self._fh, self.base_size, self.tile_size, self._jpeg
        )

    @lru_cache(None)
    def get_label(self, page: int = 0) -> TiffImage:
        """Create a label image.

        Returns
        ----------
        TiffImage
            Created image.
        """
        assert self._overview_series_index is not None
        tiff_page = self._tiff_file.series[self._overview_series_index].pages.pages[
            page
        ]
        assert isinstance(tiff_page, TiffPage)
        return NdpiCroppedImage(
            tiff_page, self._fh, self._jpeg, (0.0, self._label_crop_position)
        )

    @lru_cache(None)
    def get_overview(self, page: int = 0) -> TiffImage:
        """Create a overview image.

        Returns
        ----------
        TiffImage
            Created image.
        """
        assert self._overview_series_index is not None
        tiff_page = self._tiff_file.series[self._overview_series_index].pages.pages[
            page
        ]
        assert isinstance(tiff_page, TiffPage)
        return NdpiCroppedImage(
            tiff_page, self._fh, self._jpeg, (self._label_crop_position, 1.0)
        )
