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
from typing import Optional, Union

from tifffile.tifffile import TiffFile, TiffPage, TiffPageSeries
from upath import UPath

from opentile.formats.philips.philips_tiff_image import PhilipsTiffImage
from opentile.formats.philips.philips_tiff_metadata import PhilipsTiffMetadata
from opentile.geometry import SizeMm
from opentile.jpeg import Jpeg
from opentile.metadata import Metadata
from opentile.tiff_image import TiffImage
from opentile.tiler import Tiler


class PhilipsTiffTiler(Tiler):
    def __init__(
        self,
        file: Union[str, Path, UPath, TiffFile],
        turbo_path: Optional[Union[str, Path]] = None,
    ):
        """Tiler for Philips tiff file.

        Parameters
        ----------
        file: Union[str, Path, UPath, TiffFile]
            Filepath to a Philips TiffFile or a Philips TiffFile.
        turbo_path: Optional[Union[str, Path]] = None
            Path to turbojpeg (dll or so).
        """
        super().__init__(file)
        self._jpeg = Jpeg(turbo_path)
        self._metadata = PhilipsTiffMetadata(self._tiff_file)
        assert self._metadata.pixel_spacing is not None
        self._base_mpp = SizeMm.from_tuple(self._metadata.pixel_spacing) * 1000.0

    @property
    def metadata(self) -> Metadata:
        return self._metadata

    @classmethod
    def supported(cls, tiff_file: TiffFile) -> bool:
        return tiff_file.is_philips

    def get_level(self, level: int, page: int = 0) -> TiffImage:
        return self._get_image(self._level_series_index, level, page)

    def get_label(self, page: int = 0) -> TiffImage:
        return self._get_image(self._label_series_index, 0, page)

    def get_overview(self, page: int = 0) -> TiffImage:
        return self._get_image(self._overview_series_index, 0, page)

    @lru_cache(None)
    def _get_image(self, series: int, level: int, page: int = 0) -> TiffImage:
        """Return PhilipsTiffTiledPage for series, level, page."""
        return PhilipsTiffImage(
            self._get_tiff_page(series, level, page),
            self._fh,
            self.base_size,
            self._base_mpp,
            self._jpeg,
        )

    @staticmethod
    def _is_level_series(series: TiffPageSeries) -> bool:
        return series.index == 0

    @staticmethod
    def _is_overview_series(series: TiffPageSeries) -> bool:
        """Return true if series is a overview series."""
        page = series.pages[0]
        assert isinstance(page, TiffPage)
        return page.description.find("Macro") > -1

    @staticmethod
    def _is_label_series(series: TiffPageSeries) -> bool:
        """Return true if series is a label series."""
        page = series.pages[0]
        assert isinstance(page, TiffPage)
        return page.description.find("Label") > -1

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
