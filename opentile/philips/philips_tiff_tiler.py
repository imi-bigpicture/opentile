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

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from tifffile.tifffile import (
    TiffFile,
    TiffPage,
    TiffPageSeries,
)

from opentile.tiler import Tiler
from opentile.geometry import SizeMm
from opentile.jpeg import Jpeg
from opentile.metadata import Metadata
from opentile.philips.philips_metadata import PhilipsMetadata
from opentile.philips.philips_page import PhilipsTiffTiledPage


class PhilipsTiffTiler(Tiler):
    def __init__(
        self, filepath: Union[str, Path], turbo_path: Optional[Union[str, Path]] = None
    ):
        """Tiler for Philips tiff file.

        Parameters
        ----------
        filepath: Union[str, Path]
            Filepath to a Philips-TiffFile.
        turbo_path: Optional[Union[str, Path]] = None
            Path to turbojpeg (dll or so).
        """
        super().__init__(Path(filepath))
        self._fh = self._tiff_file.filehandle

        self._turbo_path = turbo_path
        self._jpeg = Jpeg(self._turbo_path)

        self._level_series_index = 0
        for series_index, series in enumerate(self.series):
            if self.is_label(series):
                self._label_series_index = series_index
            elif self.is_overview(series):
                self._overview_series_index = series_index
        self._metadata = PhilipsMetadata(self._tiff_file)
        assert self._metadata.pixel_spacing is not None
        self._base_mpp = SizeMm.from_tuple(self._metadata.pixel_spacing) * 1000.0
        self._pages: Dict[Tuple[int, int, int], PhilipsTiffTiledPage] = {}

    @property
    def metadata(self) -> Metadata:
        return self._metadata

    @classmethod
    def supported(cls, tiff_file: TiffFile) -> bool:
        return tiff_file.is_philips

    def get_page(self, series: int, level: int, page: int = 0) -> PhilipsTiffTiledPage:
        """Return PhilipsTiffTiledPage for series, level, page."""
        if not (series, level, page) in self._pages:
            self._pages[series, level, page] = PhilipsTiffTiledPage(
                self._get_tiff_page(series, level, page),
                self._fh,
                self.base_size,
                self._base_mpp,
                self._jpeg,
            )
        return self._pages[series, level, page]

    @staticmethod
    def is_overview(series: TiffPageSeries) -> bool:
        """Return true if series is a overview series."""
        page = series.pages[0]
        assert isinstance(page, TiffPage)
        return page.description.find("Macro") > -1

    @staticmethod
    def is_label(series: TiffPageSeries) -> bool:
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
