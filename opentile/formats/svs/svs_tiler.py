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

from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

from tifffile.tifffile import TiffFile, TiffPageSeries

from opentile.formats.svs.svs_image import SvsLZWImage, SvsStripedImage, SvsTiledImage
from opentile.formats.svs.svs_metadata import SvsMetadata
from opentile.geometry import SizeMm
from opentile.jpeg import Jpeg
from opentile.metadata import Metadata
from opentile.tiler import TiffImage, Tiler


class SvsTiler(Tiler):
    def __init__(
        self, filepath: Union[str, Path], turbo_path: Optional[Union[str, Path]] = None
    ):
        """Tiler for svs file.

        Parameters
        ----------
        filepath: Union[str, Path]
            Filepath to a svs TiffFile.
        """
        super().__init__(Path(filepath))
        self._jpeg = Jpeg(turbo_path)
        if "InterColorProfile" in self._tiff_file.pages.first.tags:
            icc_profile = self._tiff_file.pages.first.tags["InterColorProfile"].value
            assert isinstance(icc_profile, bytes) or icc_profile is None
            self._icc_profile = icc_profile
        self._metadata = SvsMetadata(self.base_page)
        self._base_mpp = SizeMm(self._metadata.mpp, self._metadata.mpp)

    @property
    def base_mpp(self) -> SizeMm:
        """Return pixel spacing in um/pixel for base level."""
        return self._base_mpp

    @property
    def metadata(self) -> Metadata:
        return self._metadata

    @classmethod
    def supported(cls, tiff_file: TiffFile) -> bool:
        return tiff_file.is_svs

    @staticmethod
    def _is_level_series(series: TiffPageSeries) -> bool:
        return series.name == "Baseline"

    @staticmethod
    def _is_label_series(series: TiffPageSeries) -> bool:
        return series.name == "Label"

    @staticmethod
    def _is_overview_series(series: TiffPageSeries) -> bool:
        return series.name == "Macro"

    @lru_cache(None)
    def get_level(self, level: int, page: int = 0) -> TiffImage:
        series = self._level_series_index
        if level > 0:
            parent = self.get_level(level - 1, page)
        else:
            parent = None
        svs_page = SvsTiledImage(
            self._get_tiff_page(series, level, page),
            self._fh,
            self.base_size,
            self.base_mpp,
            parent,
        )
        return svs_page

    @lru_cache(None)
    def get_label(self, page: int = 0) -> TiffImage:
        if self._label_series_index is None:
            raise ValueError("No label detected in file")
        return SvsLZWImage(
            self._get_tiff_page(self._label_series_index, 0, page), self._fh, self._jpeg
        )

    @lru_cache(None)
    def get_overview(self, page: int = 0) -> TiffImage:
        if self._overview_series_index is None:
            raise ValueError("No overview detected in file")
        return SvsStripedImage(
            self._get_tiff_page(self._overview_series_index, 0, page),
            self._fh,
            self._jpeg,
        )
