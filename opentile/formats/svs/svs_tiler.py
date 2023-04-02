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
from typing import Dict, Optional, Tuple, Union, cast

from tifffile.tifffile import TiffFile

from opentile.formats.svs.svs_metadata import SvsMetadata
from opentile.formats.svs.svs_image import SvsLZWImage, SvsStripedImage, SvsTiledImage
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
        self._turbo_path = turbo_path
        self._jpeg = Jpeg(self._turbo_path)

        for series_index, series in enumerate(self.series):
            if series.name == "Baseline":
                self._level_series_index = series_index
            elif series.name == "Label":
                self._label_series_index = series_index
            elif series.name == "Macro":
                self._overview_series_index = series_index
        self._images: Dict[Tuple[int, int, int], TiffImage] = {}
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

    def _get_level_page(self, level: int, page: int = 0) -> SvsTiledImage:
        series = self._level_series_index
        if level > 0:
            parent = self.get_image(series, level - 1, page)
            parent = cast(SvsTiledImage, parent)
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

    def get_image(self, series: int, level: int, page: int = 0) -> TiffImage:
        """Return SvsTiledImage for series, level, page."""
        if not (series, level, page) in self._images:
            if series == self._overview_series_index:
                svs_page = SvsStripedImage(
                    self._get_tiff_page(series, level, page), self._fh, self._jpeg
                )
            elif series == self._label_series_index:
                svs_page = SvsLZWImage(
                    self._get_tiff_page(series, level, page), self._fh, self._jpeg
                )
            elif series == self._level_series_index:
                svs_page = self._get_level_page(level, page)
            else:
                raise NotImplementedError()

            self._images[series, level, page] = svs_page
        return self._images[series, level, page]
