#    Copyright 2022 SECTRA AB
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

import ome_types
from ome_types.model.simple_types import UnitsLength
from tifffile.tifffile import TiffFile, TiffPageSeries

from opentile.tiler import OpenTilePage, Tiler
from opentile.geometry import Size, SizeMm
from opentile.jpeg import Jpeg
from opentile.ome.ome_tiff_page import (
    OmeTiffOneFramePage,
    OmeTiffPage,
    OmeTiffTiledPage,
)


class OmeTiffTiler(Tiler):
    """Simple tiler for ome-tiff. Works with images converted with QuPath using
    jpeg. Might report 'wrong' photometric interpretation. Does not support rgb
    images where the colors are separated. This could maybe be supported by
    using turbo-jpeg to losslessly merge the rgb components (assuming they have
    the same tables)."""

    def __init__(
        self, filepath: Union[str, Path], turbo_path: Optional[Union[str, Path]] = None
    ):
        super().__init__(Path(filepath))
        self._fh = self._tiff_file.filehandle
        self._turbo_path = turbo_path
        self._jpeg = Jpeg(self._turbo_path)
        self._base_mpp = self._get_mpp(self._level_series_index)
        self._pages: Dict[Tuple[int, int, int], OpenTilePage] = {}

    @classmethod
    def supported(cls, tiff_file: TiffFile) -> bool:
        return tiff_file.is_ome

    def _is_level_series(self, series: TiffPageSeries) -> bool:
        return not self._is_label_series(series) and not self._is_overview_series(
            series
        )

    def _is_label_series(self, series: TiffPageSeries) -> bool:
        return series.name.strip() == "label"

    def _is_overview_series(self, series: TiffPageSeries) -> bool:
        return series.name.strip() == "macro"

    def _get_mpp(self, series_index: int) -> SizeMm:
        mpp = self._get_optional_mpp(series_index)
        if mpp is None:
            raise ValueError("Could not find physical size of x and y.")
        return mpp

    def _get_optional_mpp(self, series_index: int) -> Optional[SizeMm]:
        assert self._tiff_file.ome_metadata is not None
        metadata = ome_types.from_xml(self._tiff_file.ome_metadata, parser="lxml")
        pixels = metadata.images[series_index].pixels
        if (
            pixels.physical_size_x_unit != UnitsLength.MICROMETER
            or pixels.physical_size_y_unit != UnitsLength.MICROMETER
        ):
            raise NotImplementedError("Only um physical size implemented.")
        mpp_x, mpp_y = pixels.physical_size_x, pixels.physical_size_y
        if mpp_x is None or mpp_y is None:
            return None
        return SizeMm(mpp_x, mpp_y)

    def get_page(self, series: int, level: int, page: int = 0) -> OpenTilePage:
        """Return OpenTilePage for series, level, page."""
        if (series, level, page) not in self._pages:
            tiff_page = self._get_tiff_page(series, level, page)
            if tiff_page.is_tiled:
                self._pages[series, level, page] = OmeTiffTiledPage(
                    tiff_page,
                    self._fh,
                    self.base_size,
                    self._base_mpp,
                )
            elif series == self._level_series_index:
                self._pages[series, level, page] = OmeTiffOneFramePage(
                    tiff_page,
                    self._fh,
                    self.base_size,
                    Size(self.base_page.tilewidth, self.base_page.tilelength),
                    self._base_mpp,
                    self._jpeg,
                )
            else:
                self._pages[series, level, page] = OmeTiffPage(
                    tiff_page,
                    self._fh,
                    self._get_optional_mpp(series),
                )
        return self._pages[series, level, page]
