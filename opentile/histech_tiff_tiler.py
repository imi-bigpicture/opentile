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

from tifffile.tifffile import FileHandle, TiffFile, TiffPage, TiffPageSeries

from opentile.common import NativeTiledPage, Tiler
from opentile.geometry import Size, SizeMm
from opentile.jpeg import Jpeg
from opentile.metadata import Metadata


class HistechTiffTiledPage(NativeTiledPage):
    def __init__(
        self,
        page: TiffPage,
        fh: FileHandle,
        base_shape: Size
    ):
        """OpenTiledPage for 3DHistech Tiff-page.

        Parameters
        ----------
        page: TiffPage
            TiffPage defining the page.
        fh: FileHandler
            Filehandler to read data from.
        base_shape: Size
            Size of base level in pyramid.
        """
        super().__init__(page, fh)
        self._base_shape = base_shape
        self._pyramid_index = self._calculate_pyramidal_index(self._base_shape)
        self._mpp = self._get_mpp_from_page()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._page}, {self._fh}, "
            f"{self._base_shape})"
        )

    @property
    def pixel_spacing(self) -> SizeMm:
        """Return pixel spacing in mm per pixel."""
        return self.mpp * 1000

    @property
    def mpp(self) -> SizeMm:
        """Return pixel spacing in um per pixel."""
        return self._mpp

    @property
    def photometric_interpretation(self) -> str:
        photometric_interpretation = str(
            self._page.photometric
        ).split('.', maxsplit=1)[1]
        if photometric_interpretation == 'PALETTE':
            return 'MINISBLACK'
        else:
            return photometric_interpretation

    def _get_mpp_from_page(self) -> SizeMm:
        items_split = self._page.description.split('|')
        header = items_split.pop(0)
        items = {
            key: value
            for (key, value) in (
                item.split(' = ', 1) for item in items_split
            )
        }
        return SizeMm(
            float(items['3dh_PixelSizeX']),
            float(items['3dh_PixelSizeY'])
        ) / 1000 / 1000


class HistechTiffTiler(Tiler):
    def __init__(
        self,
        filepath: Union[str, Path],
        turbo_path: Optional[Union[str, Path]] = None
    ):
        """Tiler for 3DHistech tiff file.

        Parameters
        ----------
        filepath: Union[str, Path]
            Filepath to a 3DHistech-TiffFile.
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
        self._pages: Dict[Tuple[int, int, int], HistechTiffTiledPage] = {}

    @property
    def metadata(self) -> Metadata:
        """No known metadata for 3DHistech tiff files."""
        return Metadata()

    @classmethod
    def supported(cls, tiff_file: TiffFile) -> bool:
        return '3dh_PixelSizeX' in tiff_file.pages.first.description

    def get_page(
        self,
        series: int,
        level: int,
        page: int = 0
    ) -> HistechTiffTiledPage:
        """Return PhilipsTiffTiledPage for series, level, page."""
        if not (series, level, page) in self._pages:
            self._pages[series, level, page] = HistechTiffTiledPage(
                self._get_tiff_page(series, level, page),
                self._fh,
                self.base_size
            )
        return self._pages[series, level, page]

    @staticmethod
    def is_overview(series: TiffPageSeries) -> bool:
        """Return true if series is a overview series."""
        return False

    @staticmethod
    def is_label(series: TiffPageSeries) -> bool:
        """Return true if series is a label series."""
        return False
