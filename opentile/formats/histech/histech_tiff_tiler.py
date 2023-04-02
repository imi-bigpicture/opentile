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

from tifffile.tifffile import TiffFile, TiffPageSeries

from opentile.formats.histech.histech_tiff_image import HistechTiffImage
from opentile.jpeg import Jpeg
from opentile.metadata import Metadata
from opentile.tiler import Tiler


class HistechTiffTiler(Tiler):
    def __init__(
        self, filepath: Union[str, Path], turbo_path: Optional[Union[str, Path]] = None
    ):
        """Tiler for 3DHistech tiff file.

        Parameters
        ----------
        filepath: Union[str, Path]
            Filepath to a 3DHistech-TiffFile.
        """
        super().__init__(Path(filepath))
        self._turbo_path = turbo_path
        self._jpeg = Jpeg(self._turbo_path)

        self._level_series_index = 0
        for series_index, series in enumerate(self.series):
            if self.is_label(series):
                self._label_series_index = series_index
            elif self.is_overview(series):
                self._overview_series_index = series_index
        self._images: Dict[Tuple[int, int, int], HistechTiffImage] = {}

    @property
    def metadata(self) -> Metadata:
        """No known metadata for 3DHistech tiff files."""
        return Metadata()

    @classmethod
    def supported(cls, tiff_file: TiffFile) -> bool:
        return "3dh_PixelSizeX" in tiff_file.pages.first.description

    def get_image(self, series: int, level: int, page: int = 0) -> HistechTiffImage:
        """Return HistechTiffImage for series, level, page."""
        if not (series, level, page) in self._images:
            self._images[series, level, page] = HistechTiffImage(
                self._get_tiff_page(series, level, page), self._fh, self.base_size
            )
        return self._images[series, level, page]

    @staticmethod
    def is_overview(series: TiffPageSeries) -> bool:
        """Return true if series is a overview series."""
        return False

    @staticmethod
    def is_label(series: TiffPageSeries) -> bool:
        """Return true if series is a label series."""
        return False
