#    Copyright 2022-2023 SECTRA AB
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

"""Image implementation for 3Dhistech tiff files."""

from typing import List, Optional

from tifffile import COMPRESSION, PHOTOMETRIC, TiffPage

from opentile.file import OpenTileFile
from opentile.geometry import Size, SizeMm
from opentile.tiff_image import NativeTiledTiffImage


class HistechTiffImage(NativeTiledTiffImage):
    def __init__(self, page: TiffPage, file: OpenTileFile, base_size: Size):
        """OpenTiledPage for 3DHistech Tiff image.

        Parameters
        ----------
        page: TiffPage
            TiffPage defining the page.
        file: OpenTileFile
            Fileto read data from.
        base_size: Size
            Size of base level in pyramid.
        """
        super().__init__(page, file)
        self._base_size = base_size
        self._pyramid_index = self._calculate_pyramidal_index(self._base_size)
        self._mpp = self._get_mpp_from_page()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._page}, {self._file}, " f"{self._base_size})"
        )

    @property
    def pixel_spacing(self) -> SizeMm:
        """Return pixel spacing in mm per pixel."""
        return self.mpp / 1000

    @property
    def supported_compressions(self) -> Optional[List[COMPRESSION]]:
        return None

    @property
    def mpp(self) -> SizeMm:
        """Return pixel spacing in um per pixel."""
        return self._mpp

    @property
    def photometric_interpretation(self) -> PHOTOMETRIC:
        photometric_interpretation = PHOTOMETRIC(self._page.photometric)
        if photometric_interpretation == PHOTOMETRIC.PALETTE:
            return PHOTOMETRIC.MINISBLACK
        return photometric_interpretation

    def _get_mpp_from_page(self) -> SizeMm:
        items_split = self._page.description.split("|")
        header = items_split.pop(0)
        items = {
            key: value
            for (key, value) in (item.split(" = ", 1) for item in items_split)
        }
        mpp = float(items["MPP"])
        return SizeMm(mpp, mpp)
