#    Copyright 2022-2024 SECTRA AB
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

"""Tiler for reading tiles from OME tiff files."""

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Union

import ome_types
from ome_types.model.simple_types import UnitsLength
from tifffile import TiffFile, TiffPageSeries
from upath import UPath

from opentile.file import OpenTileFile
from opentile.formats.ome.ome_tiff_image import (
    OmeTiffImage,
    OmeTiffOneFrameImage,
    OmeTiffTiledImage,
)
from opentile.geometry import Size, SizeMm
from opentile.jpeg import Jpeg
from opentile.metadata import Metadata
from opentile.tiler import TiffImage, Tiler


class OmeTiffTiler(Tiler):
    """Simple tiler for OME tiff. Works with images converted with QuPath using
    jpeg. Might report 'wrong' photometric interpretation. Does not support rgb
    images where the colors are separated. This could maybe be supported by
    using turbo-jpeg to losslessly merge the rgb components (assuming they have
    the same tables)."""

    def __init__(
        self,
        file: Union[str, Path, UPath, OpenTileFile],
        turbo_path: Optional[Union[str, Path]] = None,
        file_options: Optional[Dict[str, Any]] = None,
    ):
        """Tiler for ome tiff file.

        Parameters
        ----------
        file: Union[str, Path, UPath, OpenTileFile]
            Filepath to a ome tiff TiffFile or an opened ome tiff OpenTileFile.
        turbo_path: Optional[Union[str, Path]] = None
            Path to turbojpeg (dll or so).
        file_options: Optional[Dict[str, Any]] = None
            Options to pass to filesystem when opening file.
        """
        super().__init__(file, file_options)
        self._jpeg = Jpeg(turbo_path)
        self._base_mpp = self._get_mpp(self._level_series_index)

    @property
    def metadata(self) -> Metadata:
        """Metadata parsing not implemented for OmeTiff."""
        return Metadata()

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
        assert self._file.tiff.ome_metadata is not None
        metadata = ome_types.from_xml(self._file.tiff.ome_metadata)
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

    @lru_cache(None)
    def get_level(self, level: int, page: int = 0) -> TiffImage:
        tiff_page = self._get_tiff_page(self._level_series_index, level, page)
        if tiff_page.is_tiled:
            return OmeTiffTiledImage(
                tiff_page,
                self._file,
                self.base_size,
                self._base_mpp,
            )
        return OmeTiffOneFrameImage(
            tiff_page,
            self._file,
            self.base_size,
            Size(self.base_page.tilewidth, self.base_page.tilelength),
            self._base_mpp,
            self._jpeg,
        )

    @lru_cache(None)
    def get_label(self, page: int = 0) -> TiffImage:
        if self._label_series_index is None:
            raise ValueError("No label detected in file")
        return self._get_associated_image(self._label_series_index, page)

    @lru_cache(None)
    def get_overview(self, page: int = 0) -> TiffImage:
        if self._overview_series_index is None:
            raise ValueError("No overview detected in file")
        return self._get_associated_image(self._overview_series_index, page)

    def _get_associated_image(self, series: int, page: int = 0) -> TiffImage:
        tiff_page = self._get_tiff_page(series, 0, page)
        return OmeTiffImage(
            tiff_page,
            self._file,
            self._get_optional_mpp(series),
        )
