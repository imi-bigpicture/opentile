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

from functools import cached_property
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import ome_types
from ome_types.model.simple_types import UnitsLength
from tifffile import COMPRESSION, TiffFile, TiffPageSeries
from upath import UPath

from opentile.file import OpenTileFile
from opentile.formats.ome.ome_tiff_image import (
    OmeTiffAssociatedImage,
    OmeTiffOneFrameImage,
    OmeTiffStripedImage,
    OmeTiffThumbnailImage,
    OmeTiffTiledImage,
)
from opentile.geometry import Size, SizeMm
from opentile.jpeg import Jpeg
from opentile.metadata import Metadata
from opentile.tiff_format import TiffFormat
from opentile.tiff_image import AssociatedTiffImage, LevelTiffImage, ThumbnailTiffImage
from opentile.tiler import Tiler


class OmeTiffTiler(Tiler):
    """Simple tiler for OME tiff. Works with images converted with QuPath using
    jpeg. Might report 'wrong' photometric interpretation. Does not support rgb
    images where the colors are separated. This could maybe be supported by
    using turbo-jpeg to losslessly merge the rgb components (assuming they have
    the same tables)."""

    def __init__(
        self,
        file: Union[str, Path, UPath, OpenTileFile],
        tile_size: int = 512,
        turbo_path: Optional[Union[str, Path]] = None,
        file_options: Optional[dict[str, Any]] = None,
    ):
        """Tiler for ome tiff file.

        Parameters
        ----------
        file: Union[str, Path, UPath, OpenTileFile]
            Filepath to a ome tiff TiffFile or an opened ome tiff OpenTileFile.
        tile_size: int = 512
            Tile size for levels that are not natively tiled (e.g. strip-stored).
        turbo_path: Optional[Union[str, Path]] = None
            Path to turbojpeg (dll or so).
        file_options: Optional[Dict[str, Any]] = None
            Options to pass to filesystem when opening file.
        """
        super().__init__(file, file_options)
        self._tile_size = Size(tile_size, tile_size)
        self._jpeg = Jpeg(turbo_path)
        self._base_mpp = self._get_mpp(self._level_series_index)

    @property
    def metadata(self) -> Metadata:
        """Metadata parsing not implemented for OmeTiff."""
        return Metadata()

    @property
    def format(self) -> TiffFormat:
        return TiffFormat.OME_TIFF

    @classmethod
    def supported(cls, tiff_file: TiffFile) -> bool:
        return tiff_file.is_ome

    @staticmethod
    def _is_level_series(series: TiffPageSeries) -> bool:
        return (
            not OmeTiffTiler._is_label_series(series)
            and not OmeTiffTiler._is_overview_series(series)
            and not OmeTiffTiler._is_thumbnail_series(series)
        )

    @staticmethod
    def _is_label_series(series: TiffPageSeries) -> bool:
        return series.name.strip() == "label"

    @staticmethod
    def _is_overview_series(series: TiffPageSeries) -> bool:
        return series.name.strip() == "macro"

    @staticmethod
    def _is_thumbnail_series(series: TiffPageSeries) -> bool:
        return series.name.strip() == "thumbnail"

    def _get_mpp(self, series_index: int) -> SizeMm:
        mpp = self._get_optional_mpp(series_index)
        if mpp is None:
            raise ValueError("Could not find physical size of x and y.")
        return mpp

    @cached_property
    def _ome(self) -> ome_types.OME:
        assert self._file.tiff.ome_metadata is not None
        return ome_types.from_xml(self._file.tiff.ome_metadata)

    def _get_optional_mpp(self, series_index: int) -> Optional[SizeMm]:
        pixels = self._ome.images[series_index].pixels
        if (
            pixels.physical_size_x_unit != UnitsLength.MICROMETER
            or pixels.physical_size_y_unit != UnitsLength.MICROMETER
        ):
            raise NotImplementedError("Only um physical size implemented.")
        mpp_x, mpp_y = pixels.physical_size_x, pixels.physical_size_y
        if mpp_x is None or mpp_y is None:
            return None
        return SizeMm(mpp_x, mpp_y)

    def _get_focal_plane_and_optical_path(
        self, series_index: int, page_index: int
    ) -> tuple[float, str]:
        """Map a level page index to its (focal plane in um, optical path).

        A series can hold several pages for the Z (focal plane) and C (optical path)
        dimensions; the page order follows the series axes (excluding the in-page Y,
        X and sample S axes). The focal plane is the Z index times the physical z
        spacing; the optical path is the C index."""
        series = self._file.series[series_index]
        page_axes = [axis for axis in series.axes if axis not in "YXS"]
        page_sizes = [series.shape[series.axes.index(axis)] for axis in page_axes]
        indices = np.unravel_index(page_index, page_sizes) if page_sizes else ()
        axis_index = {axis: int(i) for axis, i in zip(page_axes, indices)}
        physical_size_z = self._ome.images[series_index].pixels.physical_size_z
        z_index = axis_index.get("Z", 0)
        focal_plane = float(z_index * physical_size_z) if physical_size_z else 0.0
        return focal_plane, str(axis_index.get("C", 0))

    def _create_level(self, level: int, page: int = 0) -> LevelTiffImage:
        tiff_page = self._get_tiff_page(self._level_series_index, level, page)
        focal_plane, optical_path = self._get_focal_plane_and_optical_path(
            self._level_series_index, page
        )
        if tiff_page.is_tiled:
            return OmeTiffTiledImage(
                tiff_page,
                self._file,
                self._base_size,
                self._base_mpp,
                focal_plane,
                optical_path,
            )
        if tiff_page.compression == COMPRESSION.JPEG:
            # Untiled single jpeg frame that is re-tiled by lossless jpeg cropping.
            return OmeTiffOneFrameImage(
                tiff_page,
                self._file,
                self._base_size,
                self._tile_size,
                self._base_mpp,
                focal_plane,
                optical_path,
                self._jpeg,
            )
        # Strip-stored (e.g. uncompressed) level: decode once and serve a tile grid.
        return OmeTiffStripedImage(
            tiff_page,
            self._file,
            self._base_size,
            self._tile_size,
            self._base_mpp,
            focal_plane,
            optical_path,
        )

    def _create_label(self, page: int = 0) -> AssociatedTiffImage:
        if self._label_series_index is None:
            raise ValueError("No label detected in file")
        return self._get_associated_image(self._label_series_index, page)

    def _create_overview(self, page: int = 0) -> AssociatedTiffImage:
        if self._overview_series_index is None:
            raise ValueError("No overview detected in file")
        return self._get_associated_image(self._overview_series_index, page)

    def _create_thumbnail(self, page: int = 0) -> ThumbnailTiffImage:
        if self._thumbnail_series_index is None:
            raise ValueError("No thumbnail detected in file")
        tiff_page = self._get_tiff_page(self._thumbnail_series_index, 0, page)
        return OmeTiffThumbnailImage(
            tiff_page,
            self._file,
            self._base_size,
            self._get_mpp(self._thumbnail_series_index),
        )

    def _get_associated_image(self, series: int, page: int = 0) -> AssociatedTiffImage:
        tiff_page = self._get_tiff_page(series, 0, page)
        return OmeTiffAssociatedImage(
            tiff_page,
            self._file,
            self._get_optional_mpp(series),
        )
