#    Copyright 2026 SECTRA AB
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

"""Tiler for reading tiles from PerkinElmer/Akoya qptiff files.

qptiff stores a single-file pyramidal tiled tiff (BigTIFF for large scans) holding, in
order: the baseline image(s), a thumbnail, the reduced-resolution levels, and, for
whole-slide scans, a macro (overview) and a label image. Every directory is classified
by the ``ImageType`` element of its ImageDescription rather than by position, so files
missing an optional image are still read correctly.

Brightfield scans hold one RGB image per level. Fluorescence and unmixed multispectral
files instead hold one grayscale image per band; each band is exposed as its own level
image, with the band name (e.g. ``DAPI``) as ``optical_path``.
"""

from pathlib import Path
from typing import Any, Optional, Union

from tifffile import TiffFile, TiffFrame, TiffPage, TiffPageSeries
from upath import UPath

from opentile.exceptions import MissingAssociatedImageError
from opentile.file import OpenTileFile
from opentile.formats.qptiff.qptiff_image import (
    QptiffAssociatedImage,
    QptiffLevelImage,
    QptiffStripedLevelImage,
    QptiffThumbnailImage,
)
from opentile.formats.qptiff.qptiff_metadata import (
    QptiffMetadata,
    band_name,
    image_type,
)
from opentile.jpeg import Jpeg
from opentile.metadata import Metadata
from opentile.tiff_format import TiffFormat
from opentile.tiff_image import (
    AssociatedTiffImage,
    LevelTiffImage,
    ThumbnailTiffImage,
)
from opentile.tiler import Tiler


class QptiffTiler(Tiler):
    def __init__(
        self,
        file: Union[str, Path, UPath, OpenTileFile],
        turbo_path: Optional[Union[str, Path]] = None,
        file_options: Optional[dict[str, Any]] = None,
    ):
        """Tiler for qptiff file.

        Parameters
        ----------
        file: Union[str, Path, UPath, OpenTileFile]
            Filepath to a qptiff file or an opened qptiff OpenTileFile.
        turbo_path: Optional[Union[str, Path]] = None
            Path to turbojpeg (dll or so).
        file_options: Optional[Dict[str, Any]] = None
            Options to pass to filesystem when opening file.
        """
        super().__init__(file, file_options)
        self._jpeg = Jpeg(turbo_path)
        self._metadata = QptiffMetadata(self._base_page)
        base_mpp = self._metadata.mpp
        if base_mpp is None:
            raise ValueError("Could not find pixel size of image.")
        self._base_mpp = base_mpp

    @property
    def metadata(self) -> Metadata:
        return self._metadata

    @property
    def format(self) -> TiffFormat:
        return TiffFormat.QPTIFF

    @classmethod
    def supported(cls, tiff_file: TiffFile) -> bool:
        return tiff_file.is_qpi

    def _create_level(self, level: int, page: int = 0) -> LevelTiffImage:
        # `page` is the band index within the level (1 band for brightfield RGB).
        tiff_page = self._get_tiff_page(self._level_series_index, level, page)
        # Multi-band files name their bands (e.g. "DAPI"); RGB files do not.
        optical_path = band_name(tiff_page) or str(page + 1)
        if tiff_page.is_tiled:
            return QptiffLevelImage(
                tiff_page,
                self._file,
                self._base_size,
                self._base_mpp,
                optical_path,
            )
        return QptiffStripedLevelImage(
            tiff_page,
            self._file,
            self._base_size,
            self._base_mpp,
            self._jpeg,
            optical_path,
        )

    def _create_label(self, page: int = 0) -> AssociatedTiffImage:
        if self._label_series_index is None:
            raise MissingAssociatedImageError("No label series found in this file.")
        return QptiffAssociatedImage(
            self._get_tiff_page(self._label_series_index, 0, page),
            self._file,
            self._jpeg,
        )

    def _create_overview(self, page: int = 0) -> AssociatedTiffImage:
        if self._overview_series_index is None:
            raise MissingAssociatedImageError("No overview series found in this file.")
        return QptiffAssociatedImage(
            self._get_tiff_page(self._overview_series_index, 0, page),
            self._file,
            self._jpeg,
        )

    def _create_thumbnail(self, page: int = 0) -> ThumbnailTiffImage:
        if self._thumbnail_series_index is None:
            raise MissingAssociatedImageError("No thumbnail series found in this file.")
        return QptiffThumbnailImage(
            self._get_tiff_page(self._thumbnail_series_index, 0, page),
            self._file,
            self._base_size,
            self._base_mpp,
            self._jpeg,
        )

    def _is_level_series(self, series: TiffPageSeries) -> bool:
        # Only the baseline series, the reduced-resolution directories are levels of it.
        return self._image_type(series) == "FullResolution"

    def _is_overview_series(self, series: TiffPageSeries) -> bool:
        return self._image_type(series) == "Overview"

    def _is_label_series(self, series: TiffPageSeries) -> bool:
        return self._image_type(series) == "Label"

    def _is_thumbnail_series(self, series: TiffPageSeries) -> bool:
        return self._image_type(series) == "Thumbnail"

    @staticmethod
    def _image_type(series: TiffPageSeries) -> Optional[str]:
        page = series.pages[0]
        if isinstance(page, TiffFrame):
            page = page.aspage()
        assert isinstance(page, TiffPage)
        return image_type(page)
