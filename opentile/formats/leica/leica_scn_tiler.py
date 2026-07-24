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

"""Tiler for reading tiles from Leica SCN files.

Leica SCN stores a single-file pyramidal tiled JPEG BigTIFF with a non-standard XML
slide description (parsed by tifffile as ``TiffFile.scn_metadata``). tifffile splits the
file into one series per SCN ``image``; the collection-spanning macro image (view offset
0,0) is exposed as the overview, and the largest main image is served as the pyramid
levels. Tiles do not overlap. Z-stacks and multi-channel (fluorescence) SCN files are
not supported.
"""

from pathlib import Path
from typing import Any, Optional, Union

from tifffile import TiffFile, TiffPageSeries
from upath import UPath

from opentile.exceptions import MissingAssociatedImageError, UnsupportedFileError
from opentile.file import OpenTileFile
from opentile.formats.leica.leica_scn_image import (
    LeicaScnAssociatedImage,
    LeicaScnLabelImage,
    LeicaScnLevelTiffImage,
)
from opentile.formats.leica.leica_scn_metadata import LeicaScnMetadata
from opentile.metadata import Metadata
from opentile.tiff_format import TiffFormat
from opentile.tiff_image import (
    AssociatedTiffImage,
    LevelTiffImage,
    ThumbnailTiffImage,
)
from opentile.tiler import Tiler


class LeicaScnTiler(Tiler):
    def __init__(
        self,
        file: Union[str, Path, UPath, OpenTileFile],
        label_crop_position: float = 0.72,
        file_options: Optional[dict[str, Any]] = None,
    ):
        """Tiler for Leica SCN file.

        Parameters
        ----------
        file: Union[str, Path, UPath, OpenTileFile]
            Filepath to a Leica SCN file or an opened Leica SCN OpenTileFile.
        label_crop_position: float = 0.72
            Top edge of the label as a fraction of the macro height; the label is
            cropped from here to the bottom of the overview (macro) image. The crop is
            rounded down to a whole tile row, so nearby fractions give the same crop.
        file_options: Optional[Dict[str, Any]] = None
            Options to pass to filesystem when opening file.
        """
        # The base class loop classifies the series (via the static methods below,
        # which read the SCN XML through `series.parent`) and computes the base
        # page/size from the resulting level series.
        super().__init__(file, file_options)
        self._label_crop_position = label_crop_position
        self._metadata = LeicaScnMetadata(self._scn_metadata(self._file.tiff))
        self._base_mpp = self._metadata.main_image.mpp
        # The label is cropped from the overview (macro); it has no dedicated series.
        self._label_series_index = self._overview_series_index

    @property
    def metadata(self) -> Metadata:
        return self._metadata

    @property
    def format(self) -> TiffFormat:
        return TiffFormat.LEICA_SCN

    @classmethod
    def supported(cls, tiff_file: TiffFile) -> bool:
        return tiff_file.is_scn

    @staticmethod
    def _scn_metadata(tiff_file: TiffFile) -> str:
        scn_xml = tiff_file.scn_metadata
        if scn_xml is None:
            raise UnsupportedFileError(
                "File is not a Leica SCN file: no SCN XML metadata."
            )
        return scn_xml

    def _create_level(self, level: int, page: int = 0) -> LevelTiffImage:
        return LeicaScnLevelTiffImage(
            self._get_tiff_page(self._level_series_index, level, page),
            self._file,
            self._base_size,
            self._base_mpp,
        )

    def _create_overview(self, page: int = 0) -> AssociatedTiffImage:
        if self._overview_series_index is None:
            raise MissingAssociatedImageError("No overview series found in this file.")
        return LeicaScnAssociatedImage(
            self._get_tiff_page(self._overview_series_index, 0, page), self._file
        )

    def _create_label(self, page: int = 0) -> AssociatedTiffImage:
        if self._label_series_index is None:
            raise MissingAssociatedImageError("No label series found in this file.")
        return LeicaScnLabelImage(
            self._get_tiff_page(self._label_series_index, 0, page),
            self._file,
            self._label_crop_position,
        )

    def _create_thumbnail(self, page: int = 0) -> ThumbnailTiffImage:
        raise NotImplementedError("Leica SCN files have no thumbnail image.")

    @staticmethod
    def _series_metadata(series: TiffPageSeries) -> Optional[LeicaScnMetadata]:
        """Parse the SCN XML reachable from the series' parent TiffFile, or None if
        this is not an SCN file (so the tiler ignores foreign series).

        Series classification runs during the base ``__init__`` loop, before this
        subclass has built ``self._metadata``, so the XML is read from the series
        rather than from the (not-yet-set) instance metadata."""
        scn_xml = getattr(series.parent, "scn_metadata", None)
        return LeicaScnMetadata(scn_xml) if scn_xml else None

    def _is_level_series(self, series: TiffPageSeries) -> bool:
        metadata = self._series_metadata(series)
        return metadata is not None and series.name == metadata.main_image.name

    def _is_overview_series(self, series: TiffPageSeries) -> bool:
        metadata = self._series_metadata(series)
        return metadata is not None and any(
            image.is_macro and image.name == series.name for image in metadata.images
        )

    def _is_label_series(self, series: TiffPageSeries) -> bool:
        return False

    def _is_thumbnail_series(self, series: TiffPageSeries) -> bool:
        return False
