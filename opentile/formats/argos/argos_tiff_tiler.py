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

"""Tiler for reading tiles from Argos avs files.

Argos stores a single-file pyramidal tiled BigTIFF with sparse JPEG tiles (missing tiles
have zero offset and byte count, served as blank tiles), followed by two striped images:
the thumbnail (``Map``, second-to-last directory) and the overview (``Macro``, last).
The directory order is positional per the format spec, so the associated images are
classified by position rather than by tifffile's unreliable series names.

Stacked (z-stack) files store the focal planes as the Z axis of the base series; each
plane is exposed as its own level image with its ``focal_plane`` set from the ``MinZ``/
``ZRange`` metadata. Argos has no dedicated label image, so the label is cropped from
the right side of the overview (see `ArgosLabelImage`).
"""

from pathlib import Path
from typing import Any, Optional, Union

from tifffile import TiffFile, TiffPageSeries
from upath import UPath

from opentile.exceptions import MissingAssociatedImageError
from opentile.file import OpenTileFile
from opentile.formats.argos.argos_tiff_image import (
    ArgosLabelImage,
    ArgosLevelTiffImage,
)
from opentile.formats.argos.argos_tiff_metadata import ARGOS_METADATA_TAG, ArgosMetadata
from opentile.geometry import SizeMm
from opentile.jpeg import Jpeg
from opentile.metadata import Metadata
from opentile.tiff_format import TiffFormat
from opentile.tiff_image import (
    AssociatedTiffImage,
    LevelTiffImage,
    StripedAssociatedImage,
    StripedThumbnailImage,
    ThumbnailTiffImage,
)
from opentile.tiler import Tiler


class ArgosTiffTiler(Tiler):
    def __init__(
        self,
        file: Union[str, Path, UPath, OpenTileFile],
        turbo_path: Optional[Union[str, Path]] = None,
        label_crop_position: float = 0.76,
        file_options: Optional[dict[str, Any]] = None,
    ):
        """Tiler for Argos avs file.

        Parameters
        ----------
        file: Union[str, Path, UPath, OpenTileFile]
            Filepath to an Argos avs file or an opened Argos OpenTileFile.
        turbo_path: Optional[Union[str, Path]] = None
            Path to turbojpeg (dll or so).
        label_crop_position: float = 0.76
            Left edge of the label as a fraction of the macro width; the label is
            cropped from here to the right edge of the overview (macro) image. The crop
            snaps down to the nearest JPEG MCU, so nearby fractions give the same crop.
        file_options: Optional[Dict[str, Any]] = None
            Options to pass to filesystem when opening file.
        """
        super().__init__(file, file_options)
        self._jpeg = Jpeg(turbo_path)
        self._label_crop_position = label_crop_position
        self._metadata = ArgosMetadata(self._base_page)
        assert self._metadata.pixel_spacing is not None
        self._base_mpp = SizeMm.from_tuple(self._metadata.pixel_spacing) * 1000.0
        # Positional layout: pyramid(s) first, then thumbnail (second-to-last) and
        # overview (last). tifffile's series names are not a reliable classifier.
        series_count = len(self._file.series)
        if series_count >= 3:
            self._thumbnail_series_index = series_count - 2
            self._overview_series_index = series_count - 1
            # The label is cropped from the overview (macro); it has no dedicated IFD.
            self._label_series_index = self._overview_series_index

    @property
    def metadata(self) -> Metadata:
        return self._metadata

    @property
    def format(self) -> TiffFormat:
        return TiffFormat.ARGOS

    @classmethod
    def supported(cls, tiff_file: TiffFile) -> bool:
        page = tiff_file.pages.first
        if not page.is_tiled:
            return False
        tag = page.tags.get(ARGOS_METADATA_TAG)
        return (
            tag is not None
            and isinstance(tag.value, str)
            and "Argos.Scan.Metadata" in tag.value
        )

    def _create_level(self, level: int, page: int = 0) -> LevelTiffImage:
        # `page` is the z-plane index within the level (base series axes are ZYXS).
        return ArgosLevelTiffImage(
            self._get_tiff_page(self._level_series_index, level, page),
            self._file,
            self._base_size,
            self._base_mpp,
            self._jpeg,
            self._metadata.focal_plane(page),
        )

    def _create_label(self, page: int = 0) -> AssociatedTiffImage:
        if self._label_series_index is None:
            raise MissingAssociatedImageError("No label series found in this file.")
        return ArgosLabelImage(
            self._get_tiff_page(self._label_series_index, 0, page),
            self._file,
            self._jpeg,
            self._label_crop_position,
        )

    def _create_overview(self, page: int = 0) -> AssociatedTiffImage:
        if self._overview_series_index is None:
            raise MissingAssociatedImageError("No overview series found in this file.")
        return StripedAssociatedImage(
            self._get_tiff_page(self._overview_series_index, 0, page),
            self._file,
            self._jpeg,
        )

    def _create_thumbnail(self, page: int = 0) -> ThumbnailTiffImage:
        if self._thumbnail_series_index is None:
            raise MissingAssociatedImageError(
                "No thumbnail series found in this file."
            )
        return StripedThumbnailImage(
            self._get_tiff_page(self._thumbnail_series_index, 0, page),
            self._file,
            self._base_size,
            self._base_mpp,
            self._jpeg,
        )

    def _is_level_series(self, series: TiffPageSeries) -> bool:
        return series.index == 0

    def _is_overview_series(self, series: TiffPageSeries) -> bool:
        return False

    def _is_label_series(self, series: TiffPageSeries) -> bool:
        return False

    def _is_thumbnail_series(self, series: TiffPageSeries) -> bool:
        return False
