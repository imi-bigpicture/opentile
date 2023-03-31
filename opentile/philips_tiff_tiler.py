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

from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
from xml.etree import ElementTree

from tifffile.tifffile import FileHandle, TiffFile, TiffPage, TiffPageSeries

from opentile.common import NativeTiledPage, Tiler
from opentile.geometry import Size, SizeMm
from opentile.jpeg import Jpeg
from opentile.metadata import Metadata


class PhilipsTiffTiledPage(NativeTiledPage):
    def __init__(
        self,
        page: TiffPage,
        fh: FileHandle,
        base_shape: Size,
        base_mpp: SizeMm,
        jpeg: Jpeg,
    ):
        """OpenTiledPage for Philips Tiff-page.

        Parameters
        ----------
        page: TiffPage
            TiffPage defining the page.
        fh: FileHandle
            Filehandler to read data from.
        base_shape: Size
            Size of base level in pyramid.
        base_mpp: SizeMm
            Mpp (um/pixel) for base level in pyramid.
        jpeg: Jpeg
            Jpeg instance to use.
        """
        super().__init__(page, fh)
        self._jpeg = jpeg
        self._base_shape = base_shape
        self._base_mpp = base_mpp
        self._pyramid_index = self._calculate_pyramidal_index(self._base_shape)
        self._mpp = self._calculate_mpp(self._base_mpp)
        self._blank_tile = self._create_blank_tile()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._page}, {self._fh}, "
            f"{self._base_shape}, {self._base_mpp}, {self._jpeg})"
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
    def blank_tile(self) -> bytes:
        """Return blank tile."""
        return self._blank_tile

    def _create_blank_tile(self, luminance: float = 1.0) -> bytes:
        """Create a blank tile from a valid tile. Uses the first found
        valid frame (first frame with non-zero value in databytescounts) and
        fills that image with white.

        Parameters
        ----------
        luminance: float = 1.0
            Luminance for tile, 0 = black, 1 = white.

        Returns
        ----------
        bytes:
            Frame bytes from blank tile.

        """
        # Todo, figure out what color to fill with.
        try:
            # Get first frame in page that is not 0 bytes
            valid_frame_index = next(
                index
                for index, datalength in enumerate(self.page.databytecounts)
                if datalength != 0
            )
        except StopIteration:
            raise ValueError("Could not find valid frame in page.")
        tile = self._read_frame(valid_frame_index)
        if self.page.jpegtables is not None:
            tile = Jpeg.add_jpeg_tables(tile, self.page.jpegtables, False)
        tile = self._jpeg.fill_frame(tile, luminance)
        return tile

    def _read_frame(self, index: int) -> bytes:
        """Read frame at frame index from page. Return blank tile if tile is
        sparse (length of frame is zero or frame indexis outside length of
        frames)

        Parameters
        ----------
        index: int
            Frame index to read from page.

        Returns
        ----------
        bytes:
            Frame bytes from frame index or blank tile.

        """
        if (
            index >= len(self.page.databytecounts)
            or self.page.databytecounts[index] == 0
        ):
            # Sparse tile
            return self.blank_tile
        return super()._read_frame(index)


CastType = TypeVar("CastType", int, float, str)


class PhilipsMetadata(Metadata):
    _scanner_manufacturer: Optional[str] = None
    _scanner_software_versions: Optional[List[str]] = None
    _scanner_serial_number: Optional[str] = None
    _pixel_spacing: Optional[Tuple[float, float]] = None

    TAGS = [
        "DICOM_PIXEL_SPACING",
        "DICOM_ACQUISITION_DATETIME",
        "DICOM_MANUFACTURER",
        "DICOM_SOFTWARE_VERSIONS",
        "DICOM_DEVICE_SERIAL_NUMBER",
        "DICOM_LOSSY_IMAGE_COMPRESSION_METHOD",
        "DICOM_LOSSY_IMAGE_COMPRESSION_RATIO",
        "DICOM_BITS_ALLOCATED",
        "DICOM_BITS_STORED",
        "DICOM_HIGH_BIT",
        "DICOM_PIXEL_REPRESENTATION",
    ]

    def __init__(self, tiff_file: TiffFile):
        if tiff_file.philips_metadata is None:
            return

        metadata = ElementTree.fromstring(tiff_file.philips_metadata)
        self._tags: Dict[str, Optional[str]] = {tag: None for tag in self.TAGS}
        for element in metadata.iter():
            if element.tag == "Attribute" and element.text is not None:
                name = element.attrib["Name"]
                if name in self._tags and self._tags[name] is None:
                    self._tags[name] = element.text

    @property
    def scanner_manufacturer(self) -> Optional[str]:
        return self._tags["DICOM_MANUFACTURER"]

    @cached_property
    def scanner_software_versions(self) -> Optional[List[str]]:
        print(self._tags["DICOM_SOFTWARE_VERSIONS"])
        if self._tags["DICOM_SOFTWARE_VERSIONS"] is None:
            return None
        return self._split_and_cast_text(self._tags["DICOM_SOFTWARE_VERSIONS"], str)

    @property
    def scanner_serial_number(self) -> Optional[str]:
        return self._tags["DICOM_DEVICE_SERIAL_NUMBER"]

    @cached_property
    def aquisition_datetime(self) -> Optional[datetime]:
        if self._tags["DICOM_ACQUISITION_DATETIME"] is None:
            return None
        try:
            return datetime.strptime(
                self._tags["DICOM_ACQUISITION_DATETIME"], r"%Y%m%d%H%M%S.%f"
            )
        except ValueError:
            return None

    @cached_property
    def pixel_spacing(self) -> Optional[Tuple[float, float]]:
        if self._tags["DICOM_PIXEL_SPACING"] is None:
            return None
        return tuple(
            self._split_and_cast_text(self._tags["DICOM_PIXEL_SPACING"], float)[0:2]
        )

    @cached_property
    def properties(self) -> Dict[str, Any]:
        properties: Dict[str, Any] = {}
        if self._tags["DICOM_LOSSY_IMAGE_COMPRESSION_METHOD"] is not None:
            properties["lossy_image_compression_method"] = self._split_and_cast_text(
                self._tags["DICOM_LOSSY_IMAGE_COMPRESSION_METHOD"], str
            )
        if self._tags["DICOM_LOSSY_IMAGE_COMPRESSION_RATIO"] is not None:
            properties["lossy_image_compression_ratio"] = self._split_and_cast_text(
                self._tags["DICOM_LOSSY_IMAGE_COMPRESSION_RATIO"], float
            )[0]
        if self._tags["DICOM_BITS_ALLOCATED"] is not None:
            properties["bits_allocated"] = int(self._tags["DICOM_BITS_ALLOCATED"])
        if self._tags["DICOM_BITS_STORED"] is not None:
            properties["bits_stored"] = int(self._tags["DICOM_BITS_STORED"])
        if self._tags["DICOM_HIGH_BIT"] is not None:
            properties["high_bit"] = int(self._tags["DICOM_HIGH_BIT"])

        if self._tags["DICOM_PIXEL_REPRESENTATION"] is not None:
            properties["pixel_representation"] = self._tags[
                "DICOM_PIXEL_REPRESENTATION"
            ]
        return properties

    @staticmethod
    def _split_and_cast_text(string: str, cast_type: Type[CastType]) -> List[CastType]:
        return [cast_type(element) for element in string.replace('"', "").split()]


class PhilipsTiffTiler(Tiler):
    def __init__(
        self, filepath: Union[str, Path], turbo_path: Optional[Union[str, Path]] = None
    ):
        """Tiler for Philips tiff file.

        Parameters
        ----------
        filepath: Union[str, Path]
            Filepath to a Philips-TiffFile.
        turbo_path: Optional[Union[str, Path]] = None
            Path to turbojpeg (dll or so).
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
        self._metadata = PhilipsMetadata(self._tiff_file)
        assert self._metadata.pixel_spacing is not None
        self._base_mpp = SizeMm.from_tuple(self._metadata.pixel_spacing) * 1000.0
        self._pages: Dict[Tuple[int, int, int], PhilipsTiffTiledPage] = {}

    @property
    def metadata(self) -> Metadata:
        return self._metadata

    @classmethod
    def supported(cls, tiff_file: TiffFile) -> bool:
        return tiff_file.is_philips

    def get_page(self, series: int, level: int, page: int = 0) -> PhilipsTiffTiledPage:
        """Return PhilipsTiffTiledPage for series, level, page."""
        if not (series, level, page) in self._pages:
            self._pages[series, level, page] = PhilipsTiffTiledPage(
                self._get_tiff_page(series, level, page),
                self._fh,
                self.base_size,
                self._base_mpp,
                self._jpeg,
            )
        return self._pages[series, level, page]

    @staticmethod
    def is_overview(series: TiffPageSeries) -> bool:
        """Return true if series is a overview series."""
        page = series.pages[0]
        assert isinstance(page, TiffPage)
        return page.description.find("Macro") > -1

    @staticmethod
    def is_label(series: TiffPageSeries) -> bool:
        """Return true if series is a label series."""
        page = series.pages[0]
        assert isinstance(page, TiffPage)
        return page.description.find("Label") > -1

    @staticmethod
    def _get_associated_mpp_from_page(page: TiffPage):
        """Return mpp (um/pixel) for associated image (label or
        macro) from page."""
        pixel_size_start_string = "pixelsize=("
        pixel_size_start = page.description.find(pixel_size_start_string)
        pixel_size_end = page.description.find(")", pixel_size_start)
        pixel_size_string = page.description[
            pixel_size_start + len(pixel_size_start_string) : pixel_size_end
        ]
        pixel_spacing = SizeMm.from_tuple(
            [float(v) for v in pixel_size_string.replace('"', "").split(",")]
        )
        return pixel_spacing / 1000.0
