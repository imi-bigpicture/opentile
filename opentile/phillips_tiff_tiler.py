import io
from functools import cached_property
from pathlib import Path
from typing import Dict
from xml.etree import ElementTree as etree

from tifffile.tifffile import FileHandle, TiffPage, TiffPageSeries

from opentile.geometry import Size, SizeMm
from opentile.interface import NativeTiledPage, Tiler
from opentile.turbojpeg_patch import TurboJPEG_patch as TurboJPEG
from opentile.utils import (calculate_mpp, calculate_pyramidal_index,
                            split_and_cast_text)


class PhillipsTiffTiledPage(NativeTiledPage):
    def __init__(
        self,
        page: TiffPage,
        fh: FileHandle,
        base_shape: Size,
        base_mpp: SizeMm,
        jpeg: TurboJPEG
    ):
        """TiledPage for Phillips Tiff-page.

        Parameters
        ----------
        page: TiffPage
            TiffPage defining the page.
        fh: NdpiFileHandle
            Filehandler to read data from.
        base_shape: Size
            Size of base level in pyramid.
        base_mpp: SizeMm
            Mpp (um/pixel) for base level in pyramid.
        jpeg: TurboJpeg
            TurboJpeg instance to use.
        """
        super().__init__(page, fh)
        self._jpeg = jpeg
        self._base_shape = base_shape
        self._base_mpp = base_mpp
        self._pyramid_index = calculate_pyramidal_index(
            self._base_shape,
            self.image_size
        )
        self._mpp = calculate_mpp(self._base_mpp, self.pyramid_index)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._page}, {self._fh}, "
            f"{self._base_shape}, {self._base_mpp}, {self._jpeg})"
        )

    def __str__(self) -> str:
        return f"{type(self).__name__} of page {self._page}"

    @property
    def pyramid_index(self) -> int:
        return self._pyramid_index

    @property
    def pixel_spacing(self) -> SizeMm:
        """Return pixel spacing in mm per pixel."""
        return self.mpp * 1000

    @property
    def mpp(self) -> SizeMm:
        """Return pixel spacing in um per pixel."""
        return self._mpp

    @cached_property
    def blank_tile(self) -> bytes:
        """Create a blank (white) tile from a valid tile."""
        # Todo, figure out what color to fill with.
        try:
            valid_frame_index = next(
                index
                for index, datalength in enumerate(self.page.databytecounts)
                if datalength != 0
            )
        except StopIteration:
            raise ValueError
        valid_frame = self._read_frame(valid_frame_index)
        valid_tile = self._add_jpeg_tables(valid_frame)
        return self._jpeg.fill_image(valid_tile)

    def _read_frame(self, frame_index: int) -> bytes:
        """Read frame at frame index from page. Return blank tile if tile is
        sparse."""
        if (
            frame_index >= len(self.page.databytecounts) or
            self.page.databytecounts[frame_index] == 0
        ):
            # Sparse tile
            return self.blank_tile
        return super()._read_frame(frame_index)

    def _add_jpeg_tables(self, frame: bytes) -> bytes:
        """Add jpeg tables to frame."""
        # frame has jpeg header but no tables. Insert tables before start
        # of scan tag.
        start_of_scan = frame.find(bytes([0xFF, 0xDA]))
        with io.BytesIO() as buffer:
            buffer.write(frame[0:start_of_scan])
            tables = self.page.jpegtables[2:-2]  # No start and end tags
            buffer.write(tables)
            buffer.write(frame[start_of_scan:None])
            return buffer.getvalue()


class PhillipsTiffTiler(Tiler):
    def __init__(self, filepath: Path, turbo_path: Path):
        """Tiler for Phillips tiff file.

        Parameters
        ----------
        filepath: str
            File path to Phillips tiff file.
        turbo_path: Path
            Path to turbojpeg (dll or so).
        """
        super().__init__(filepath)
        self._fh = self._tiff_file.filehandle

        self._turbo_path = turbo_path
        self._jpeg = TurboJPEG(self._turbo_path)

        self._level_series_index = 0
        for series_index, series in enumerate(self.series):
            if self.is_label(series):
                self._label_series_index = series_index
            elif self.is_overview(series):
                self._overview_series_index = series_index

    @cached_property
    def base_mpp(self) -> SizeMm:
        """Return pixel spacing in um/pixel for base level."""
        return SizeMm.from_tuple(self.properties['pixel_spacing']) / 1000.0

    @cached_property
    def properties(self) -> Dict[str, any]:
        """Return dictionary with phillips tiff file properties."""
        metadata = etree.fromstring(self._tiff_file.philips_metadata)
        pixel_spacing = None
        for element in metadata.iter():
            if element.tag == 'Attribute':
                name = element.attrib['Name']
                if name == 'DICOM_PIXEL_SPACING' and pixel_spacing is None:
                    pixel_spacing = split_and_cast_text(element.text, float)
                elif name == 'DICOM_ACQUISITION_DATETIME':
                    aquisition_datatime = element.text
                elif name == 'DICOM_DEVICE_SERIAL_NUMBER':
                    device_serial_number = element.text
                elif name == 'DICOM_MANUFACTURER':
                    manufacturer = element.text
                elif name == 'DICOM_SOFTWARE_VERSIONS':
                    software_versions = split_and_cast_text(element.text, str)
                elif name == 'DICOM_LOSSY_IMAGE_COMPRESSION_METHOD':
                    lossy_image_compression_method = split_and_cast_text(
                        element.text, str
                    )
                elif name == 'DICOM_LOSSY_IMAGE_COMPRESSION_RATIO':
                    lossy_image_compression_ratio = split_and_cast_text(
                        element.text, float
                    )
                elif name == 'DICOM_PHOTOMETRIC_INTERPRETATION':
                    photometric_interpretation = element.text
                elif name == 'DICOM_BITS_ALLOCATED':
                    bits_allocated = int(element.text)
                elif name == 'DICOM_BITS_STORED':
                    bits_stored = int(element.text)
                elif name == 'DICOM_HIGH_BIT':
                    high_bit = int(element.text)
                elif name == 'DICOM_PIXEL_REPRESENTATION':
                    pixel_representation = element.text
        return {
            'pixel_spacing': pixel_spacing,
            'aquisition_datatime': aquisition_datatime,
            'device_serial_number': device_serial_number,
            'manufacturer': manufacturer,
            'software_versions': software_versions,
            'lossy_image_compression_method': lossy_image_compression_method,
            'lossy_image_compression_ratio': lossy_image_compression_ratio,
            'photometric_interpretation': photometric_interpretation,
            'bits_allocated': bits_allocated,
            'bits_stored': bits_stored,
            'high_bit': high_bit,
            'pixel_representation': pixel_representation
        }

    def get_page(
        self,
        series: int,
        level: int,
        page: int = 0
    ) -> PhillipsTiffTiledPage:
        """Return PhillipsTiffTiledPage for series, level, page.
        """
        tiff_page = self.series[series].levels[level].pages[page]
        return PhillipsTiffTiledPage(
            tiff_page,
            self._fh,
            self.base_size,
            self.base_mpp,
            self._jpeg
        )

    @staticmethod
    def is_overview(series: TiffPageSeries) -> bool:
        """Return true if series is a overview series."""
        return series.pages[0].description.find('Macro') > - 1

    @staticmethod
    def is_label(series: TiffPageSeries) -> bool:
        """Return true if series is a label series."""
        return series.pages[0].description.find('Label') > - 1

    @staticmethod
    def get_associated_mpp_from_page(page: TiffPage):
        """Return mpp (um/pixel) for associated image (label or
        macro) from page."""
        pixel_size_start_string = 'pixelsize=('
        pixel_size_start = page.description.find(pixel_size_start_string)
        pixel_size_end = page.description.find(')', pixel_size_start)
        pixel_size_string = page.description[
            pixel_size_start+len(pixel_size_start_string):pixel_size_end
        ]
        pixel_spacing = SizeMm.from_tuple(
            [float(v) for v in pixel_size_string.replace('"', '').split(',')]
        )
        return pixel_spacing / 1000.0
