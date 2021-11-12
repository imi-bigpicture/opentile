from pathlib import Path
from typing import Any, Dict, List, Type
from xml.etree import ElementTree as etree

from tifffile.tifffile import FileHandle, TiffPage, TiffPageSeries

from opentile.common import NativeTiledPage, Tiler
from opentile.geometry import Size, SizeMm
from opentile.turbojpeg_patch import TurboJPEG_patch as TurboJPEG


class PhilipsTiffTiledPage(NativeTiledPage):
    def __init__(
        self,
        page: TiffPage,
        fh: FileHandle,
        base_shape: Size,
        base_mpp: SizeMm,
        jpeg: TurboJPEG
    ):
        """OpenTiledPage for Philips Tiff-page.

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
        valid_frame = self._read_frame(valid_frame_index)
        valid_tile = self._add_jpeg_tables(valid_frame)
        return self._jpeg.fill_image(valid_tile, luminance)

    def _read_frame(self, frame_index: int) -> bytes:
        """Read frame at frame index from page. Return blank tile if tile is
        sparse (length of frame is zero or frame indexis outside length of
        frames)

        Parameters
        ----------
        frame_index: int
            Frame index to read from page.

        Returns
        ----------
        bytes:
            Frame bytes from frame index or blank tile.

        """
        if (
            frame_index >= len(self.page.databytecounts) or
            self.page.databytecounts[frame_index] == 0
        ):
            # Sparse tile
            return self.blank_tile
        return super()._read_frame(frame_index)


class PhilipsTiffTiler(Tiler):
    def __init__(self, filepath: Path, turbo_path: Path = None):
        """Tiler for Philips tiff file.

        Parameters
        ----------
        filepath: Path
            Filepath to a Philips-TiffFile.
        turbo_path: Path = None
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
        self._properties = self._read_properties()
        self._base_mpp = SizeMm.from_tuple(
            self.properties['pixel_spacing']
        ) / 1000.0

    @property
    def base_mpp(self) -> SizeMm:
        """Return pixel spacing in um/pixel for base level."""
        return self._base_mpp

    @property
    def properties(self) -> Dict[str, Any]:
        """Return dictionary with philips tiff file properties."""
        return self._properties

    def get_page(
        self,
        series: int,
        level: int,
        page: int = 0
    ) -> PhilipsTiffTiledPage:
        """Return PhilipsTiffTiledPage for series, level, page."""
        if not (series, level, page) in self._pages:
            tiff_page = self.series[series].levels[level].pages[page]
            self._pages[series, level, page] = PhilipsTiffTiledPage(
                tiff_page,
                self._fh,
                self.base_size,
                self.base_mpp,
                self._jpeg
            )
        return self._pages[series, level, page]

    @staticmethod
    def is_overview(series: TiffPageSeries) -> bool:
        """Return true if series is a overview series."""
        return series.pages[0].description.find('Macro') > - 1

    @staticmethod
    def is_label(series: TiffPageSeries) -> bool:
        """Return true if series is a label series."""
        return series.pages[0].description.find('Label') > - 1

    @staticmethod
    def _get_associated_mpp_from_page(page: TiffPage):
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

    @staticmethod
    def _split_and_cast_text(string: str, cast_type: Type) -> List[Any]:
        return [
            cast_type(element) for element in string.replace('"', '').split()
        ]

    def _read_properties(self) -> Dict[str, Any]:
        """Return dictionary with philips tiff file properties."""
        metadata = etree.fromstring(self._tiff_file.philips_metadata)
        pixel_spacing = None
        for element in metadata.iter():
            if element.tag == 'Attribute':
                name = element.attrib['Name']
                if name == 'DICOM_PIXEL_SPACING' and pixel_spacing is None:
                    pixel_spacing = self._split_and_cast_text(
                        element.text,
                        float
                    )
                elif name == 'DICOM_ACQUISITION_DATETIME':
                    aquisition_datatime = element.text
                elif name == 'DICOM_DEVICE_SERIAL_NUMBER':
                    device_serial_number = element.text
                elif name == 'DICOM_MANUFACTURER':
                    manufacturer = element.text
                elif name == 'DICOM_SOFTWARE_VERSIONS':
                    software_versions = self._split_and_cast_text(
                        element.text,
                        str
                    )
                elif name == 'DICOM_LOSSY_IMAGE_COMPRESSION_METHOD':
                    lossy_image_compression_method = self._split_and_cast_text(
                        element.text,
                        str
                    )
                elif name == 'DICOM_LOSSY_IMAGE_COMPRESSION_RATIO':
                    lossy_image_compression_ratio = self._split_and_cast_text(
                        element.text,
                        float
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
