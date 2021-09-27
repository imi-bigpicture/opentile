import io
from functools import cached_property

from tifffile.tifffile import (FileHandle, TiffFile, TiffPage,
                               svs_description_metadata)

from opentile.geometry import Size, SizeMm
from opentile.common import NativeTiledPage, Tiler
from opentile.utils import Jpeg


class SvsTiledPage(NativeTiledPage):
    def __init__(
        self,
        page: TiffPage,
        fh: FileHandle,
        base_shape: Size,
        base_mpp: SizeMm
    ):
        """OpenTiledPage for Svs Tiff-page.

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
        """
        super().__init__(page, fh)
        self._base_shape = base_shape
        self._base_mpp = base_mpp
        self._pyramid_index = self._calculate_pyramidal_index(self._base_shape)
        self._mpp = self._calculate_mpp(self._base_mpp)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._page}, {self._fh}, "
            f"{self._base_shape}, {self._base_mpp})"
        )

    @property
    def pixel_spacing(self) -> SizeMm:
        """Return pixel spacing in mm per pixel."""
        return self.mpp * 1000

    @property
    def mpp(self) -> SizeMm:
        """Return pixel spacing in um per pixel."""
        return self._mpp

    def _add_jpeg_tables(
        self,
        frame: bytes
    ) -> bytes:
        """Add jpeg tables to frame. Tables are insterted before 'start of
        scan'-tag, and leading 'start of image' and ending 'end of image' tags
        are removed from the header prior to insertion. Adds colorspace fix at
        end of header.

        Parameters
        ----------
        frame: bytes
            'Abbreviated' jpeg frame lacking jpeg tables.

        Returns
        ----------
        bytes:
            'Interchange' jpeg frame containg jpeg tables.

        """
        start_of_scan = frame.find(Jpeg.start_of_scan())
        with io.BytesIO() as buffer:
            buffer.write(frame[0:start_of_scan])
            buffer.write(self.page.jpegtables[2:-2])  # No start and end tags
            buffer.write(
                b"\xFF\xEE\x00\x0E\x41\x64\x6F\x62"
                b"\x65\x00\x64\x80\x00\x00\x00\x00"
            )  # colorspace fix
            buffer.write(frame[start_of_scan:None])
            return buffer.getvalue()


class SvsTiler(Tiler):
    def __init__(self, tiff_file: TiffFile):
        """Tiler for svs file.

        Parameters
        ----------
        tiff_file: TiffFile
            A Svs-TiffFile.
        """
        super().__init__(tiff_file)
        self._fh = self._tiff_file.filehandle

        for series_index, series in enumerate(self.series):
            if series.name == 'Baseline':
                self._level_series_index = series_index
            elif series.name == 'Label':
                self._label_series_index = series_index
            elif series.name == 'Macro':
                self._overview_series_index = series_index

    @cached_property
    def base_mpp(self) -> SizeMm:
        """Return pixel spacing in um/pixel for base level."""
        mpp = svs_description_metadata(self.base_page.description)['MPP']
        return SizeMm(mpp, mpp)

    def get_page(
        self,
        series: int,
        level: int,
        page: int = 0
    ) -> SvsTiledPage:
        """Return SvsTiledPage for series, level, page."""
        tiff_page = self.series[series].levels[level].pages[page]
        return SvsTiledPage(
            tiff_page,
            self._fh,
            self.base_size,
            self.base_mpp
        )
