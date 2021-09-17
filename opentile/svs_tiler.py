import io
import math
from functools import cached_property
from pathlib import Path
from typing import Tuple

from tifffile.tifffile import (TIFF, FileHandle, TiffPage,
                               svs_description_metadata)

from opentile.geometry import Point, Size, SizeMm
from opentile.interface import TiledPage, Tiler


class SvsTiledPage(TiledPage):
    def __init__(
        self,
        page: TiffPage,
        fh: FileHandle,
        base_shape: Size,
        base_mpp: SizeMm
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
        """
        super().__init__(page, fh)
        self._pyramid_index = int(
            math.log2(base_shape.width/self.image_size.width)
        )
        self._mpp = base_mpp * pow(2, self.pyramid_index)

    @property
    def compression(self) -> str:
        return str(self._page.compression)

    @property
    def pyramid_index(self) -> int:
        return self._pyramid_index

    @cached_property
    def tile_size(self) -> Size:
        return Size(
            int(self.page.tilewidth),
            int(self.page.tilelength)
        )

    @cached_property
    def tiled_size(self) -> Size:
        if self.tile_size != Size(0, 0):
            return Size(
                math.ceil(self.image_size.width / self.tile_size.width),
                math.ceil(self.image_size.height / self.tile_size.height)
            )
        else:
            return Size(1, 1)

    @cached_property
    def image_size(self) -> Size:
        """The size of the image."""
        return Size(
            self.page.shape[1],
            self.page.shape[0]
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
        with io.BytesIO() as buffer:
            buffer.write(self.page.jpegtables[:-2])
            buffer.write(
                b"\xFF\xEE\x00\x0E\x41\x64\x6F\x62"
                b"\x65\x00\x64\x80\x00\x00\x00\x00"
            )  # colorspace fix
            buffer.write(frame[2:])
            return buffer.getvalue()

    def get_tile(
        self,
        tile_position: Tuple[int, int]
    ) -> bytes:
        """Return tile for tile position.

        Parameters
        ----------
        tile_position: Tuple[int, int]
            Tile position to get.

        Returns
        ----------
        bytes
            Produced tile at position.
        """
        tile_point = Point.from_tuple(tile_position)
        tile_index = tile_point.y * self.tiled_size.width + tile_point.x
        self._fh.seek(self.page.dataoffsets[tile_index])
        data = self._fh.read(self.page.databytecounts[tile_index])
        if self.compression == 'COMPRESSION.JPEG':
            return self._add_jpeg_tables(data)
        return data


class SvsTiler(Tiler):
    def __init__(self, filepath: Path):
        """Tiler for svs file.

        Parameters
        ----------
        filepath: str
            File path to svs file.
        """
        super().__init__(filepath)
        self._fh = self._tiff_file.filehandle

        for series_index, series in enumerate(self.series):
            if series.name == 'Baseline':
                self._volume_series_index = series_index
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
        """Return SvsTiledPage for series, level, page.
        """
        tiff_page = self.series[series].levels[level].pages[page]
        return SvsTiledPage(
            tiff_page,
            self._fh,
            self.base_size,
            self.base_mpp
        )
