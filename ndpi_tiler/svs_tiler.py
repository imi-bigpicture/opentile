import io
import math
from pathlib import Path
from typing import Dict, Tuple, List, Iterator

from tifffile.tifffile import (FileHandle, TiffFile, TiffPage, TiffPageSeries,
                               svs_description_metadata)
from wsidicom.geometry import Point, Size, SizeMm
from wsidicom.interface import TiledLevel
from .interface import TifffileTiler


class TiffTiledLevel(TiledLevel):
    def __init__(
        self,
        filehandle: FileHandle,
        level: TiffPageSeries,
        base_shape: Tuple[int, int],
        base_mpp: Tuple[int, int]
    ):
        self._level = level
        self._fh = filehandle
        pyramid_index = int(math.log2(base_shape[0]/level.shape[0]))
        self._mpp = SizeMm(base_mpp, base_mpp) * pow(2, pyramid_index) * 1000
        self._tile_size = Size(
            int(self.page.tilewidth),
            int(self.page.tilelength)
        )
        self._level_size = Size(
            self.page.shape[1],
            self.page.shape[0]
        )
        self._tiled_size = Size(
            math.ceil(self.level_size.width / self.tile_size.width),
            math.ceil(self.level_size.height / self.tile_size.height)
        )

    @property
    def page(self) -> TiffPage:
        return self.level.pages[0]

    @property
    def level(self) -> TiffPageSeries:
        return self._level

    @property
    def tile_size(self) -> Size:
        return self._tile_size

    @property
    def tiled_size(self) -> Size:
        return self._tiled_size

    @property
    def level_size(self) -> Size:
        return self._level_size

    @property
    def mpp(self) -> SizeMm:
        return self._mpp

    def get_encoded_tile(self, tile: Point) -> bytes:
        # index for reading tile
        tile_index = tile.y * self.tiled_size.width + tile.x
        self._fh.seek(self.page.dataoffsets[tile_index])
        data = self._fh.read(self.page.databytecounts[tile_index])

        with io.BytesIO() as buffer:
            buffer.write(self.page.jpegtables[:-2])
            buffer.write(
                b"\xFF\xEE\x00\x0E\x41\x64\x6F\x62"
                b"\x65\x00\x64\x80\x00\x00\x00\x00"
            )  # colorspace fix
            buffer.write(data[2:])
            return buffer.getvalue()


class SvsTiler(TifffileTiler):
    def _get_level_from_series(
        self,
        series: int,
        level: int
    ) -> TiffTiledLevel:
        tiff_level = self.series[series].levels[level]
        base = self.series[series].levels[0]
        base_mpp: Tuple[int, int] = svs_description_metadata(
            base.pages[0].description
        )['MPP']

        return TiffTiledLevel(
            self._tiff_file.filehandle, tiff_level, base.shape, base_mpp
        )
