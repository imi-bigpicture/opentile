import io
import math
from pathlib import Path
from typing import List, Tuple

from tifffile.tifffile import (FileHandle, TiffPage, TiffPageSeries,
                               svs_description_metadata)
from wsidicom.geometry import Point, Size, SizeMm
from wsidicom.image_data import ImageData

from .interface import TifffileTiler


class TiffTiledLevel(ImageData):
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
        self._image_size = Size(
            self.page.shape[1],
            self.page.shape[0]
        )
        if self._tile_size != Size(0, 0):
            self._tiled_size = Size(
                math.ceil(self.image_size.width / self.tile_size.width),
                math.ceil(self.image_size.height / self.tile_size.height)
            )
        else:
            self._tiled_size = Size(1, 1)

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
    def image_size(self) -> Size:
        return self._image_size

    @property
    def mpp(self) -> SizeMm:
        return self._mpp

    @property
    def focal_planes(self) -> List[float]:
        # No support for multiple focal planes
        return [0]

    @property
    def optical_paths(self) -> List[str]:
        # No support for multiple optical paths
        return ['0']

    def close(self) -> None:
        self._fh.close()

    def get_encoded_tile(self, tile_position: Point) -> bytes:
        return self.get_tile(tile_position)

    def get_tile(
        self,
        tile_position: Point,
        z: float = 0,
        path: str = '0'
    ) -> bytes:
        # index for reading tile
        tile_index = tile_position.y * self.tiled_size.width + tile_position.x
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
    def __init__(self, filepath: Path):
        super().__init__(filepath)

        for series_index, series in enumerate(self.series):
            if series.name == 'Label':
                self._label_series_index = series_index
            elif series.name == 'Macro':
                self._overview_series_index = series_index

    def _get_level_from_series(
        self,
        series: int,
        level: int
    ) -> TiffTiledLevel:
        tiff_level = self.series[series].levels[level]
        base = self.series[series].levels[0]

        if series == self._volume_series_index:
            base_mpp: Tuple[int, int] = svs_description_metadata(
                base.pages[0].description
            )['MPP']
        else:
            base_mpp = 1.0
        return TiffTiledLevel(
            self._tiff_file.filehandle, tiff_level, base.shape, base_mpp
        )
