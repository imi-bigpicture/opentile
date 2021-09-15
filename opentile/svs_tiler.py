import io
import math
from functools import cached_property
from pathlib import Path

from tifffile.tifffile import FileHandle, TiffPage, svs_description_metadata

from opentile.geometry import Point, Region, Size, SizeMm
from opentile.interface import TiledPage, Tiler


class SvsTiledPage(TiledPage):
    def __init__(
        self,
        page: TiffPage,
        fh: FileHandle,
        base_shape: Size,
        base_mpp: SizeMm
    ):
        super().__init__(page, fh)
        self._pyramid_index = int(
            math.log2(base_shape.width/self.image_size.width)
        )
        self._mpp = base_mpp * pow(2, self.pyramid_index)

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
        return Size(
            self.page.shape[1],
            self.page.shape[0]
        )

    @property
    def pixel_spacing(self) -> SizeMm:
        return self.mpp * 1000

    @property
    def mpp(self) -> SizeMm:
        return self._mpp

    def close(self) -> None:
        self._fh.close()

    def get_encoded_tile(self, tile_position: Point) -> bytes:
        return self.get_tile(tile_position)

    def get_tile(
        self,
        tile_position: Point
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


class SvsTiler(Tiler):
    def __init__(self, filepath: Path):
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
    def base_page(self) -> TiffPage:
        return self.series[self._volume_series_index].pages[0]

    @cached_property
    def base_mpp(self) -> SizeMm:
        mpp = svs_description_metadata(self.base_page.description)['MPP']
        return SizeMm(mpp, mpp)

    def get_page(
        self,
        series: int,
        level: int,
        page: int = 0
    ) -> SvsTiledPage:
        tiff_page = self.series[series].levels[level].pages[page]
        return SvsTiledPage(
            tiff_page,
            self._fh,
            self.base_size,
            self.base_mpp
        )
