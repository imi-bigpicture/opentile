from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import List, Tuple

from tifffile.tifffile import FileHandle, TiffFile, TiffPage, TiffPageSeries
from wsidicom.geometry import Point, Size, SizeMm, Region
from functools import cached_property


class TiledPage(metaclass=ABCMeta):
    def __init__(
        self,
        page: TiffPage,
        fh: FileHandle
    ):
        self._page = page
        self._fh = fh

    @property
    def page(self) -> TiffPage:
        return self._page

    @property
    def default_z(self) -> float:
        return 0.0

    @property
    def default_path(self) -> str:
        return '0'

    @property
    @abstractmethod
    def pyramid_index(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def image_size(self) -> Size:
        raise NotImplementedError

    @property
    @abstractmethod
    def tile_size(self) -> Size:
        raise NotImplementedError

    @property
    @abstractmethod
    def tiled_size(self) -> Size:
        raise NotImplementedError

    @property
    @abstractmethod
    def pixel_spacing(self) -> SizeMm:
        raise NotImplementedError

    @abstractmethod
    def get_tile(self, tile: Point) -> bytes:
        raise NotImplementedError

    def close(self) -> None:
        self._fh.close()

    def pretty_str(
        self,
        indent: int = 0,
        depth: int = None
    ) -> str:
        return str(self)

    @cached_property
    def plane_region(self) -> Region:
        return Region(position=Point(0, 0), size=self.tiled_size - 1)

    def valid_tiles(self, region: Region) -> bool:
        """Check if tile region is inside tile geometry and z coordinate and
        optical path exists.

        Parameters
        ----------
        region: Region
            Tile region.
        """
        return region.is_inside(self.plane_region)


class Tiler:
    def __init__(self, filepath: Path):
        self._filepath = filepath
        self._tiff_file = TiffFile(self._filepath)
        self._volume_series_index: int = None
        self._overview_series_index: int = None
        self._label_series_index: int = None

    @cached_property
    def base_page(self) -> TiffPage:
        return self.series[self._volume_series_index].pages[0]

    @cached_property
    def base_size(self) -> Size:
        return Size(self.base_page.shape[1], self.base_page.shape[0])

    @property
    def series(self) -> List[TiffPageSeries]:
        return self._tiff_file.series

    @property
    def levels(self) -> List[TiledPage]:
        if self._volume_series_index is None:
            return []
        return [
            self.get_level(level_index, page_index)
            for level_index, level
            in enumerate(self.series[self._volume_series_index].levels)
            for page_index, page in enumerate(level.pages)
        ]

    @property
    def labels(self) -> List[TiledPage]:
        if self._label_series_index is None:
            return []
        return [
            self.get_label(level_index, page_index)
            for level_index, level
            in enumerate(self.series[self._label_series_index].levels)
            for page_index, page in enumerate(level.pages)
        ]

    @property
    def overviews(self) -> List[TiledPage]:
        if self._overview_series_index is None:
            return []
        return [
            self.get_overview(level_index, page_index)
            for level_index, level
            in enumerate(self.series[self._overview_series_index].levels)
            for page_index, page in enumerate(level.pages)
        ]

    @abstractmethod
    def get_page(self, series: int, level: int, page: int) -> TiledPage:
        raise NotImplementedError

    def close(self) -> None:
        self._tiff_file.close()

    def get_tile(
        self,
        series: int,
        level: int,
        page: int,
        tile_position: Tuple[int, int]
    ) -> bytes:
        """Return tile for tile position x and y.

        Parameters
        ----------
        level: int
            Level of tile to get.
        tile_position: Tuple[int, int]
            Position of tile to get.

        Returns
        ----------
        bytes
            Produced tile at position.
        """
        tiled_page = self.get_page(series, level, page)
        return tiled_page.get_tile(Point(*tile_position))

    def get_level(
        self,
        level: int,
        page: int = 0
    ) -> TiledPage:
        return self.get_page(self._volume_series_index, level, page)

    def get_label(
        self,
        index: int = 0,
        page: int = 0
    ) -> TiledPage:
        return self.get_page(self._label_series_index, index, page)

    def get_overview(
        self,
        index: int = 0,
        page: int = 0
    ) -> TiledPage:
        return self.get_page(self._overview_series_index, index, page)
