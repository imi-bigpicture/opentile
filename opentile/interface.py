from abc import abstractmethod
from wsidicom.interface import TiledLevel, Tiler
from pathlib import Path
from tifffile.tifffile import TiffFile, TiffPageSeries
from typing import List, Tuple
from wsidicom.geometry import Point

DEFAULT_VOLUME_SERIES_INDEX = 0
DEFAULT_LABEL_SERIES_INDEX = 2
DEFAULT_OVERVIEW_SERIES_INDEX = 3


class TifffileTiler(Tiler):
    def __init__(
        self,
        filepath: Path,
        volume_series_index: int = DEFAULT_VOLUME_SERIES_INDEX,
        label_series_index: int = DEFAULT_LABEL_SERIES_INDEX,
        overview_series_index: int = DEFAULT_OVERVIEW_SERIES_INDEX
    ):
        self._filepath = filepath
        self._tiff_file = TiffFile(self._filepath)
        self._volume_series_index = volume_series_index
        self._overview_series_index = overview_series_index
        self._label_series_index = label_series_index

    @property
    def series(self) -> List[TiffPageSeries]:
        return self._tiff_file.series

    @property
    def level_count(self) -> int:
        return len(self.series[self._volume_series_index].levels)

    @property
    def label_count(self) -> int:
        return len(self.series[self._label_series_index].levels)

    @property
    def overview_count(self) -> int:
        return len(self.series[self._overview_series_index].levels)

    @abstractmethod
    def _get_level_from_series(self, series: int, level: int) -> TiledLevel:
        raise NotImplementedError

    def close(self) -> None:
        self._tiff_file.close()

    def get_tile(
        self,
        level: int,
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
        tiled_level = self.get_level(level)
        return tiled_level.get_tile(Point(*tile_position))

    def get_level(self, level: int) -> TiledLevel:
        """Return level from volume series.

        Parameters
        ----------
        level: int
            Level to get.

        Returns
        ----------
        NdpiLevel
            Requested level.
        """
        return self._get_level_from_series(self._volume_series_index, level)

    def get_label(self, index: int = 0) -> TiledLevel:
        """Return label from label series.

        Parameters
        ----------
        inex: int
            Index of label to get.

        Returns
        ----------
        NdpiLevel
            Requested level.
        """
        return self._get_level_from_series(self._label_series_index, index)

    def get_overview(self, index: int = 0) -> TiledLevel:
        """Return overview from overview series.

        Parameters
        ----------
        inex: int
            Index of overview to get.

        Returns
        ----------
        NdpiLevel
            Requested level.
        """
        return self._get_level_from_series(self._overview_series_index, index)
