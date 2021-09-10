from abc import abstractmethod
from pathlib import Path
from typing import List, Tuple

from tifffile.tifffile import TiffFile, TiffPageSeries
from wsidicom.geometry import Point
from wsidicom.image_data import ImageData, Tiler

DEFAULT_VOLUME_SERIES_INDEX = 0
DEFAULT_LABEL_SERIES_INDEX = None
DEFAULT_OVERVIEW_SERIES_INDEX = None


class TifffileTiler(Tiler):
    def __init__(self, filepath: Path):
        self._filepath = filepath
        self._tiff_file = TiffFile(self._filepath)
        self._volume_series_index = DEFAULT_VOLUME_SERIES_INDEX
        self._overview_series_index = DEFAULT_LABEL_SERIES_INDEX
        self._label_series_index = DEFAULT_OVERVIEW_SERIES_INDEX

    @property
    def series(self) -> List[TiffPageSeries]:
        return self._tiff_file.series

    @property
    def level_count(self) -> int:
        if self._volume_series_index is None:
            return 0
        return len(self.series[self._volume_series_index].levels)

    @property
    def label_count(self) -> int:
        if self._label_series_index is None:
            return 0
        return len(self.series[self._label_series_index].levels)

    @property
    def overview_count(self) -> int:
        if self._overview_series_index is None:
            return 0
        return len(self.series[self._overview_series_index].levels)

    @abstractmethod
    def _get_level_from_series(self, series: int, level: int) -> ImageData:
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

    def get_level(self, level: int) -> ImageData:
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

    def get_label(self, index: int = 0) -> ImageData:
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

    def get_overview(self, index: int = 0) -> ImageData:
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
