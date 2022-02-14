#    Copyright 2021 SECTRA AB
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import io
import math
import threading
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Any

import numpy as np
from tifffile.tifffile import (FileHandle, TiffFile, TiffPage, TiffPageSeries,
                               TiffTags)

from opentile.geometry import Point, Region, Size, SizeMm
from opentile.jpeg import Jpeg


class LockableFileHandle:
    """A lockable file handle for reading frames."""
    def __init__(self, fh: FileHandle):
        self._fh = fh
        self._lock = threading.Lock()

    def __str__(self) -> str:
        return f"{type(self).__name__} for FileHandle {self._fh}"

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._fh})"

    @property
    def filepath(self) -> Path:
        return Path(self._fh.path)

    def read(self, offset: int, bytecount: int) -> bytes:
        """Return bytes from file handle.

        Parameters
        ----------
        offset: int
            Offset in bytes.
        bytecount: int
            Length in bytes.

        Returns
        ----------
        bytes
            Requested bytes.
        """
        with self._lock:
            self._fh.seek(offset)
            data: bytes = self._fh.read(bytecount)
        return data

    def close(self) -> None:
        """Close the file handle"""
        self._fh.close()


class OpenTilePage(metaclass=ABCMeta):
    """Abstract class for reading tiles from TiffPage. Should be inherited to
    support different tiff formats:

    OpenTilePage
        NativeTiledPage - Meta class for pages that are natively tiled
            PhilipsTiffTiledPage - OpenTiledPage for Philips Tiff-page
            SvsTiledPage - OpenTiledPage for Svs Tiff-page
        NdpiPage - Ndpi page that should not be tiled
            NdpiTiledPage - Metaclass for a tiled Ndpi page
                NdpiOneFramePage - Tiled Ndpi page of only one frame
                NdpiStripedPage - Tiled Ndpi page of striped frames

    """
    _pyramid_index: int

    def __init__(
        self,
        page: TiffPage,
        fh: FileHandle
    ):
        """Abstract class for reading tiles from TiffPage.

        Parameters
        ----------
        page: TiffPage
            TiffPage to get tiles from.
        fh: FileHandle
            FileHandle for reading data.
        """
        self._page = page
        self._fh = LockableFileHandle(fh)
        self._image_size = Size(self._page.shape[1], self._page.shape[0])
        if self.page.is_tiled:
            self._tile_size = Size(
                self.page.tilewidth,
                self.page.tilelength
            )
        else:
            self._tile_size = self.image_size
        self._tiled_region = Region(
            position=Point(0, 0),
            size=self.tiled_size - 1
        )

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError

    def __str__(self) -> str:
        return f"{type(self).__name__} of page {self._page}"

    @property
    def filepath(self) -> Path:
        return self._fh.filepath

    @property
    def suggested_minimum_chunk_size(self) -> int:
        """Suggested minimum chunk size regarding performance for reading
        multiple tiles (get_tiles())."""
        return 1

    @property
    def compression(self) -> str:
        """Return compression of page."""
        return str(self._page.compression)

    @property
    def page(self) -> TiffPage:
        """Return source TiffPage."""
        return self._page

    @property
    def focal_plane(self) -> float:
        """Return focal plane (in um)."""
        return 0.0

    @property
    def optical_path(self) -> str:
        """Return optical path identifier."""
        # Not sure if optical paths are defined in tiff files...
        return '0'

    @property
    def image_size(self) -> Size:
        """The pixel size of the image."""
        return self._image_size

    @property
    def tile_size(self) -> Size:
        """The pixel size of the tiles. Returns image size if not tiled page
        """
        return self._tile_size

    @property
    def tiled_size(self) -> Size:
        """The size of the image when tiled."""
        if self.tile_size == Size(0, 0):
            return Size(1, 1)
        return self.image_size.ceil_div(self.tile_size)

    @property
    def pyramid_index(self) -> int:
        """The pyramidal index in relation to the base layer. Returns 0 for
        images not in pyramidal series."""
        return self._pyramid_index

    @property
    @abstractmethod
    def pixel_spacing(self) -> Optional[SizeMm]:
        """Should return the pixel size in mm/pixel of the page."""
        raise NotImplementedError

    @abstractmethod
    def get_tile(self, tile_position: Tuple[int, int]) -> bytes:
        """Should return image bytes for tile at tile position.

        Parameters
        ----------
        tile_position: Tuple[int, int]
            Tile position to get.

        Returns
        ----------
        bytes
            Produced tile at position.
        """
        raise NotImplementedError

    @abstractmethod
    def get_decoded_tile(self, tile_position: Tuple[int, int]) -> np.ndarray:
        """Should return decoded tile for tile position.

        Parameters
        ----------
        tile_position: Tuple[int, int]
            Tile position to get.

        Returns
        ----------
        bytes
            Produced tile at position.
        """
        raise NotImplementedError

    def get_tiles(
        self,
        tile_positions: Sequence[Tuple[int, int]]
    ) -> List[bytes]:
        """Return list of image bytes for tiles at tile positions.

        Parameters
        ----------
        tile_positions: Sequence[Tuple[int, int]]
            Tile positions to get.

        Returns
        ----------
        List[bytes]
            List of tile bytes.
        """
        return [
            self.get_tile(tile) for tile in tile_positions
        ]

    def get_decoded_tiles(
        self, tile_positions: Sequence[Tuple[int, int]]
    ) -> List[np.ndarray]:
        """Return list of decoded tiles for tiles at tile positions.

        Parameters
        ----------
        tile_positions: Sequence[Tuple[int, int]]
            Tile positions to get.

        Returns
        ----------
        List[np.ndarray]
            List of decoded tiles.
        """
        return [self.get_decoded_tile(tile) for tile in tile_positions]

    def close(self) -> None:
        """Close filehandle."""
        self._fh.close()

    def pretty_str(
        self,
        indent: int = 0,
        depth: Optional[int] = None
    ) -> str:
        return str(self)

    @property
    def tiled_region(self) -> Region:
        """Tile region covering the OpenTilePage."""
        return self._tiled_region

    def valid_tiles(self, region: Region) -> bool:
        """Check if tile region is inside tile geometry and z coordinate and
        optical path exists.

        Parameters
        ----------
        region: Region
            Tile region.
        """
        return region.is_inside(self.tiled_region)

    def _read_frame(self, index: int) -> bytes:
        """Read frame bytes at index from file.

        Parameters
        ----------
        index: int
            Index of frame to read.

        Returns
        ----------
        bytes
            Frame bytes.
        """
        return self._fh.read(
            self._page.dataoffsets[index],
            self._page.databytecounts[index]
        )

    def _check_if_tile_inside_image(self, tile_position: Point) -> bool:
        """Return true if tile position is inside tiled image."""
        return (
            tile_position.x < self.tiled_size.width and
            tile_position.y < self.tiled_size.height
        )

    @staticmethod
    def _get_value_from_tiff_tags(
        tiff_tags: TiffTags,
        value_name: str
    ) -> Optional[str]:
        for tag in tiff_tags:
            if tag.name == value_name:
                return str(tag.value)
        return None

    def _calculate_pyramidal_index(
        self,
        base_shape: Size,
    ) -> int:
        return int(math.log2(base_shape.width/self.image_size.width))

    def _calculate_mpp(
        self,
        base_mpp: SizeMm
    ) -> SizeMm:
        return base_mpp * pow(2, self.pyramid_index)


class NativeTiledPage(OpenTilePage, metaclass=ABCMeta):
    """Meta class for pages that are natively tiled (e.g. not ndpi)"""
    def get_tile(
        self,
        tile_position: Tuple[int, int]
    ) -> bytes:
        """Return image bytes for tile at tile position.

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
        frame_index = self._tile_point_to_frame_index(tile_point)
        tile = self._read_frame(frame_index)
        if self.page.jpegtables is not None:
            tile = Jpeg.add_jpeg_tables(tile, self.page.jpegtables)
        return tile

    def get_decoded_tile(self, tile_position: Tuple[int, int]) -> np.ndarray:
        """Return decoded tile for tile position. Returns a white tile if tile
        is outside of image.

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
        if not self._check_if_tile_inside_image(tile_point):
            return np.full(
                self.tile_size.to_tuple() + (3,),
                255,
                dtype=np.dtype(np.uint8)
            )

        data: np.ndarray
        shape: tuple[int, int, int, int]
        frame = self.get_tile(tile_position)
        frame_index = self._tile_point_to_frame_index(tile_point)
        data, _, shape = self.page.decode(frame, frame_index)  # TBD!
        data.shape = shape[1:]
        return data

    def _tile_point_to_frame_index(
        self,
        tile_point: Point
    ) -> int:
        """Return linear frame index for tile position."""
        frame_index = tile_point.y * self.tiled_size.width + tile_point.x
        return frame_index


class Tiler(metaclass=ABCMeta):
    """Abstract class for reading pages from TiffFile."""
    _level_series_index: int = 0
    _overview_series_index: Optional[int] = None
    _label_series_index: Optional[int] = None

    def __init__(self, filepath: Path):
        """Abstract class for reading pages from TiffFile.

        Parameters
        ----------
        filepath: Path
            Filepath to a TiffFile.
        """
        self._tiff_file = TiffFile(filepath)
        self._base_page = self.series[self._level_series_index].pages[0]
        self._base_size = Size(
            self.base_page.shape[1],
            self.base_page.shape[0]
        )
        # self._pages: Dict[Tuple[int, int, int], OpenTilePage] = {}

    @property
    def properties(self) -> Dict[str, Any]:
        """Dictionary of properties read from TiffFile."""
        return {}

    @property
    def base_page(self) -> TiffPage:
        """Return base pyramid level in pyramid series."""
        return self._base_page

    @property
    def base_size(self) -> Size:
        """Return size of base pyramid level in pyramid series."""
        return self._base_size

    @property
    def series(self) -> List[TiffPageSeries]:
        """Return contained TiffPageSeries."""
        return self._tiff_file.series

    @property
    def levels(self) -> List[OpenTilePage]:
        """Return list of pyramid level OpenTilePages."""
        if self._level_series_index is None:
            return []
        return [
            self.get_level(level_index, page_index)
            for level_index, level
            in enumerate(self.series[self._level_series_index].levels)
            for page_index, page in enumerate(level.pages)
        ]

    @property
    def labels(self) -> List[OpenTilePage]:
        """Return list of label OpenTilePage."""
        if self._label_series_index is None:
            return []
        return [
            self.get_label(page_index)
            for page_index, page
            in enumerate(self.series[self._label_series_index].pages)
        ]

    @property
    def overviews(self) -> List[OpenTilePage]:
        """Return list of overview OpenTilePage."""
        if self._overview_series_index is None:
            return []
        return [
            self.get_overview(page_index)
            for page_index, page
            in enumerate(self.series[self._overview_series_index].pages)
        ]

    @abstractmethod
    def get_page(self, series: int, level: int, page: int) -> OpenTilePage:
        """Should return a OpenTilePage for series, level, page in file."""
        raise NotImplementedError

    def close(self) -> None:
        """CLose tiff-file."""
        self._tiff_file.close()

    def get_tile(
        self,
        series: int,
        level: int,
        page: int,
        tile_position: Tuple[int, int]
    ) -> bytes:
        """Return list of image bytes for tiles at tile positions.

        Parameters
        ----------
        series: int
            Series of page to get tile from.
        level: int
            Level of page to get tile from.
        page: int
            Page to get tile from.
        tile_position: Tuple[int, int]
            Position of tile to get.

        Returns
        ----------
        bytes
            Tile at position.
        """
        tiled_page = self.get_page(series, level, page)
        return tiled_page.get_tile(tile_position)

    def get_level(
        self,
        level: int,
        page: int = 0
    ) -> OpenTilePage:
        """Return OpenTilePage for level in pyramid series.

        Parameters
        ----------
        level: int
            Level to get.
        page: int
            Index of page to get.

        Returns
        ----------
        OpenTilePage
            Level OpenTilePage.
        """
        return self.get_page(self._level_series_index, level, page)

    def get_label(
        self,
        page: int = 0
    ) -> OpenTilePage:
        """Return OpenTilePage for label in label series.

        Parameters
        ----------
        page: int
            Index of page to get.

        Returns
        ----------
        OpenTilePage
            Label OpenTilePage.
        """
        if self._label_series_index is None:
            raise ValueError("No label detected in file")
        return self.get_page(self._label_series_index, 0, page)

    def get_overview(
        self,
        page: int = 0
    ) -> OpenTilePage:
        """Return OpenTilePage for overview in overview series.

        Parameters
        ----------
        page: int
            Index of page to get.

        Returns
        ----------
        OpenTilePage
            Overview OpenTilePage.
        """
        if self._overview_series_index is None:
            raise ValueError("No overview detected in file")
        return self.get_page(self._overview_series_index, 0, page)
