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

import math
import threading
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
from tifffile.tifffile import (
    COMPRESSION,
    PHOTOMETRIC,
    FileHandle,
    TiffPage,
    TiffTags,
)

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
        """Return bytes from single location from file handle. Is thread safe.

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
            return self._read(offset, bytecount)

    def read_multiple(
        self, offsets_bytecounts: Sequence[Tuple[int, int]]
    ) -> List[bytes]:
        """Return bytes from multiple locations from file handle. Is thread
        safe.

        Parameters
        ----------
        offsets_bytecounts: Sequence[Tuple[int, int]]
            List of tuples with offset and lengths to read.

        Returns
        ----------
        List[bytes]
            List of requested bytes.
        """
        with self._lock:
            return [
                self._read(offset, bytecount)
                for (offset, bytecount) in offsets_bytecounts
            ]

    def _read(self, offset: int, bytecount: int):
        """Read bytes from file handle. Is not thread safe.

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
        self._fh.seek(offset)
        return self._fh.read(bytecount)

    def close(self) -> None:
        """Close the file handle"""
        self._fh.close()


class TiffImage(metaclass=ABCMeta):
    """Abstract class for reading tiles from TiffPage. Should be inherited to
    support different tiff formats.
    """

    _pyramid_index: int

    def __init__(
        self, page: TiffPage, fh: FileHandle, add_rgb_colorspace_fix: bool = False
    ):
        """Abstract class for reading tiles from TiffPage.

        Parameters
        ----------
        page: TiffPage
            TiffPage to get tiles from.
        fh: FileHandle
            FileHandle for reading data.
        add_rgb_colorspace_fix: bool = False
            If to add color space fix for rgb image data.
        """
        if (
            self.supported_compressions is not None
            and page.compression not in self.supported_compressions
        ):
            raise NotImplementedError(f"Non-supported compression {self.compression}.")
        self._page = page
        self._fh = LockableFileHandle(fh)
        self._add_rgb_colorspace_fix = add_rgb_colorspace_fix
        self._image_size = Size(self._page.imagewidth, self._page.imagelength)
        if self.page.is_tiled:
            self._tile_size = Size(self.page.tilewidth, self.page.tilelength)
        else:
            self._tile_size = self.image_size
        self._tiled_region = Region(position=Point(0, 0), size=self.tiled_size)

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError()

    def __str__(self) -> str:
        return f"{type(self).__name__} of page {self._page}"

    @property
    @abstractmethod
    def pixel_spacing(self) -> Optional[SizeMm]:
        """Should return the pixel size in mm/pixel of the image."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def supported_compressions(self) -> Optional[List[COMPRESSION]]:
        """Should return list of compressions supported, or None if class is
        indenpendent on compression."""
        raise NotImplementedError()

    @property
    def filepath(self) -> Path:
        return self._fh.filepath

    @property
    def suggested_minimum_chunk_size(self) -> int:
        """Suggested minimum chunk size regarding performance for reading
        multiple tiles (get_tiles())."""
        return 1

    @property
    def compression(self) -> COMPRESSION:
        """Return compression of image."""
        return COMPRESSION(self._page.compression)

    @property
    def photometric_interpretation(self) -> PHOTOMETRIC:
        """Return photometric interpretation, e.g. 'YCBCR" or "RGB"."""
        return PHOTOMETRIC(self._page.photometric)

    @property
    def subsampling(self) -> Optional[Tuple[int, int]]:
        """Return subsampling, or None if only one component."""
        return self._page.subsampling

    @property
    def samples_per_pixel(self) -> int:
        """Return samples per pixel."""
        return self._page.samplesperpixel

    @property
    def bit_depth(self) -> int:
        """Return the sample bit depth."""
        return self._page.bitspersample

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
        return "0"

    @property
    def image_size(self) -> Size:
        """The pixel size of the image."""
        return self._image_size

    @property
    def tile_size(self) -> Size:
        """The pixel size of the tiles. Returns image size if not tiled image"""
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
        raise NotImplementedError()

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
        raise NotImplementedError()

    def get_tiles(self, tile_positions: Sequence[Tuple[int, int]]) -> List[bytes]:
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
        return [self.get_tile(tile) for tile in tile_positions]

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

    def get_all_tiles(self, raw: bool = False) -> Iterator[bytes]:
        """Return iterator of all tiles in image.

        Parameters
        ----------
        raw: bool = False
            Set to True to not do any format-specifc processing on the tile.

        Returns
        ----------
        Iterator[bytes]
            Iterator of all tiles in image.
        """
        if raw:
            return (self._read_frame(index) for index in range(self.tiled_size.area))
        return (
            self.get_tile(tile.to_tuple()) for tile in self.tiled_region.iterate_all()
        )

    def get_all_tiles_decoded(self) -> Iterator[np.ndarray]:
        """Return iterator of all tiles in image decoded.

        Returns
        ----------
        Iterator[np.ndarray]
            Iterator of all tiles in image decoded.
        """
        return (
            self.get_decoded_tile(tile.to_tuple())
            for tile in self.tiled_region.iterate_all()
        )

    def close(self) -> None:
        """Close filehandle."""
        self._fh.close()

    def pretty_str(self, indent: int = 0, depth: Optional[int] = None) -> str:
        return str(self)

    @property
    def tiled_region(self) -> Region:
        """Tile region covering the TiffImage."""
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
            self._page.dataoffsets[index], self._page.databytecounts[index]
        )

    def _read_frames(self, indices: Sequence[int]) -> List[bytes]:
        return self._fh.read_multiple(
            [
                (self._page.dataoffsets[index], self._page.databytecounts[index])
                for index in indices
            ]
        )

    def _check_if_tile_inside_image(self, tile_position: Point) -> bool:
        """Return true if tile position is inside tiled image."""
        return (
            tile_position.x < self.tiled_size.width
            and tile_position.y < self.tiled_size.height
        )

    @staticmethod
    def _get_value_from_tiff_tags(
        tiff_tags: TiffTags, value_name: str
    ) -> Optional[str]:
        return next(
            (str(tag.value) for tag in tiff_tags if tag.name == value_name), None
        )

    def _calculate_pyramidal_index(
        self,
        base_size: Size,
    ) -> int:
        return int(math.log2(base_size.width / self.image_size.width))

    def _calculate_mpp(self, base_mpp: SizeMm) -> SizeMm:
        return base_mpp * pow(2, self.pyramid_index)


class NativeTiledTiffImage(TiffImage, metaclass=ABCMeta):

    """Meta class for images that are natively tiled (e.g. not ndpi)"""

    def get_tile(self, tile_position: Tuple[int, int]) -> bytes:
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
            tile = Jpeg.add_jpeg_tables(
                tile, self.page.jpegtables, self._add_rgb_colorspace_fix
            )
        return tile

    def get_tiles(self, tile_positions: Sequence[Tuple[int, int]]) -> List[bytes]:
        """Return image bytes for tiles at tile positions.

        Parameters
        ----------
        tile_positions: Sequence[Tuple[int, int]]
            Tile positions to get.

        Returns
        ----------
        List[bytes]
            Produced tiles at positions.
        """
        tile_points = [
            Point.from_tuple(tile_position) for tile_position in tile_positions
        ]
        frame_indices = [
            self._tile_point_to_frame_index(tile_point) for tile_point in tile_points
        ]
        tiles = self._read_frames(frame_indices)
        if self.page.jpegtables is not None:
            tiles = [
                Jpeg.add_jpeg_tables(
                    tile, self.page.jpegtables, self._add_rgb_colorspace_fix
                )
                for tile in tiles
            ]
        return tiles

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
                self.tile_size.to_tuple() + (3,), 255, dtype=np.dtype(np.uint8)
            )

        shape: tuple[int, int, int, int]
        frame = self.get_tile(tile_position)
        frame_index = self._tile_point_to_frame_index(tile_point)
        data, _, shape = self.page.decode(frame, frame_index)
        assert isinstance(data, np.ndarray)
        data.shape = shape[1:]
        return data

    def _tile_point_to_frame_index(self, tile_point: Point) -> int:
        """Return linear frame index for tile position."""
        frame_index = tile_point.y * self.tiled_size.width + tile_point.x
        return frame_index
