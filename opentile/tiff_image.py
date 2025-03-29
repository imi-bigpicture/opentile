#    Copyright 2021-2023 SECTRA AB
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

"""Base image classes."""

from functools import cached_property
import math
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
from tifffile import (
    COMPRESSION,
    PHOTOMETRIC,
    TiffPage,
    TiffTags,
)

from opentile.file import OpenTileFile
from opentile.geometry import Point, Region, Size, SizeMm
from opentile.jpeg import Jpeg


class TiffImage(metaclass=ABCMeta):
    """Abstract class for reading tiles from TiffPage."""

    @property
    @abstractmethod
    def supported_compressions(self) -> Optional[List[COMPRESSION]]:
        """List of compressions supported, or None if image is indenpendent on
        compression."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def filepath(self) -> Path:
        """Filepath of image."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def suggested_minimum_chunk_size(self) -> int:
        """Suggested minimum chunk size regarding performance for reading multiple tiles
        (get_tiles())."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def compression(self) -> COMPRESSION:
        """Compression of image."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def photometric_interpretation(self) -> PHOTOMETRIC:
        """Photometric interpretation of image, e.g. 'YCBCR" or "RGB"."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def subsampling(self) -> Optional[Tuple[int, int]]:
        """Subsampling of image, or None if only one component."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def samples_per_pixel(self) -> int:
        """Samples per pixel in image."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def bit_depth(self) -> int:
        """Sample bit depth of image, e.g. 8."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def focal_plane(self) -> float:
        """Focal plane (in um) of image."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def optical_path(self) -> str:
        """Optical path identifier of image.."""
        # Not sure if optical paths are defined in tiff files...
        raise NotImplementedError()

    @property
    @abstractmethod
    def image_size(self) -> Size:
        """Pixel size of the image."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def tile_size(self) -> Size:
        """Pixel size of the tiles. Returns image size if not tiled image"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def tiled_size(self) -> Size:
        """Size of the image when tiled."""
        raise NotImplementedError()

    @cached_property
    @abstractmethod
    def compressed_size(self) -> int:
        """Size of the compressed image data."""
        raise NotImplementedError()

    @abstractmethod
    def get_tile(self, tile_position: Tuple[int, int]) -> bytes:
        """Read image bytes for tile at tile position.

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
        """Read decoded tile for tile position.

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
    def get_tiles(self, tile_positions: Sequence[Tuple[int, int]]) -> Iterator[bytes]:
        """Read image bytes for tiles at tile positions.

        Parameters
        ----------
        tile_positions: Sequence[Tuple[int, int]]
            Tile positions to get.

        Returns
        ----------
        Iterator[bytes]
            Iterator of tile bytes.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_decoded_tiles(
        self, tile_positions: Sequence[Tuple[int, int]]
    ) -> Iterator[np.ndarray]:
        """Read decoded tiles for tiles at tile positions.

        Parameters
        ----------
        tile_positions: Sequence[Tuple[int, int]]
            Tile positions to get.

        Returns
        ----------
        Iterator[np.ndarray]
            List of decoded tiles.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_all_tiles(self, raw: bool = False) -> Iterator[bytes]:
        """Iterator of all tiles in image as bytes.

        Parameters
        ----------
        raw: bool = False
            Set to True to not do any format-specifc processing on the tile.

        Returns
        ----------
        Iterator[bytes]
            Iterator of all tiles in image.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_all_tiles_decoded(self) -> Iterator[np.ndarray]:
        """Iterator of all tiles in image decoded.

        Returns
        ----------
        Iterator[np.ndarray]
            Iterator of all tiles in image decoded.
        """
        raise NotImplementedError()

    @abstractmethod
    def close(self) -> None:
        """Close filehandle."""
        raise NotImplementedError()


class AssociatedTiffImage(TiffImage):
    """Abstract class for associated image."""

    @property
    @abstractmethod
    def pixel_spacing(self) -> Optional[SizeMm]:
        """Should return the pixel size in mm/pixel of the image."""
        raise NotImplementedError()


class ThumbnailTiffImage(TiffImage):
    """Abstract class for thumbnail image."""

    @property
    @abstractmethod
    def scale(self) -> float:
        """The scale of the image in relation to the base level."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def pixel_spacing(self) -> SizeMm:
        """The pixel size in mm/pixel of the image."""
        raise NotImplementedError()


class LevelTiffImage(TiffImage):
    """Abstract class for level image."""

    @property
    @abstractmethod
    def pixel_spacing(self) -> SizeMm:
        """The pixel size in mm/pixel of the image."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def scale(self) -> float:
        """The scale of the image in relation to the base level."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def pyramid_index(self) -> int:
        """The pyramidal index in relation to the base layer."""
        raise NotImplementedError()


class BaseTiffImage(TiffImage):
    """Base class for reading tiles from TiffPage. Should be inherited to support
    different tiff formats.
    """

    def __init__(
        self,
        page: TiffPage,
        file: OpenTileFile,
        add_rgb_colorspace_fix: bool = False,
    ):
        """Base class for reading tiles from TiffPage.

        Parameters
        ----------
        page: TiffPage
            TiffPage to get tiles from.
        file: OpenTileFile
            FileHandle for reading data.
        add_rgb_colorspace_fix: bool = False
            If to add color space fix for rgb image data.
        """
        if (
            self.supported_compressions is not None
            and page.compression not in self.supported_compressions
        ):
            raise NotImplementedError(f"Non-supported compression {page.compression}.")
        self._page = page
        self._file = file
        self._add_rgb_colorspace_fix = add_rgb_colorspace_fix
        self._image_size = Size(self._page.imagewidth, self._page.imagelength)
        if self._page.is_tiled:
            self._tile_size = Size(self._page.tilewidth, self._page.tilelength)
        else:
            self._tile_size = self.image_size
        self._tiled_region = Region(position=Point(0, 0), size=self.tiled_size)

    def __str__(self) -> str:
        return f"{type(self).__name__} of page {self._page}"

    @property
    def filepath(self) -> Path:
        return self._file.filepath

    @property
    def suggested_minimum_chunk_size(self) -> int:
        return 1

    @property
    def compression(self) -> COMPRESSION:
        return COMPRESSION(self._page.compression)

    @property
    def photometric_interpretation(self) -> PHOTOMETRIC:
        return PHOTOMETRIC(self._page.photometric)

    @property
    def subsampling(self) -> Optional[Tuple[int, int]]:
        return self._page.subsampling

    @property
    def samples_per_pixel(self) -> int:
        return self._page.samplesperpixel

    @property
    def bit_depth(self) -> int:
        return self._page.bitspersample

    @property
    def focal_plane(self) -> float:
        return 0.0

    @property
    def optical_path(self) -> str:
        # Not sure if optical paths are defined in tiff files...
        return "0"

    @property
    def image_size(self) -> Size:
        return self._image_size

    @property
    def tile_size(self) -> Size:
        return self._tile_size

    @property
    def tiled_size(self) -> Size:
        if self.tile_size == Size(0, 0):
            return Size(1, 1)
        return self.image_size.ceil_div(self.tile_size)

    @cached_property
    def compressed_size(self) -> int:
        frames = sum(self._page.databytecounts)
        if self._page.jpegheader is not None:
            jpeg_header_length = len(self._page.jpegheader)
        elif self._page.jpegtables is not None:
            jpeg_header_length = len(self._page.jpegtables)
        else:
            jpeg_header_length = 0
        return frames + len(self._page.dataoffsets) * jpeg_header_length

    def get_tiles(self, tile_positions: Sequence[Tuple[int, int]]) -> Iterator[bytes]:
        return (self.get_tile(tile) for tile in tile_positions)

    def get_decoded_tiles(
        self, tile_positions: Sequence[Tuple[int, int]]
    ) -> Iterator[np.ndarray]:
        return (self.get_decoded_tile(tile) for tile in tile_positions)

    def get_all_tiles(self, raw: bool = False) -> Iterator[bytes]:
        if raw:
            return (self._read_frame(index) for index in range(self.tiled_size.area))
        return (
            self.get_tile(tile.to_tuple()) for tile in self._tiled_region.iterate_all()
        )

    def get_all_tiles_decoded(self) -> Iterator[np.ndarray]:
        return (
            self.get_decoded_tile(tile.to_tuple())
            for tile in self._tiled_region.iterate_all()
        )

    def close(self) -> None:
        self._file.close()

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
        return self._file.read(
            self._page.dataoffsets[index], self._page.databytecounts[index]
        )

    def _read_frames(self, indices: Sequence[int]) -> List[bytes]:
        return self._file.read_multiple(
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

    def _calculate_scale(self, base_size: Size) -> float:
        """Calculate scale of image in relation to base size."""
        return base_size.width / self.image_size.width

    def _calculate_pyramidal_index(
        self,
        scale: float,
    ) -> int:
        float_index = math.log2(scale)
        index = int(round(float_index))
        TOLERANCE = 1e-2
        if not math.isclose(float_index, index, abs_tol=TOLERANCE):
            raise NotImplementedError(
                f"Pyramid index needs to be integer. Got {float_index} that is more than set"
                f"tolerance {TOLERANCE} from the closest integer {index}. "
            )
        return index

    def _calculate_mpp(self, base_mpp: SizeMm, scale: float) -> SizeMm:
        return base_mpp * scale


class NativeTiledTiffImage(BaseTiffImage, metaclass=ABCMeta):
    """Meta class for images that are natively tiled (e.g. not ndpi)"""

    def get_tile(self, tile_position: Tuple[int, int]) -> bytes:
        tile_point = Point.from_tuple(tile_position)
        frame_index = self._tile_point_to_frame_index(tile_point)
        tile = self._read_frame(frame_index)
        if self._page.jpegtables is not None:
            tile = Jpeg.add_jpeg_tables(
                tile, self._page.jpegtables, self._add_rgb_colorspace_fix
            )
        return tile

    def get_tiles(self, tile_positions: Sequence[Tuple[int, int]]) -> Iterator[bytes]:
        tile_points = [
            Point.from_tuple(tile_position) for tile_position in tile_positions
        ]
        frame_indices = [
            self._tile_point_to_frame_index(tile_point) for tile_point in tile_points
        ]
        tiles = self._read_frames(frame_indices)
        if self._page.jpegtables is not None:
            return (
                Jpeg.add_jpeg_tables(
                    tile, self._page.jpegtables, self._add_rgb_colorspace_fix
                )
                for tile in tiles
            )
        return iter(tiles)

    def get_decoded_tile(self, tile_position: Tuple[int, int]) -> np.ndarray:
        tile_point = Point.from_tuple(tile_position)
        if not self._check_if_tile_inside_image(tile_point):
            return np.full(
                self.tile_size.to_tuple() + (3,), 255, dtype=np.dtype(np.uint8)
            )

        shape: tuple[int, int, int, int]
        frame = self.get_tile(tile_position)
        frame_index = self._tile_point_to_frame_index(tile_point)
        data, _, shape = self._page.decode(frame, frame_index)
        assert isinstance(data, np.ndarray)
        data.shape = shape[1:]
        return data

    def _tile_point_to_frame_index(self, tile_point: Point) -> int:
        """Return linear frame index for tile position."""
        frame_index = tile_point.y * self.tiled_size.width + tile_point.x
        return frame_index
