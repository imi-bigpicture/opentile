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

"""Abstract image classes: the contract a TiffImage and its roles fulfil.

Concrete implementations of these live in `opentile.tiff_image_bases`.
"""

from abc import ABCMeta, abstractmethod
from collections.abc import Iterator, Sequence
from functools import cached_property
from typing import Optional, Union

import numpy as np
from tifffile import (
    COMPRESSION,
    PHOTOMETRIC,
)
from upath import UPath

from opentile.geometry import Size, SizeMm
from opentile.jpeg import JpegInfo
from opentile.jpeg2000 import Jpeg2000Info
from opentile.tile_overlap import TileOverlap


class TiffImage(metaclass=ABCMeta):
    """Abstract class for reading tiles from TiffPage."""

    @property
    def overlap(self) -> Optional[TileOverlap]:
        """How this level's stored tiles compose into opentile's regular tile grid.

        opentile presents each level as a regular, non-overlapping grid of ``tile_size``
        tiles (``tiled_size`` of them). ``None`` (the common case) means the stored
        tiles already form that regular grid and are served directly.

        When the stored tiles do not match that grid - because they overlap their
        neighbours (Trestle/Ventana) or use a different native tiling (JPEG XR ndpi) -
        this returns a ``TileOverlap`` describing where each stored tile is placed, so a
        consumer can compose (de-overlap and/or stitch) them into the regular grid.
        """
        return None

    @property
    @abstractmethod
    def supported_compressions(self) -> Optional[list[COMPRESSION]]:
        """List of compressions supported, or None if image is independent on
        compression."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def filepath(self) -> UPath:
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
    def encoded_info(self) -> Optional[Union[JpegInfo, Jpeg2000Info]]:
        """Parsed properties of the encoded image data: a `JpegInfo` for JPEG, a
        `Jpeg2000Info` for JPEG 2000, or None for other compressions."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def photometric_interpretation(self) -> PHOTOMETRIC:
        """Photometric interpretation of image, e.g. 'YCBCR" or "RGB"."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def subsampling(self) -> Optional[tuple[int, int]]:
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

    @property
    @abstractmethod
    def compressed_size(self) -> int:
        """Size of the compressed image data."""
        raise NotImplementedError()

    @abstractmethod
    def get_tile(self, tile_position: tuple[int, int]) -> bytes:
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
    def get_decoded_tile(self, tile_position: tuple[int, int]) -> np.ndarray:
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
    def get_tiles(self, tile_positions: Sequence[tuple[int, int]]) -> Iterator[bytes]:
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
        self, tile_positions: Sequence[tuple[int, int]]
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
            Set to True to not do any format-specific processing on the tile.

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

    @cached_property
    def np_dtype(self) -> np.dtype:
        """Numpy dtype of the image data."""
        if self.bit_depth <= 8:
            return np.dtype(np.uint8)
        if self.bit_depth <= 16:
            return np.dtype(np.uint16)
        if self.bit_depth <= 32:
            return np.dtype(np.uint32)
        raise NotImplementedError(f"Bit depth {self.bit_depth} not supported.")

    @cached_property
    def fill_value(self) -> int:
        data_type = self.np_dtype
        if (
            self.photometric_interpretation == PHOTOMETRIC.RGB
            or self.photometric_interpretation == PHOTOMETRIC.YCBCR
            or self.photometric_interpretation == PHOTOMETRIC.MINISWHITE
        ):
            return int(np.iinfo(data_type).max)
        if self.photometric_interpretation == PHOTOMETRIC.MINISBLACK:
            return int(np.iinfo(data_type).min)
        raise NotImplementedError(
            "Fill color not defined for photometric interpretation "
            f"{self.photometric_interpretation}."
        )


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
