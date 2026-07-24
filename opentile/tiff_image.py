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

import math
from abc import ABCMeta, abstractmethod
from collections.abc import Iterator, Sequence
from functools import cached_property
from typing import Optional, Union

import numpy as np
from tifffile import (
    COMPRESSION,
    PHOTOMETRIC,
    TiffPage,
    TiffTags,
)
from upath import UPath

from opentile.exceptions import (
    NonDyadicPyramidLevelError,
    NonSupportedCompressionError,
)
from opentile.file import OpenTileFile
from opentile.geometry import Point, Region, Size, SizeMm
from opentile.jpeg import Jpeg, JpegInfo
from opentile.jpeg2000 import Jpeg2000, Jpeg2000Info
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
            raise NonSupportedCompressionError(
                f"Non-supported compression {page.compression}."
            )
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
    def filepath(self) -> UPath:
        return self._file.filepath

    @property
    def suggested_minimum_chunk_size(self) -> int:
        return 1

    @property
    def compression(self) -> COMPRESSION:
        return COMPRESSION(self._page.compression)

    @property
    def encoded_info(self) -> Optional[Union[JpegInfo, Jpeg2000Info]]:
        return self._encoded_info

    @cached_property
    def _encoded_info(self) -> Optional[Union[JpegInfo, Jpeg2000Info]]:
        compression = self.compression
        if compression == COMPRESSION.JPEG:
            header = self._page.jpegheader
            if header is not None:
                return Jpeg.info(bytes(header))
            return Jpeg.info(self.get_tile((0, 0)))
        if compression in (
            COMPRESSION.JPEG2000,
            COMPRESSION.JPEG_2000_LOSSY,
            COMPRESSION.APERIO_JP2000_YCBC,
            COMPRESSION.APERIO_JP2000_RGB,
        ):
            return Jpeg2000.parse(self.get_tile((0, 0)))
        return None

    @property
    def photometric_interpretation(self) -> PHOTOMETRIC:
        info = self.encoded_info
        if isinstance(info, JpegInfo):
            if info.rgb_signalled:
                return PHOTOMETRIC.RGB
            if info.subsampling not in (None, (1, 1)):
                return PHOTOMETRIC.YCBCR
        elif isinstance(info, Jpeg2000Info):
            if info.uses_mct or info.subsampling not in (None, (1, 1)):
                return PHOTOMETRIC.YCBCR
            if self.compression == COMPRESSION.APERIO_JP2000_YCBC:
                return PHOTOMETRIC.YCBCR
        return PHOTOMETRIC(self._page.photometric)

    @property
    def subsampling(self) -> Optional[tuple[int, int]]:
        info = self.encoded_info
        if info is not None:
            return info.subsampling
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
        return "1"

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

    @property
    def compressed_size(self) -> int:
        return self._compressed_size

    @cached_property
    def _compressed_size(self) -> int:
        frames = sum(self._page.databytecounts)
        if self._page.jpegheader is not None:
            jpeg_header_length = len(self._page.jpegheader)
        elif self._page.jpegtables is not None:
            jpeg_header_length = len(self._page.jpegtables)
        else:
            jpeg_header_length = 0
        return frames + len(self._page.dataoffsets) * jpeg_header_length

    def get_tiles(self, tile_positions: Sequence[tuple[int, int]]) -> Iterator[bytes]:
        return (self.get_tile(tile) for tile in tile_positions)

    def get_decoded_tiles(
        self, tile_positions: Sequence[tuple[int, int]]
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

    def _read_frames(self, indices: Sequence[int]) -> list[bytes]:
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
            raise NonDyadicPyramidLevelError(
                f"Pyramid index needs to be integer. Got {float_index} that is more "
                f"than set tolerance {TOLERANCE} from the closest integer {index}. "
            )
        return index

    def _calculate_mpp(self, base_mpp: SizeMm, scale: float) -> SizeMm:
        return base_mpp * scale


class NativeTiledTiffImage(BaseTiffImage, metaclass=ABCMeta):
    """Meta class for images that are natively tiled (e.g. not ndpi)"""

    _cached_prefix_and_scan_offset: Optional[tuple[bytes, int]] = None

    def __init__(
        self,
        page: TiffPage,
        file: OpenTileFile,
        add_rgb_colorspace_fix: bool = False,
    ):
        super().__init__(page, file, add_rgb_colorspace_fix)
        self._jpeg_tables = page.jpegtables

    def get_tile(self, tile_position: tuple[int, int]) -> bytes:
        tile_point = Point.from_tuple(tile_position)
        return self._read_tile_frame(self._tile_point_to_frame_index(tile_point))

    def get_tiles(self, tile_positions: Sequence[tuple[int, int]]) -> Iterator[bytes]:
        tile_points = [
            Point.from_tuple(tile_position) for tile_position in tile_positions
        ]
        frame_indices = [
            self._tile_point_to_frame_index(tile_point) for tile_point in tile_points
        ]
        tiles = self._read_frames(frame_indices)
        if not self._jpeg_tables:
            return iter(tiles)
        return (self._add_jpeg_tables(tile, self._jpeg_tables) for tile in tiles)

    def _read_tile_frame(self, frame_index: int) -> bytes:
        """Read a tile frame, prepending the page's jpeg tables to the abbreviated
        frame when the page uses them (otherwise the frame is returned as read)."""
        tile = self._read_frame(frame_index)
        if not self._jpeg_tables:
            return tile
        return self._add_jpeg_tables(tile, self._jpeg_tables)

    def _add_jpeg_tables(self, tile: bytes, tables: bytes) -> bytes:
        """Prepend the page's jpeg tables (and, for svs, the rgb color space fix) to an
        abbreviated jpeg tile, reusing the page's cached prefix. Only called when the
        page has jpeg tables (see `_has_jpeg_tables`)."""
        if self._cached_prefix_and_scan_offset is None:
            self._cached_prefix_and_scan_offset = Jpeg.calculate_prefix_and_scan_offset(
                tile, tables, self._add_rgb_colorspace_fix
            )
        prefix, scan_offset = self._cached_prefix_and_scan_offset
        return Jpeg.add_jpeg_prefix(prefix, scan_offset, tile)

    def get_decoded_tile(self, tile_position: tuple[int, int]) -> np.ndarray:
        tile_point = Point.from_tuple(tile_position)
        if not self._check_if_tile_inside_image(tile_point):
            shape = self.tile_size.to_tuple()
            if self.samples_per_pixel > 1:
                shape = self.tile_size.to_tuple() + (self.samples_per_pixel,)

            return np.full(
                shape,
                fill_value=self.fill_value,
                dtype=self.np_dtype,
            )

        frame = self.get_tile(tile_position)
        frame_index = self._tile_point_to_frame_index(tile_point)
        data, _, _ = self._page.decode(frame, frame_index)
        assert isinstance(data, np.ndarray)
        return data.squeeze((0, 3) if self.samples_per_pixel == 1 else 0)

    def _tile_point_to_frame_index(self, tile_point: Point) -> int:
        """Return linear frame index for tile position."""
        frame_index = tile_point.y * self.tiled_size.width + tile_point.x
        return frame_index


class NativeTiledLevelImage(NativeTiledTiffImage, LevelTiffImage):
    """A pyramid level whose tiles are natively tiled and served as-is, with scale,
    pyramid index, and pixel spacing derived from the base level. Used by formats that
    only need this plain passthrough level (e.g. Huron, Mikroscan, Motic)."""

    def __init__(
        self, page: TiffPage, file: OpenTileFile, base_size: Size, base_mpp: SizeMm
    ):
        """Parameters
        ----------
        page: TiffPage
            TiffPage defining the level.
        file: OpenTileFile
            File to read data from.
        base_size: Size
            Size of the base level in the pyramid.
        base_mpp: SizeMm
            Pixel spacing (um/pixel) of the base level in the pyramid.
        """
        super().__init__(page, file)
        self._base_size = base_size
        self._base_mpp = base_mpp
        self._scale = self._calculate_scale(base_size)
        self._pyramid_index = self._calculate_pyramidal_index(self._scale)
        self._mpp = self._calculate_mpp(base_mpp, self._scale)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._page}, {self._file}, "
            f"{self._base_size}, {self._base_mpp})"
        )

    @property
    def supported_compressions(self) -> Optional[list[COMPRESSION]]:
        return None

    @property
    def mpp(self) -> SizeMm:
        return self._mpp

    @property
    def pixel_spacing(self) -> SizeMm:
        return self.mpp / 1000

    @property
    def scale(self) -> float:
        return self._scale

    @property
    def pyramid_index(self) -> int:
        return self._pyramid_index


class DecodedTiledTiffImage(BaseTiffImage, metaclass=ABCMeta):
    """Meta class for images that are not natively tiled and have no per-tile encoded
    representation (e.g. strip-stored or uncompressed pages). The whole page is decoded
    once and served as a tile grid; `get_tile` returns the raw pixel bytes of the
    cropped region since there is nothing to pass through.

    Subclasses must set `self._tile_size` (the requested grid tile size) and recompute
    `self._tiled_region` after calling `super().__init__`."""

    @cached_property
    def _decoded_image(self) -> np.ndarray:
        """The whole page decoded once (tifffile reassembles the strips/tiles)."""
        return self._page.asarray(squeeze=True)

    def get_decoded_tile(self, tile_position: tuple[int, int]) -> np.ndarray:
        """Return the decoded tile at tile position, cropped from the decoded page and
        padded with the fill value where the tile extends past the image edge.

        Parameters
        ----------
        tile_position: Tuple[int, int]
            Tile position to get.

        Returns
        ----------
        np.ndarray
            Decoded tile at position.
        """
        point = Point.from_tuple(tile_position)
        if not self._check_if_tile_inside_image(point):
            raise ValueError(f"Tile {tile_position} is outside {self.tiled_size}.")
        left = point.x * self.tile_size.width
        top = point.y * self.tile_size.height
        region = self._decoded_image[
            top : top + self.tile_size.height, left : left + self.tile_size.width
        ]
        pad_height = self.tile_size.height - region.shape[0]
        pad_width = self.tile_size.width - region.shape[1]
        if pad_height or pad_width:
            padding = [(0, pad_height), (0, pad_width)] + [(0, 0)] * (region.ndim - 2)
            region = np.pad(region, padding, constant_values=self.fill_value)
        return region

    def get_tile(self, tile_position: tuple[int, int]) -> bytes:
        """Return the tile at tile position as raw pixel bytes. There is no per-tile
        encoded data, so the "encoded" tile is the raw bytes of the decoded tile.

        Parameters
        ----------
        tile_position: Tuple[int, int]
            Tile position to get.

        Returns
        ----------
        bytes
            Raw pixel bytes of the tile at position.
        """
        return self.get_decoded_tile(tile_position).tobytes()


class OverlappingLevelTiffImage(NativeTiledTiffImage, LevelTiffImage):
    """Level image whose stored tiles do not form opentile's regular tile grid and so
    must be composed by the consumer (see `TiffImage.overlap`).

    Despite the name, this covers two cases: formats whose source tiles overlap their
    neighbours (Trestle, Ventana), and levels with a non-overlapping but irregular
    native tiling (JPEG XR ndpi). The raw stored tiles are read by native grid position
    exactly like any NativeTiledTiffImage; `overlap` additionally describes how to
    compose them into the regular grid - de-overlapping when the tiles overlap, or a
    plain stitch when the overlap is zero.
    """

    def __init__(
        self,
        page: TiffPage,
        file: OpenTileFile,
        base_mpp: SizeMm,
        scale: float,
        overlap: TileOverlap,
    ):
        """
        Parameters
        ----------
        page: TiffPage
            TiffPage defining the page.
        file: OpenTileFile
            File to read data from.
        base_mpp: SizeMm
            Mpp (um/pixel) for the base level in the pyramid.
        scale: float
            Downsample of this level relative to the base level. Computed by the
            tiler from whichever dimension forms a clean pyramid (the de-overlapped
            size for Trestle, the raw mosaic for Ventana).
        overlap: TileOverlap
            Placement of this level's overlapping source tiles.
        """
        super().__init__(page, file)
        self._base_mpp = base_mpp
        self._overlap = overlap
        self._scale = scale
        self._pyramid_index = self._calculate_pyramidal_index(scale)
        self._mpp = self._calculate_mpp(base_mpp, scale)

    @property
    def overlap(self) -> TileOverlap:
        return self._overlap

    @property
    def supported_compressions(self) -> Optional[list[COMPRESSION]]:
        return None

    @property
    def pixel_spacing(self) -> SizeMm:
        return self._mpp / 1000

    @property
    def mpp(self) -> SizeMm:
        return self._mpp

    @property
    def scale(self) -> float:
        return self._scale

    @property
    def pyramid_index(self) -> int:
        return self._pyramid_index


class SparseTiledLevelImage(NativeTiledTiffImage, LevelTiffImage):
    """Level image for a natively tiled JPEG pyramid whose tiles may be sparse.

    Missing tiles (zero-length frame, or a frame index past the stored frames) are
    served as a blank (white) tile built from the first valid frame. Used by formats
    that leave un-scanned tiles out of the pyramid (Philips, Argos).
    """

    def __init__(
        self,
        page: TiffPage,
        file: OpenTileFile,
        base_size: Size,
        base_mpp: SizeMm,
        jpeg: Jpeg,
    ):
        """
        Parameters
        ----------
        page: TiffPage
            TiffPage defining the page.
        file: OpenTileFile
            File to read data from.
        base_size: Size
            Size of base level in pyramid.
        base_mpp: SizeMm
            Mpp (um/pixel) for base level in pyramid.
        jpeg: Jpeg
            Jpeg instance to use.
        """
        super().__init__(page, file)
        self._jpeg = jpeg
        self._base_size = base_size
        self._base_mpp = base_mpp
        self._scale = self._calculate_scale(base_size)
        self._pyramid_index = self._calculate_pyramidal_index(self._scale)
        self._mpp = self._calculate_mpp(self._base_mpp, self._scale)
        self._blank_tile = self._create_blank_tile()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._page}, {self._file}, "
            f"{self._base_size}, {self._base_mpp}, {self._jpeg})"
        )

    @property
    def supported_compressions(self) -> Optional[list[COMPRESSION]]:
        """Handling of sparse tiles assumes JPEG compression."""
        return [COMPRESSION.JPEG]

    @property
    def pixel_spacing(self) -> SizeMm:
        return self.mpp / 1000

    @property
    def mpp(self) -> SizeMm:
        return self._mpp

    @property
    def scale(self) -> float:
        return self._scale

    @property
    def pyramid_index(self) -> int:
        return self._pyramid_index

    @property
    def blank_tile(self) -> bytes:
        """Return blank tile."""
        return self._blank_tile

    def _create_blank_tile(self, luminance: float = 1.0) -> bytes:
        """Create a blank tile from a valid tile. Uses the first found
        valid frame (first frame with non-zero value in databytescounts) and
        fills that image with white.

        Parameters
        ----------
        luminance: float = 1.0
            Luminance for tile, 0 = black, 1 = white.

        Returns
        ----------
        bytes:
            Frame bytes from blank tile.

        """
        try:
            # Get first frame in page that is not 0 bytes
            valid_frame_index = next(
                index
                for index, datalength in enumerate(self._page.databytecounts)
                if datalength != 0
            )
        except StopIteration as exception:
            raise ValueError("Could not find valid frame in image.") from exception
        tile = self._read_frame(valid_frame_index)
        if self._page.jpegtables is not None:
            prefix, scan_offset = Jpeg.calculate_prefix_and_scan_offset(
                tile, self._page.jpegtables, False
            )
            tile = Jpeg.add_jpeg_prefix(prefix, scan_offset, tile)
        tile = self._jpeg.fill_frame(tile, luminance)
        return tile

    def _read_frame(self, index: int) -> bytes:
        """Read frame at frame index from image. Return blank tile if tile is
        sparse (length of frame is zero or frame index is outside length of
        frames).

        Parameters
        ----------
        index: int
            Frame index to read from image.

        Returns
        ----------
        bytes:
            Frame bytes from frame index or blank tile.

        """
        if (
            index >= len(self._page.databytecounts)
            or self._page.databytecounts[index] == 0
        ):
            # Sparse tile
            return self.blank_tile
        return super()._read_frame(index)


class StripedTiffImage(BaseTiffImage):
    """Non-tiled image stored as horizontal strips (e.g. thumbnail/overview images).

    `get_tile((0, 0))` returns the whole image: JPEG strips are concatenated into a
    single JPEG scan, other compressions (NONE, LZW) have their strip bytes joined.
    """

    def __init__(self, page: TiffPage, file: OpenTileFile, jpeg: Jpeg):
        """
        Parameters
        ----------
        page: TiffPage
            TiffPage defining the page.
        file: OpenTileFile
            File to read data from.
        jpeg: Jpeg
            Jpeg instance to use.
        """
        add_rgb_colorspace_fix = (
            page.compression == COMPRESSION.JPEG and page.photometric == PHOTOMETRIC.RGB
        )
        super().__init__(page, file, add_rgb_colorspace_fix)
        self._jpeg = jpeg

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._page}, {self._file}, {self._jpeg}"

    @property
    def supported_compressions(self) -> Optional[list[COMPRESSION]]:
        """`get_tile()` concatenates JPEG scans for JPEG-compressed stripes and
        returns the raw stripe bytes for uncompressed (NONE) and LZW stripes."""
        return [COMPRESSION.JPEG, COMPRESSION.NONE, COMPRESSION.LZW]

    def get_tile(self, tile_position: tuple[int, int]) -> bytes:
        if tile_position != (0, 0):
            raise ValueError("Non-tiled image, expected tile_position (0, 0)")
        indices = range(len(self._page.dataoffsets))
        frames = self._read_frames(indices)
        if self.compression != COMPRESSION.JPEG:
            return b"".join(frames)
        return self._jpeg.concatenate_scans(
            iter(frames), self._page.jpegtables, self._add_rgb_colorspace_fix
        )

    def get_decoded_tile(self, tile_position: tuple[int, int]) -> np.ndarray:
        if tile_position != (0, 0):
            raise ValueError("Non-tiled image, expected tile_position (0, 0)")
        return self._page.asarray(squeeze=True)


class StripedThumbnailImage(StripedTiffImage, ThumbnailTiffImage):
    """Striped thumbnail image, scaled relative to the pyramid base level."""

    def __init__(
        self,
        page: TiffPage,
        file: OpenTileFile,
        base_size: Size,
        base_mpp: SizeMm,
        jpeg: Jpeg,
    ):
        super().__init__(page, file, jpeg)
        self._base_size = base_size
        self._base_mpp = base_mpp
        self._scale = self._calculate_scale(base_size)
        self._mpp = self._calculate_mpp(base_mpp, self._scale)

    @property
    def mpp(self) -> SizeMm:
        return self._mpp

    @property
    def pixel_spacing(self) -> SizeMm:
        return self.mpp / 1000

    @property
    def scale(self) -> float:
        return self._scale


class StripedAssociatedImage(StripedTiffImage, AssociatedTiffImage):
    """Striped associated image (e.g. overview/macro) with no defined pixel spacing."""

    @property
    def pixel_spacing(self) -> Optional[SizeMm]:
        return None


class DecodedSingleFrameImage(BaseTiffImage, metaclass=ABCMeta):
    """Meta class for an uncompressed, multi-strip page served as one tile.

    The strips are read and assembled into one raw-pixel tile; there is no per-tile
    encoded representation to pass through, so the image is served uncompressed. Unlike
    `DecodedTiledTiffImage` the whole page is one tile, not a grid.
    """

    @property
    def supported_compressions(self) -> Optional[list[COMPRESSION]]:
        # Only uncompressed pages: the tile is served as raw pixels, so the reported
        # (inherited) compression of NONE must match the served bytes.
        return [COMPRESSION.NONE]

    def get_decoded_tile(self, tile_position: tuple[int, int]) -> np.ndarray:
        if tile_position != (0, 0):
            raise ValueError("Non-tiled image, expected tile_position (0, 0)")
        return self._page.asarray(squeeze=True)

    def get_tile(self, tile_position: tuple[int, int]) -> bytes:
        return self.get_decoded_tile(tile_position).tobytes()


class DecodedAssociatedImage(DecodedSingleFrameImage, AssociatedTiffImage):
    """Uncompressed associated image (label or macro) served as raw pixels."""

    @property
    def pixel_spacing(self) -> Optional[SizeMm]:
        return None


class DecodedThumbnailImage(DecodedSingleFrameImage, ThumbnailTiffImage):
    """Uncompressed thumbnail served as raw pixels, scaled from the base level."""

    def __init__(
        self, page: TiffPage, file: OpenTileFile, base_size: Size, base_mpp: SizeMm
    ):
        """
        Parameters
        ----------
        page: TiffPage
            TiffPage defining the thumbnail.
        file: OpenTileFile
            File to read data from.
        base_size: Size
            Size of the base level in the pyramid.
        base_mpp: SizeMm
            Pixel spacing (um/pixel) of the base level in the pyramid.
        """
        super().__init__(page, file)
        self._base_size = base_size
        self._scale = self._calculate_scale(base_size)
        self._mpp = self._calculate_mpp(base_mpp, self._scale)

    @property
    def scale(self) -> float:
        return self._scale

    @property
    def pixel_spacing(self) -> SizeMm:
        return self._mpp / 1000


class SingleFrameAssociatedImage(BaseTiffImage, AssociatedTiffImage):
    """Associated image stored as a single frame, served as one tile.

    The stored frame is passed through untouched, whatever its compression (e.g. an lzw
    label). Used by formats whose label is written as one frame rather than as strips
    (svs and the Aperio-like motic).
    """

    @property
    def pixel_spacing(self) -> Optional[SizeMm]:
        return None

    @property
    def supported_compressions(self) -> Optional[list[COMPRESSION]]:
        return None

    def get_tile(self, tile_position: tuple[int, int]) -> bytes:
        if tile_position != (0, 0):
            raise ValueError("Non-tiled image, expected tile_position (0, 0)")
        return self._read_frame(0)

    def get_decoded_tile(self, tile_position: tuple[int, int]) -> np.ndarray:
        if tile_position != (0, 0):
            raise ValueError("Non-tiled image, expected tile_position (0, 0)")
        return self._page.asarray(squeeze=True)
