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
from dataclasses import dataclass
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

from opentile.file import OpenTileFile
from opentile.geometry import Point, Region, Size, SizeMm
from opentile.jpeg import Jpeg, JpegInfo
from opentile.jpeg2000 import Jpeg2000, Jpeg2000Info


@dataclass
class TileOverlap:
    """Placement of natively overlapping source tiles onto a de-overlapped canvas.

    Some formats (Trestle, Ventana) store tiles that overlap their neighbours.
    `image_size` is the composed (de-overlapped) level size, and `tile_positions`
    maps each native source-tile grid position to the pixel top-left it occupies on
    that canvas. A consumer de-overlaps by reading each source tile
    (`TiffImage.get_tile`/`get_decoded_tile`) and placing it at its position; where
    tiles overlap, the tile with the greater position wins (its top-left covers the
    previous tile's discarded edge).
    """

    image_size: Size
    tile_positions: dict[Point, Point]

    @classmethod
    def from_regular_grid(
        cls, raw_size: Size, tile_size: Size, overlap: Size
    ) -> "TileOverlap":
        """Build placement for a regular tile grid where every tile overlaps its
        neighbour by a constant amount, keeping its top-left footprint (Trestle, and
        a single Ventana area).

        Parameters
        ----------
        raw_size: Size
            Size of the raw (overlapping) tile mosaic, i.e. the tiff page size.
        tile_size: Size
            Size of a source tile.
        overlap: Size
            Pixels each tile overlaps its right/bottom neighbour.
        """
        tiles_across = math.ceil(raw_size.width / tile_size.width)
        tiles_down = math.ceil(raw_size.height / tile_size.height)
        step_x = tile_size.width - overlap.width
        step_y = tile_size.height - overlap.height
        image_size = Size(
            raw_size.width - (tiles_across - 1) * overlap.width,
            raw_size.height - (tiles_down - 1) * overlap.height,
        )
        tile_positions = {
            Point(x, y): Point(x * step_x, y * step_y)
            for y in range(tiles_down)
            for x in range(tiles_across)
        }
        return cls(image_size, tile_positions)

    @classmethod
    def fit_to_size(
        cls, raw_size: Size, tile_size: Size, target_size: Size
    ) -> "TileOverlap":
        """Build a regular-grid placement whose composed size is exactly
        `target_size`, spacing the tiles uniformly to fill it. Used for reduced
        pyramid levels, whose de-overlapped size must equal the base level's composed
        size divided by the level downsample so the pyramid is a clean power of two
        (matching how the raw tiles halve). The per-tile overlap is derived from the
        target rather than measured.

        Parameters
        ----------
        raw_size: Size
            Size of the raw (overlapping) tile mosaic, i.e. the tiff page size.
        tile_size: Size
            Size of a source tile.
        target_size: Size
            Composed (de-overlapped) size the placement must span.
        """
        tiles_across = math.ceil(raw_size.width / tile_size.width)
        tiles_down = math.ceil(raw_size.height / tile_size.height)
        step_x = (
            (target_size.width - tile_size.width) / (tiles_across - 1)
            if tiles_across > 1
            else 0.0
        )
        step_y = (
            (target_size.height - tile_size.height) / (tiles_down - 1)
            if tiles_down > 1
            else 0.0
        )
        tile_positions = {
            Point(x, y): Point(round(x * step_x), round(y * step_y))
            for y in range(tiles_down)
            for x in range(tiles_across)
        }
        return cls(target_size, tile_positions)


class TiffImage(metaclass=ABCMeta):
    """Abstract class for reading tiles from TiffPage."""

    @property
    def overlap(self) -> Optional[TileOverlap]:
        """Placement of natively overlapping source tiles for de-overlapping, or
        None if the image's tiles do not overlap (the common case)."""
        return None

    @property
    @abstractmethod
    def supported_compressions(self) -> Optional[list[COMPRESSION]]:
        """List of compressions supported, or None if image is indenpendent on
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
        return PHOTOMETRIC(self._page.photometric)

    @property
    def subsampling(self) -> Optional[tuple[int, int]]:
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
            raise NotImplementedError(
                f"Pyramid index needs to be integer. Got {float_index} that is "
                f"more than set"
                f"tolerance {TOLERANCE} from the closest integer {index}. "
            )
        return index

    def _calculate_mpp(self, base_mpp: SizeMm, scale: float) -> SizeMm:
        return base_mpp * scale


class NativeTiledTiffImage(BaseTiffImage, metaclass=ABCMeta):
    """Meta class for images that are natively tiled (e.g. not ndpi)"""

    _cached_prefix_and_scan_offset: Optional[tuple[bytes, int]] = None

    def get_tile(self, tile_position: tuple[int, int]) -> bytes:
        tile_point = Point.from_tuple(tile_position)
        frame_index = self._tile_point_to_frame_index(tile_point)
        tile = self._read_frame(frame_index)
        return self._add_jpeg_tables(tile)

    def get_tiles(self, tile_positions: Sequence[tuple[int, int]]) -> Iterator[bytes]:
        tile_points = [
            Point.from_tuple(tile_position) for tile_position in tile_positions
        ]
        frame_indices = [
            self._tile_point_to_frame_index(tile_point) for tile_point in tile_points
        ]
        tiles = self._read_frames(frame_indices)
        return (self._add_jpeg_tables(tile) for tile in tiles)

    def _add_jpeg_tables(self, tile: bytes) -> bytes:
        """Prepend jpeg tables (and, for svs, the rgb color space fix) to an
        abbreviated tile, reusing the page's cached prefix."""
        jpegtables = self._page.jpegtables
        if jpegtables is None:
            return tile
        if self._cached_prefix_and_scan_offset is None:
            self._cached_prefix_and_scan_offset = Jpeg.calculate_prefix_and_scan_offset(
                tile, jpegtables, self._add_rgb_colorspace_fix
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


class OverlappingLevelTiffImage(NativeTiledTiffImage, LevelTiffImage):
    """Level image for natively tiled formats whose source tiles overlap (Trestle,
    Ventana).

    Raw (overlapping) source tiles are read by native grid position exactly like any
    NativeTiledTiffImage; `overlap` additionally describes how to compose them into a
    de-overlapped image. Scale and pyramid index are derived from the composed
    (de-overlapped) size, not the raw overlapping tile mosaic, so the format-specific
    overlap parsing lives in the tiler and is passed in as `overlap`.
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
        return [COMPRESSION.JPEG]

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


class SubcellReducedLevelTiffImage(OverlappingLevelTiffImage):
    """Reduced pyramid level for formats (Ventana) whose reduced pages pack the
    ``downsample * downsample`` level-0 tiles of each physical tile, every level-0
    tile downsampled to ``tile / downsample``.

    Placement is inherited from the level-0 stitch scaled by ``1/downsample`` rather
    than re-derived from the reduced page (which has no per-tile placement of its
    own), so tiles meet seamlessly and a feature keeps the same scene position across
    levels. Each "tile" exposed here is one sub-cell of a physical page tile;
    ``overlap.tile_positions`` maps the level-0 tile grid position to its
    de-overlapped destination at this level.
    """

    def __init__(
        self,
        page: TiffPage,
        file: OpenTileFile,
        base_mpp: SizeMm,
        scale: float,
        overlap: TileOverlap,
        downsample: int,
    ):
        self._downsample = downsample
        self._subcell_size = Size(
            math.ceil(page.tilewidth / downsample),
            math.ceil(page.tilelength / downsample),
        )
        self._phys_grid_cols = math.ceil(page.imagewidth / page.tilewidth)
        self._phys_grid_rows = math.ceil(page.imagelength / page.tilelength)
        super().__init__(page, file, base_mpp, scale, overlap)

    @property
    def tile_size(self) -> Size:
        return self._subcell_size

    def get_tile(self, tile_position: tuple[int, int]) -> bytes:
        # A sub-cell is not independently encoded; return its physical tile's frame
        # (only used for codec/subsampling introspection - reduced levels transcode).
        gx, gy = tile_position
        px = gx // self._downsample
        py = gy // self._downsample
        return self._add_jpeg_tables(self._read_frame(py * self._phys_grid_cols + px))

    def get_decoded_tile(self, tile_position: tuple[int, int]) -> np.ndarray:
        """Sub-cell for level-0 tile ``(gx, gy)``: the ``(gx%d, gy%d)`` cell of
        physical tile ``(gx//d, gy//d)`` - that level-0 tile downsampled to this
        level."""
        gx, gy = tile_position
        d = self._downsample
        sub_w, sub_h = self._subcell_size.to_tuple()
        physical = self._read_physical(gx // d, gy // d)
        sx = round((gx % d) * self._page.tilewidth / d)
        sy = round((gy % d) * self._page.tilelength / d)
        cell = physical[sy : sy + sub_h, sx : sx + sub_w]
        if cell.shape[0] == sub_h and cell.shape[1] == sub_w:
            return cell
        out = np.full(
            (sub_h, sub_w) + cell.shape[2:], self.fill_value, dtype=self.np_dtype
        )
        out[: cell.shape[0], : cell.shape[1]] = cell
        return out

    def _read_physical(self, px: int, py: int) -> np.ndarray:
        if not (0 <= px < self._phys_grid_cols and 0 <= py < self._phys_grid_rows):
            shape: tuple[int, ...] = (self._page.tilelength, self._page.tilewidth)
            if self.samples_per_pixel > 1:
                shape = shape + (self.samples_per_pixel,)
            return np.full(shape, self.fill_value, dtype=self.np_dtype)
        frame_index = py * self._phys_grid_cols + px
        frame = self._add_jpeg_tables(self._read_frame(frame_index))
        data, _, _ = self._page.decode(frame, frame_index)
        assert isinstance(data, np.ndarray)
        return data.squeeze((0, 3) if self.samples_per_pixel == 1 else 0)
