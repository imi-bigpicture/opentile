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
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from tifffile import FileHandle, TiffPage
from tifffile.tifffile import TIFF

from opentile.common import OpenTilePage, Tiler
from opentile.geometry import Point, Region, Size, SizeMm
from opentile.jpeg import Jpeg, JpegCropError


def get_value_from_ndpi_comments(
    comments: str,
    value_name: str,
    value_type: Any
) -> Any:
    """Read value from ndpi comment string."""
    for line in comments.split("\n"):
        if value_name in line:
            value_string = line.split('=')[1]
            return(value_type(value_string))


class NdpiCache():
    """Cache for bytes ordered by tile position. Oldest entry is removed when
    size of content is above set size."""
    def __init__(self, size: int):
        """Create cache for size items.

        Parameters
        ----------
        size: int
            Size of the cache.

        """
        self._size = size
        self._content: Dict[Point, bytes] = {}
        self._history: List[Point] = []

    def __len__(self) -> int:
        return len(self._history)

    def __str__(self) -> str:
        return (
            f"{type(self).__name__} of size {len(self)} "
            f"and max size {self._size}"
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._size})"

    def __setitem__(self, key: Point, value: bytes) -> None:
        """Set item in cache. Remove old items if needed.

        Parameters
        ----------
        key: Point
            Key for item to set.
        value: bytes:
            Value for item to set.

        """
        self._content[key] = value
        self._history.append(key)
        self._remove_old()

    def __getitem__(self, key: Point) -> bytes:
        """Get item from cache.

        Parameters
        ----------
        key: Point
            Key for item to get.

        Returns
        ----------
        bytes
            Value for key.
        """
        return self._content[key]

    def __contains__(self, key: Point) -> bool:
        """Return true if key in cache.

        Parameters
        ----------
        key: Point
            Key to check for.

        Returns
        ----------
        bool
            True if key is in cache.
        """
        return key in self._content

    @property
    def size(self) -> int:
        return self._size

    def update(self, items: Dict[Point, bytes]) -> None:
        """Update items in cache. Remove old items if needed.

        Parameters
        ----------
        items: Dict[Point, bytes]
            Items to update.

        """
        self._content.update(items)
        self._history += list(items.keys())
        self._remove_old()

    def _remove_old(self) -> None:
        """Remove old items in cache if needed."""
        while len(self._history) > self._size:
            key_to_remove = self._history.pop(0)
            self._content.pop(key_to_remove)


class NdpiTile:
    """Defines a tile by position and coordinates and size for cropping out
    out frame."""
    def __init__(
        self,
        position: Point,
        tile_size: Size,
        frame_size: Size
    ) -> None:
        """Create a ndpi tile and calculate cropping parameters.

        Parameters
        ----------
        position: Point
            Tile position.
        tile_size: Size
            Tile size.
        frame_size: Size
            Frame size.

        """
        self._position = position
        self._tile_size = tile_size
        self._frame_size = frame_size

        self._tiles_per_frame = Size.max(
            self._frame_size // self._tile_size,
            Size(1, 1)
        )
        position_inside_frame: Point = (
            (self.position * self._tile_size)
            % Size.max(self._frame_size, self._tile_size)
        )
        self._left = position_inside_frame.x
        self._top = position_inside_frame.y
        self._frame_position = (
            (self.position // self._tiles_per_frame) * self._tiles_per_frame
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, NdpiTile):
            return (
                self.position == other.position and
                self._tile_size == other._tile_size and
                self._frame_size == other._frame_size
            )
        return NotImplemented

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self.position}, {self._tile_size}, "
            f"{self._frame_size})"
        )

    def __str__(self) -> str:
        return f"{type(self).__name__} of position {self.position}"

    @property
    def position(self) -> Point:
        """Return position of tile."""
        return self._position

    @property
    def frame_position(self) -> Point:
        """Return frame position for tile."""
        return self._frame_position

    @property
    def left(self) -> int:
        """Return left coordinate for tile inside frame."""
        return self._left

    @property
    def top(self) -> int:
        """Return top coordinate for tile inside frame."""
        return self._top

    @property
    def width(self) -> int:
        """Return width for tile inside frame."""
        return self._tile_size.width

    @property
    def height(self) -> int:
        """Return height for tile inside frame."""
        return self._tile_size.height

    @property
    def frame_size(self) -> Size:
        """Return frame size."""
        return self._frame_size


class NdpiFrameJob:
    """A list of tiles to create from a frame. Tiles need to have the same
    frame position."""
    def __init__(
        self,
        tiles: Union[NdpiTile, List[NdpiTile]]
    ) -> None:
        """Create a frame job from given tile(s).

        Parameters
        ----------
        tiles: Union[NdpiTile, List[NdpiTile]]
            Tile(s) to base the frame job on.

        """
        if isinstance(tiles, NdpiTile):
            tiles = [tiles]
        first_tile = tiles.pop(0)
        self._position = first_tile.frame_position
        self._frame_size = first_tile.frame_size
        self._tiles: List[NdpiTile] = [first_tile]
        for tile in tiles:
            self.append(tile)

    def __repr__(self) -> str:
        return f"{type(self).__name__}{self.tiles})"

    def __str__(self) -> str:
        return f"{type(self).__name__} of tiles {self.tiles}"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, NdpiFrameJob):
            return self.tiles == other.tiles
        return NotImplemented

    @property
    def position(self) -> Point:
        """The frame position of the frame job."""
        return self._position

    @property
    def frame_size(self) -> Size:
        """Frame size required for reading tiles in NdpiFrameJob."""
        return self._frame_size

    @property
    def tiles(self) -> List[NdpiTile]:
        """Tiles in NdpiFrameJob."""
        return self._tiles

    @property
    def crop_parameters(self) -> List[Tuple[int, int, int, int]]:
        """Parameters for croping tiles from frame in NdpiFrameJob."""
        return [
            (tile.left, tile.top, tile.width, tile.height)
            for tile in self._tiles
        ]

    def append(self, tile: NdpiTile) -> None:
        """Add a tile to the tile job."""
        if tile.frame_position != self.position:
            raise ValueError(f"{tile} does not match {self} frame position")
        self._tiles.append(tile)


class NdpiPage(OpenTilePage):
    _pyramid_index = 0

    def __init__(
        self,
        page: TiffPage,
        fh: FileHandle,
        jpeg: Jpeg
    ):
        """Ndpi page that should not be tiled (e.g. overview or label).
        Image data is assumed to be jpeg.

        Parameters
        ----------
        page: TiffPage
            TiffPage defining the page.
        fh: FileHandle
            Filehandler to read data from.
        jpeg: Jpeg
            Jpeg instance to use.
        """
        super().__init__(page, fh)
        if self.compression != 'COMPRESSION.JPEG':
            raise NotImplementedError(
                f'{self.compression} is unsupported for ndpi '
                '(Only jpeg is supported)'
            )
        self._jpeg = jpeg
        try:
            # Defined in nm
            self._focal_plane = (
                page.ndpi_tags['ZOffsetFromSlideCenter'] / 1000.0
            )
        except KeyError:
            self._focal_plane = 0.0

        self._mpp = self._get_mpp_from_page()
        self._properties = self._get_properties()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._page}, {self._fh}, {self._jpeg}"
        )

    @property
    def focal_plane(self) -> float:
        """Return focal plane (in um)."""
        return self._focal_plane

    @property
    def pixel_spacing(self) -> SizeMm:
        """Return pixel spacing in mm per pixel."""
        return self.mpp / 1000.0

    @property
    def mpp(self) -> SizeMm:
        """Return pixel spacing in um per pixel."""
        return self._get_mpp_from_page()

    @property
    def properties(self) -> Dict[str, Any]:
        """Return dictionary with ndpifile properties."""
        return self._properties

    def get_tile(self, tile_position: Tuple[int, int]) -> bytes:
        """Return tile for tile position.

        Parameters
        ----------
        tile_position: Tuple[int, int]
            Tile position to get.

        Returns
        ----------
        bytes
            Produced tile at position.
        """
        if tile_position != (0, 0):
            raise ValueError("Non-tiled page, expected tile_position (0, 0)")
        return self._read_frame(0)

    def get_decoded_tile(self, tile_position: Tuple[int, int]) -> np.ndarray:
        """Return decoded tile for tile position.

        Parameters
        ----------
        tile_position: Tuple[int, int]
            Tile position to get.

        Returns
        ----------
        bytes
            Produced tile at position.
        """
        tile = self.get_tile(tile_position)
        return self._jpeg.decode(tile)

    def _get_mpp_from_page(self) -> SizeMm:
        """Return pixel spacing in um/pixel."""
        x_resolution = self.page.tags['XResolution'].value[0]
        y_resolution = self.page.tags['YResolution'].value[0]
        resolution_unit = self.page.tags['ResolutionUnit'].value
        if resolution_unit != TIFF.RESUNIT.CENTIMETER:
            raise ValueError("Unkown resolution unit")
        # 10*1000 um per cm
        mpp_x = 10*1000/x_resolution
        mpp_y = 10*1000/y_resolution
        return SizeMm(mpp_x, mpp_y)

    def _get_properties(self) -> Dict[str, Any]:
        """Return dictionary with ndpifile properties."""
        ndpi_tags = self.page.ndpi_tags
        manufacturer = getattr(ndpi_tags, 'Make', None)
        model = getattr(ndpi_tags, 'Model', None)
        software_version = getattr(ndpi_tags, 'Software', None)
        if software_version is not None:
            software_versions = [software_version]
        else:
            software_versions = []
        device_serial_number = getattr(ndpi_tags, 'ScannerSerialNumber', None)
        aquisition_datetime = self._get_value_from_tiff_tags(
            self.page.tags, 'DateTime'
        )
        photometric_interpretation = self._get_value_from_tiff_tags(
            self.page.tags, 'PhotometricInterpretation'
        )
        return {
            'aquisition_datetime': aquisition_datetime,
            'device_serial_number': device_serial_number,
            'manufacturer': manufacturer,
            'model': model,
            'software_versions': software_versions,
            'photometric_interpretation': photometric_interpretation
        }


class NdpiTiledPage(NdpiPage, metaclass=ABCMeta):
    def __init__(
        self,
        page: TiffPage,
        fh: FileHandle,
        base_shape: Size,
        tile_size: Size,
        jpeg: Jpeg,
        frame_cache: int = 1
    ):
        """Metaclass for a tiled ndpi page.

        Parameters
        ----------
        page: TiffPage
            TiffPage defining the page.
        fh: FileHandle
            Filehandler to read data from.
        base_shape: Size
            Size of base level in pyramid.
        tile_size: Size
            Requested tile size.
        jpeg: Jpeg
            Jpeg instance to use.
        frame_cache: int:
            Number of read frames to cache.
        """
        super().__init__(page, fh, jpeg)
        self._base_shape = base_shape
        self._tile_size = tile_size
        self._file_frame_size = self._get_file_frame_size()
        self._frame_size = Size.max(self.tile_size, self._file_frame_size)
        self._pyramid_index = self._calculate_pyramidal_index(self._base_shape)
        self._frame_cache = NdpiCache(frame_cache)
        self._headers: Dict[Size, bytes] = {}

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._page}, {self._fh}, "
            f"{self._base_shape}, {self.tile_size}, {self._jpeg}, "
            f"{self._frame_cache.size})"
        )

    @property
    def suggested_minimum_chunk_size(self) -> int:
        return max(self._frame_size.width // self._tile_size.width, 1)

    @property
    def tile_size(self) -> Size:
        """The size of the tiles to generate."""
        return self._tile_size

    @property
    def frame_size(self) -> Size:
        """The default read size used for reading frames."""
        return self._frame_size

    @property
    def mcu(self) -> Size:
        subsampling: Optional[Tuple[int, int]] = self._page.subsampling
        if subsampling is None or subsampling == (1, 1):
            return Size(8, 8)
        elif subsampling == (2, 1):
            return Size(16, 8)
        elif subsampling == (2, 2):
            return Size(16, 16)
        raise ValueError(f"Unkown subsampling {subsampling}")

    @abstractmethod
    def _read_extended_frame(
        self,
        position: Point,
        frame_size: Size
    ) -> bytes:
        """Read a frame of size frame_size covering position."""
        raise NotImplementedError

    @abstractmethod
    def _get_file_frame_size(self) -> Size:
        """Return size of single frame/stripe in file."""
        raise NotImplementedError

    @abstractmethod
    def _get_frame_size_for_tile(self, tile_position: Point) -> Size:
        """Return frame size used for creating tile at tile position."""
        raise NotImplementedError

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
        return self.get_tiles([tile_position])[0]

    def get_tiles(
        self,
        tile_positions: Sequence[Tuple[int, int]]
    ) -> List[bytes]:
        """Return list of image bytes for tile positions.

        Parameters
        ----------
        tile_positions: Sequence[Tuple[int, int]]
            Tile positions to get.

        Returns
        ----------
        List[bytes]
            List of tile bytes.
        """
        frame_jobs = self._sort_into_frame_jobs(tile_positions)
        return [
            tile
            for frame_job in frame_jobs
            for tile in self._create_tiles(frame_job).values()
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
        frame_jobs = self._sort_into_frame_jobs(tile_positions)
        return [
            self._jpeg.decode(tile)
            for frame_job in frame_jobs
            for tile in self._create_tiles(frame_job).values()
        ]

    def _create_tiles(
        self,
        frame_job: NdpiFrameJob
    ) -> Dict[Point, bytes]:
        """Return tiles defined by frame job. Read frames are cached by
        frame position.

        Parameters
        ----------
        frame_job: NdpiFrameJob
            Tile job containing tiles that should be created.

        Returns
        ----------
        Dict[Point, bytes]:
            Created tiles ordered by tile coordinate.
        """
        if frame_job.position in self._frame_cache:
            frame = self._frame_cache[frame_job.position]
        else:
            frame = self._read_extended_frame(
                frame_job.position,
                frame_job.frame_size
            )
            self._frame_cache[frame_job.position] = frame
        tiles = self._crop_to_tiles(frame_job, frame)
        return tiles

    def _crop_to_tiles(
        self,
        frame_job: NdpiFrameJob,
        frame: bytes
    ) -> Dict[Point, bytes]:
        """Crop jpeg data to tiles.

        Parameters
        ----------
        frame_job: NdpiFrameJob
            Frame job defining the tiles to produce by cropping jpeg data.
        frame: bytes
            Data to crop from.

        Returns
        ----------
        Dict[Point, bytes]:
            Created tiles ordered by tile coordinate.
        """
        try:
            tiles: List[bytes] = self._jpeg.crop_multiple(
                frame,
                frame_job.crop_parameters
            )
        except JpegCropError:
            raise ValueError(
                f'Failed to crop at position {frame_job.position} with '
                f'parameters {frame_job.crop_parameters}. '
                'This might be due using libjpeg-turbo < 2.1.'
            )
        return {
            tile.position: tiles[i]
            for i, tile in enumerate(frame_job.tiles)
        }

    def _sort_into_frame_jobs(
        self,
        tile_positions: Sequence[Tuple[int, int]]
    ) -> List[NdpiFrameJob]:
        """Sorts tile positions into frame jobs (i.e. from the same frame.)

        Parameters
        ----------
        tile_positions: Sequence[Point]
            List of position to sort.

        Returns
        ----------
        List[NdpiFrameJob]
            List of created frame jobs.

        """
        frame_jobs: Dict[Point, NdpiFrameJob] = {}
        for tile_position in tile_positions:
            tile_point = Point.from_tuple(tile_position)
            if not self._check_if_tile_inside_image(tile_point):
                raise ValueError(
                    f"Tile {tile_point} is outside "
                    f"tiled size {self.tiled_size}"
                )
            frame_size = self._get_frame_size_for_tile(tile_point)
            tile = NdpiTile(tile_point, self.tile_size, frame_size)
            if tile.frame_position in frame_jobs:
                frame_jobs[tile.frame_position].append(tile)
            else:
                frame_jobs[tile.frame_position] = NdpiFrameJob(tile)
        return list(frame_jobs.values())


class NdpiOneFramePage(NdpiTiledPage):
    """Class for a ndpi page containing only one frame that should be tiled.
    The frame can be of any size (smaller or larger than the wanted tile size).
    The frame is padded to an even multipe of tile size.
    """

    def _get_file_frame_size(self) -> Size:
        """Return size of the single frame in file. For single framed page
        this is equal to the level size.

        Returns
        ----------
        Size
            The size of frame in the file.
        """
        return self.image_size

    def _get_frame_size_for_tile(self, tile_position: Point) -> Size:
        """Return read frame size for tile position. For single frame page
        the read frame size is the image size rounded up to the closest tile
        size.

        Returns
        ----------
        Size
            The read frame size.
        """
        return ((self.frame_size) // self.tile_size + 1) * self.tile_size

    def _read_extended_frame(
        self,
        position: Point,
        frame_size: Size
    ) -> bytes:
        """Return padded image covering tile coordinate as valid jpeg bytes.

        Parameters
        ----------
        frame_position: Point
            Upper left tile position that should be covered by the frame.
        frame_size: Size
            Size of the frame to read.

        Returns
        ----------
        bytes
            Frame
        """
        if position != Point(0, 0):
            raise ValueError("Frame position not (0, 0) for one frame level.")
        frame = self._read_frame(0)
        if (
            self.image_size.width % self.mcu.width != 0
            or self.image_size.height % self.mcu.height != 0
        ):
            # Extend to whole MCUs
            even_size = Size.ceil_div(self.image_size, self.mcu) * self.mcu
            frame = Jpeg.manipulate_header(frame, even_size)
        # Use crop_multiple as it allows extending frame
        tile: bytes = self._jpeg.crop_multiple(
            frame,
            [(0, 0, frame_size.width, frame_size.height)]
        )[0]
        return tile


class NdpiStripedPage(NdpiTiledPage):
    """Class for a ndpi page containing stripes. Frames are constructed by
    concatenating multiple stripes, and from the frame one or more tiles can be
    produced by lossless cropping.
    """
    def __init__(
        self,
        page: TiffPage,
        fh: FileHandle,
        base_shape: Size,
        tile_size: Size,
        jpeg: Jpeg,
        frame_cache: int = 1
    ):
        """Ndpi page with striped image data.

        Parameters
        ----------
        page: TiffPage
            TiffPage defining the page.
        fh: FileHandle
            Filehandler to read data from.
        base_shape: Size
            Size of base level in pyramid.
        tile_size: Size
            Requested tile size.
        jpeg: Jpeg
            Jpeg instance to use.
        frame_cache: int:
            Number of read frames to cache.
        """
        super().__init__(page, fh, base_shape, tile_size, jpeg, frame_cache)
        self._striped_size = Size(self.page.chunked[1], self.page.chunked[0])

    @property
    def stripe_size(self) -> Size:
        """Size of stripes."""
        return self._file_frame_size

    @property
    def striped_size(self) -> Size:
        """Number of stripes in columns and rows."""
        return self._striped_size

    def _get_file_frame_size(self) -> Size:
        """Return size of stripes in file. For striped levels this is parsed
        from the jpeg header.

        Returns
        ----------
        Size
            The size of stripes in the file.
        """
        stripe_height, stripe_width, _ = self.page.chunks
        return Size(stripe_width, stripe_height)

    def _is_partial_frame(self, tile_position: Point) -> Tuple[bool, bool]:
        """Return a tuple of bools, that are true if tile position is at the
        edge of the image in x or y.

        Parameters
        ----------
        tile_position: int
            Tile position (x or y) to check.

        Returns
        ----------
        Tuple[bool, bool]
            Tuple that is True if tile position x or y is at edge of image.
        """
        partial_x = (
            tile_position.x == (self.tiled_size.width - 1) and
            self.stripe_size.width < self.tile_size.width
        )
        partial_y = (
            tile_position.y == (self.tiled_size.height - 1) and
            self.stripe_size.height < self.tile_size.height
        )
        return partial_x, partial_y

    def _get_frame_size_for_tile(self, tile_position: Point) -> Size:
        """Return frame size used for creating tile at tile position.
        If tile is an edge tile, ensure that the frame does not extend beyond
        the image limits.

        Parameters
        ----------
        tile_position: Point
            Tile position for frame size calculation.

        Returns
        ----------
        Size
            Frame size to be used at tile position.
        """
        is_partial_x, is_partial_y = self._is_partial_frame(tile_position)
        if is_partial_x:
            width = (
                self.stripe_size.width * self.striped_size.width
                - tile_position.x * self.tile_size.width
            )
        else:
            width = self.frame_size.width

        if is_partial_y:
            height = (
                self.stripe_size.height * self.striped_size.height
                - tile_position.y * self.tile_size.height
            )
        else:
            height = self.frame_size.height
        return Size(width, height)

    def _read_extended_frame(
        self,
        position: Point,
        frame_size: Size
    ) -> bytes:
        """Return extended frame of frame size starting at frame position.
        Returned frame is jpeg bytes including header with correct image size.
        Original restart markers are updated to get the proper incrementation.
        End of image tag is appended.

        Parameters
        ----------
        position: Point
            Upper left tile position that should be covered by the frame.
        frame_size: Size
            Size of the frame to get.

        Returns
        ----------
        bytes
            Concatenated frame as jpeg bytes.
        """
        if frame_size in self._headers:
            header = self._headers[frame_size]
        else:
            header = self._jpeg.manipulate_header(
                self.page.jpegheader,
                frame_size
            )
            self._headers[frame_size] = header

        stripe_region = Region(
            (position * self.tile_size) // self.stripe_size,
            Size.max(frame_size // self.stripe_size, Size(1, 1))
        )
        indices = [
            self._get_stripe_position_to_index(stripe_coordinate)
            for stripe_coordinate in stripe_region.iterate_all()
        ]
        frame = self._jpeg.concatenate_fragments(
            (self._read_frame(index) for index in indices),
            header
        )
        return frame

    def _get_stripe_position_to_index(self, position: Point) -> int:
        """Return stripe index from position.

        Parameters
        ----------
        position: Point
            position of stripe to get index for.

        Returns
        ----------
        int
            Stripe index.
        """
        return position.x + position.y * self.striped_size.width


class NdpiTiler(Tiler):
    def __init__(
        self,
        filepath: Union[str, Path],
        tile_size: int,
        turbo_path: Optional[Union[str, Path]] = None
    ):
        """Tiler for ndpi file, with functions to produce tiles of specified
        size.

        Parameters
        ----------
        filepath: Union[str, Path]
            Filepath to a ndpi TiffFile.
        tile_size: int
            Tile size to cache and produce. Must be multiple of 8 and will be
            adjusted to be an even multipler or divider of the smallest strip
            width in the file.
        turbo_path: Optional[Union[str, Path]] = None
            Path to turbojpeg (dll or so).

        """
        super().__init__(Path(filepath))

        self._fh = self._tiff_file.filehandle
        self._tile_size = Size(tile_size, tile_size)
        self._tile_size = self._adjust_tile_size(
            tile_size,
            self._get_smallest_stripe_width()
        )
        if self.tile_size.width % 8 != 0 or self.tile_size.height % 8 != 0:
            raise ValueError(f"Tile size {self.tile_size} not divisable by 8")
        self._turbo_path = turbo_path
        self._jpeg = Jpeg(self._turbo_path)

        self._level_series_index = 0
        for series_index, series in enumerate(self.series):
            if series.name == 'Label':
                self._label_series_index = series_index
            elif series.name == 'Macro':
                self._overview_series_index = series_index
        self._pages: Dict[Tuple[int, int, int], NdpiPage] = {}

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._tiff_file.filename}, "
            f"{self.tile_size.to_tuple}, "
            f"{self._turbo_path})"
        )

    def __str__(self) -> str:
        return f"{type(self).__name__} of Tifffile {self._tiff_file}"

    @property
    def tile_size(self) -> Size:
        """The size of the tiles to generate."""
        return self._tile_size

    def get_page(
        self,
        series: int,
        level: int,
        page: int
    ) -> NdpiPage:
        """Return NdpiPage for series, level, page. NdpiPages holds a cache, so
        store created pages.
        """
        if not (series, level, page) in self._pages:
            ndpi_page = self._create_page(series, level, page)
            self._pages[series, level, page] = ndpi_page
        return self._pages[series, level, page]

    @staticmethod
    def _adjust_tile_size(
        requested_tile_width: int,
        smallest_stripe_width: Optional[int] = None
    ) -> Size:
        """Return adjusted tile size. If file contains striped pages the
        tile size must be an n * smallest stripe width in the file, where n
        is the closest square factor of the ratio between requested tile width
        and smallest stripe width.

        Parameters
        ----------
        requested_tile_width: int
            Requested tile width.
        smallest_stripe_width: Optional[int] = None
            Smallest stripe width in file.

        Returns
        ----------
        Size
            Adjusted tile size.
        """
        if (
            smallest_stripe_width is None or
            smallest_stripe_width == requested_tile_width
        ):
            # No striped pages or requested is equald to smallest
            return Size(requested_tile_width, requested_tile_width)

        if requested_tile_width > smallest_stripe_width:
            factor = requested_tile_width / smallest_stripe_width
        else:
            factor = smallest_stripe_width / requested_tile_width
        # Factor should be a square number (in the series 2^n)
        factor_2 = pow(2, round(math.log2(factor)))
        adjusted_width = factor_2 * smallest_stripe_width
        return Size(adjusted_width, adjusted_width)

    def _get_smallest_stripe_width(self) -> Optional[int]:
        """Return smallest stripe width in file, or None if no page in the
        file is striped.

        Returns
        ----------
        Optional[int]
            The smallest stripe width in the file, or None if no page in the
            file is striped.
        """
        smallest_stripe_width: Optional[int] = None
        for page in self._tiff_file.pages:
            stripe_width = page.chunks[1]
            if (
                page.is_tiled and
                (
                    smallest_stripe_width is None or
                    smallest_stripe_width > stripe_width
                )
            ):
                smallest_stripe_width = stripe_width
        return smallest_stripe_width

    def _create_page(
        self,
        series: int,
        level: int,
        page: int,
    ) -> NdpiPage:
        """Create a new page from TiffPage.

        Parameters
        ----------
        series: int
            Series of page.
        level: int
            Level of page.
        page: int
            Page to use.

        Returns
        ----------
        NdpiLevel
            Created level.
        """
        tiff_page: TiffPage = (
            self._tiff_file.series[series].levels[level].pages[page]
        )
        if tiff_page.is_tiled:  # Striped ndpi page
            return NdpiStripedPage(
                tiff_page,
                self._fh,
                self.base_size,
                self.tile_size,
                self._jpeg
            )
        if series == self._level_series_index:  # Single frame, force tiling
            return NdpiOneFramePage(
                tiff_page,
                self._fh,
                self.base_size,
                self.tile_size,
                self._jpeg
            )
        return NdpiPage(
            tiff_page,
            self._fh,
            self._jpeg
        )  # Single frame, do not tile
