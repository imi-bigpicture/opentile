import io
import math
import struct
import threading
from abc import ABCMeta, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from struct import unpack
from typing import Dict, Generator, List, Optional, Tuple

from PIL import Image
from tifffile import FileHandle, TiffPage
from tifffile.tifffile import TiffPageSeries
from turbojpeg import TurboJPEG


@dataclass
class Size:
    width: int
    height: int

    def __str__(self):
        return f'{self.width}x{self.height}'

    def __add__(self, value):
        if isinstance(value, int):
            return Size(self.width + value, self.height + value)
        return NotImplemented

    def __mul__(self, factor):
        if isinstance(factor, (int, float)):
            return Size(int(factor*self.width), int(factor*self.height))
        elif isinstance(factor, Size):
            return Size(factor.width*self.width, factor.height*self.height)
        elif isinstance(factor, Point):
            return Size(factor.x*self.width, factor.y*self.height)
        return NotImplemented

    def __floordiv__(self, divider):
        if isinstance(divider, Size):
            return Size(
                int(self.width/divider.width),
                int(self.height/divider.height)
            )
        return NotImplemented

    def __truediv__(self, divider):
        if isinstance(divider, Size):
            return Size(
                self.width/divider.width,
                self.height/divider.height
            )
        return NotImplemented

    @staticmethod
    def max(size_1: 'Size', size_2: 'Size'):
        return Size(
            width=max(size_1.width, size_2.width),
            height=max(size_1.height, size_2.height)
        )

    def ceil(self) -> 'Size':
        return Size(
            width=int(math.ceil(self.width)),
            height=int(math.ceil(self.height))
        )


@dataclass
class Point:
    x: int
    y: int

    def __str__(self):
        return f'{self.x}, {self.y}'

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __add__(self, value):
        if isinstance(value, Size):
            return Point(self.x + value.width, self.y + value.height)
        elif isinstance(value, Point):
            return Point(self.x + value.x, self.y + value.y)
        return NotImplemented

    def __sub__(self, value):
        if isinstance(value, Point):
            return Point(self.x - value.x, self.y - value.y)
        return NotImplemented

    def __mul__(self, factor):
        if isinstance(factor, (int, float)):
            return Point(int(factor*self.x), int(factor*self.y))
        elif isinstance(factor, Size):
            return Point(factor.width*self.x, factor.height*self.y)
        elif isinstance(factor, Point):
            return Point(factor.x*self.x, factor.y*self.y)
        return NotImplemented

    def __floordiv__(self, divider):
        if isinstance(divider, Point):
            return Point(int(self.x/divider.x), int(self.y/divider.y))
        elif isinstance(divider, Size):
            return Point(int(self.x/divider.width), int(self.y/divider.height))
        return NotImplemented

    def __mod__(self, divider):
        if isinstance(divider, Size):
            return Point(
                int(self.x % divider.width),
                int(self.y % divider.height)
            )
        elif isinstance(divider, Point):
            return Point(
                int(self.x % divider.x),
                int(self.y % divider.y)
            )
        return NotImplemented


@dataclass
class Region:
    position: Point
    size: Size

    def __str__(self):
        return f'from {self.start} to {self.end}'

    @property
    def start(self) -> Point:
        return self.position

    @property
    def end(self) -> Point:
        end: Point = self.position + self.size
        return end

    def iterate_all(self, include_end=False) -> Generator[Point, None, None]:
        offset = 1 if include_end else 0
        return (
            Point(x, y)
            for y in range(self.start.y, self.end.y + offset)
            for x in range(self.start.x, self.end.x + offset)
        )


class Tags:
    TAG = 0xFF
    TAGS = {
        'start of image': 0xD8,
        'application default header': 0xE0,
        'quantization table': 0xDB,
        'start of frame': 0xC0,
        'huffman table': 0xC4,
        'start of scan': 0xDA,
        'end of image': 0xD9,
        'restart interval': 0xDD,
        'restart mark': 0xD0
    }

    @classmethod
    def start_of_frame(cls) -> bytes:
        """Return bytes representing a start of frame tag."""
        return bytes([cls.TAG, cls.TAGS['start of frame']])

    @classmethod
    def end_of_image(cls) -> bytes:
        """Return bytes representing a end of image tag."""
        return bytes([cls.TAG, cls.TAGS['end of image']])

    @classmethod
    def restart_mark(cls, index: int) -> bytes:
        """Return bytes representing a restart marker of index (0-7), without
        the prefixing tag (0xFF)."""
        return bytes([cls.TAGS['restart mark'] + index % 8])


class NdpiCache():
    """Cache for bytes ordered by tile position. Oldest entry is removed when
    size of conent is above set size."""
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

    def keys(self) -> List[Point]:
        """Returns keys in cache.

        Returns
        ----------
        List[Point]
            Keys in cache.

        """
        return self._content.keys()

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


class NdpiFileHandle:
    """A lockable file handle for reading stripes."""
    def __init__(self, fh: FileHandle):
        self._fh = fh
        self._lock = threading.Lock()

    def read(self, offset: int, bytecount: int) -> bytes:
        """Return bytes from filehandle.

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
            data = self._fh.read(bytecount)
        return data


class NdpiTile:
    """Defines a tile by position and coordinates and size for cropping out
    out frame."""
    def __init__(
        self,
        tile_position: Point,
        tile_size: Size,
        frame_size: Size
    ) -> None:
        """Create a ndpi tile and calculate cropping parameters.

        Parameters
        ----------
        tile_position: Point
            Tile position.
        tile_size
            Tile size.
        frame_size
            Frame size.

        """
        self._tile_position = tile_position
        self._tile_size = tile_size
        self._frame_size = frame_size

        self._tiles_per_frame = Size.max(
            self._frame_size // self._tile_size,
            Size(1, 1)
        )

        frame_position = self._map_tile_to_frame(tile_position)
        self._origin = self._get_origin_tile(tile_position)
        self._left = frame_position.x
        self._top = frame_position.y
        self._width = tile_size.width
        self._height = tile_size.height

    def __eq__(self, other: object) -> bool:
        if isinstance(other, NdpiTile):
            return (
                self._tile_position == other._tile_position and
                self._tile_size == other._tile_size and
                self._frame_size == other._frame_size
            )
        return NotImplemented

    def __repr__(self) -> str:
        return (
            f"NdpiTile({self.position}, {self._tile_size}, "
            f"{self._frame_size})"
        )

    @property
    def position(self) -> Point:
        return self._tile_position

    @property
    def origin(self) -> Point:
        return self._origin

    @property
    def left(self) -> int:
        return self._left

    @property
    def top(self) -> int:
        return self._top

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def _map_tile_to_frame(self, tile_position: Point) -> Point:
        """Map a tile position to position in frame.

        Parameters
        ----------
        tile_position: Point
            Tile position that should be map to frame.

        Returns
        ----------
        Point
            Frame position for tile.
        """
        return (tile_position * self._tile_size) % self._frame_size

    def _get_origin_tile(self, tile_position: Point) -> Point:
        """Return origin tile position for tile. The origin tile is the first
        tile (upper left) of the frame that containts the tile.

        Parameters
        ----------
        tile_position: Point
            Tile to get the origin for.

        Returns
        ----------
        Point
            The origin tile position for tile.

        """
        return (tile_position // self._tiles_per_frame) * self._tiles_per_frame


class NdpiTileJob:
    """A list of tiles for a thread to parse. Tiles need to have the same
    origin."""
    def __init__(
        self,
        tiles: List[NdpiTile]
    ) -> None:
        """Create a tile job from given tile.

        Parameters
        ----------
        tile: NdpiTile
            Tile to base the tile job on.

        """
        self._origin = tiles[0].origin
        self._tiles: List[NdpiTile] = []
        for tile in tiles:
            self.append(tile)

    def __repr__(self) -> str:
        return f"NdpiTileJob({self.tiles})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, NdpiTileJob):
            return self.tiles == other.tiles
        return NotImplemented

    @property
    def origin(self) -> Point:
        """The origin position of the tile job."""
        return self._origin

    def append(self, tile: NdpiTile) -> None:
        """Add a tile to the tile job."""
        if tile.origin != self.origin:
            raise ValueError(f"{tile} does not match {self} origin")
        self._tiles.append(tile)

    @property
    def tiles(self) -> List[NdpiTile]:
        return self._tiles


class NdpiLevel(metaclass=ABCMeta):
    """Metaclass for a ndpi level."""
    def __init__(
        self,
        page: TiffPage,
        fh: NdpiFileHandle,
        tile_size: Size,
        jpeg: TurboJPEG
    ):
        """Metaclass for a ndpi level, should not be used.

        Parameters
        ----------
        page: TiffPage
            TiffPage defining the level.
        fh: NdpiFileHandle
            Filehandler to read data from.
        tile_size:
            Requested tile size.
        jpeg: TurboJpeg
            TurboJpeg instance to use.

        """
        self._page = page
        self._fh = fh
        self._tile_size = tile_size
        self._jpeg = jpeg

        self._size_in_file = self._get_size_in_file()

        self.tile_cache = NdpiCache(10)
        self.frame_cache = NdpiCache(10)

    @property
    def tile_size(self) -> Size:
        """The size of the tiles to generate."""
        return self._tile_size

    @cached_property
    def level_size(self) -> Size:
        """The size of the level."""
        return Size(self._page.shape[1], self._page.shape[0])

    @cached_property
    def frame_size(self) -> Size:
        """The size of the frames to use for creating tiles."""
        return self._get_frame_size(self._size_in_file, self.tile_size)

    @cached_property
    def tiled_size(self) -> Size:
        """The level size when tiled (columns and rows of tiles)."""
        # This rounds down, correct should round up. But we cant handle
        # not complete tiles
        return self.level_size // self.tile_size
        # return (self.level_size / self.tile_size).ceil()

    @cached_property
    def tiles_per_frame(self) -> Size:
        """The number of tiles created when parsing one frame."""
        return Size.max(
            self.frame_size // self.tile_size,
            Size(1, 1)
        )

    @abstractmethod
    def _get_frame(self, tile_position: Point) -> bytes:
        """Return frame for creating tiles."""
        raise NotImplementedError

    @abstractmethod
    def _get_size_in_file(self) -> Size:
        """Return size of the single frame in file."""
        raise NotImplementedError

    @abstractmethod
    def _get_frame_size(self, size_in_file: Size, tile_size: Size) -> Size:
        """Return frame size used for creating tiles."""
        raise NotImplementedError

    def create_batches(
        self,
        number_of_tiles: int
    ) -> Generator[List[Point], None, None]:
        """Divide the tiles covering the level into batches with maximum
        number_of_tiles in each batch.

        Parameters
        ----------
        number_of_tiles: int
            Number of tiles in each batch

        Returns
        ----------
        Generator[List[Point], None, None]:
            Generator with list of points for each batch
        """
        level_region = Region(Point(0, 0), self.tiled_size)
        tiles = list(level_region.iterate_all())
        for index in range(0, len(tiles), number_of_tiles):
            yield tiles[index:index + number_of_tiles]

    def get_tile(self, tile_position: Point) -> bytes:
        """Return tile for tile position. Caches created frames and tiles.

        Parameters
        ----------
        tile_position: Point
            Tile osition to get.

        Returns
        ----------
        bytes
            Produced tile at position.
        """
        # Check if tile not in cached
        if tile_position not in self.tile_cache.keys():
            # Create a tile job
            tile = NdpiTile(
                tile_position,
                self.tile_size,
                self.frame_size
            )
            tile_job = NdpiTileJob([tile])
            # Create new tiles
            new_tiles = self._create_tiles(tile_job)
            # Add to tile cache
            self.tile_cache.update(new_tiles)
        return self.tile_cache[tile_position]

    def get_tiles(self, tile_positions: List[Point]) -> bytes:
        """Return tiles for tile positions. Sorts the requested tile positions
        into tile jobs and uses a pool of threads to parse tile jobs. The
        results are concatenated. Frames and tiles are not cached.

        Parameters
        ----------
        tile_positions: List[Point]
            List of position to get.

        Returns
        ----------
        bytes
            Concatentated tiles from positions.
        """
        tile_jobs = self._sort_into_tile_jobs(tile_positions)
        with ThreadPoolExecutor() as pool:
            def thread(tile_job: NdpiTileJob) -> bytes:
                return b"".join(self._create_tiles(tile_job).values())
            result = pool.map(thread, tile_jobs)
            return b"".join(result)

    def _create_tiles(
        self,
        tile_job: NdpiTileJob
    ) -> Dict[Point, bytes]:
        """Return tiles created by parsing frame needed for requested tile.
        Additional tiles than the requested tile is created if the
        level is striped and the stripes span multiple tiles.

        Parameters
        ----------
        tile_job: NdpiTileJob
            Tile job containing tiles that should be created.

        Returns
        ----------
        Dict[Point, bytes]:
            Created tiles ordered by tile coordiante.
        """
        try:
            frame = self.frame_cache[tile_job.origin]
        except KeyError:
            frame = self._get_frame(tile_job.origin)
            self.frame_cache[tile_job.origin] = frame
        tiles = self._crop_to_tiles(tile_job, frame)
        return tiles

    def _crop_to_tiles(
        self,
        tile_job: NdpiTileJob,
        frame: bytes
    ) -> Dict[Point, bytes]:
        """Crop jpeg data to tiles.

        Parameters
        ----------
        tile_job: NdpiTileJob
            Tile job defining the tiles to produce by cropping jpeg data.
        frame: bytes
            Data to crop from.

        Returns
        ----------
        Dict[Point, bytes]:
            Created tiles ordered by tile coordiante.
        """
        # Each tile currently requires one complete calculation
        # This could thus be faster if we could reuse some calculations between
        # tiles. For this a custom libjpeg wrapper is needed (and some magic
        # mcu handling.)
        return {
            tile.position: self._jpeg.crop(
                frame,
                tile.left,
                tile.top,
                tile.width,
                tile.height
            )
            for tile in tile_job.tiles
        }

    def _sort_into_tile_jobs(
        self,
        tile_positions: List[Point]
    ) -> List[NdpiTileJob]:
        """Sorts tile positions into tile jobs with commmon tile origin (i.e.
        from the same frame.)

        Parameters
        ----------
        tile_positions: List[Point]
            List of position to sort.

        Returns
        ----------
        List[NdpiTileJob]
            List of created tile jobs.

        """
        tile_jobs: Dict[Point, NdpiTileJob] = {}
        for tile_position in tile_positions:
            tile = NdpiTile(tile_position, self.tile_size, self.frame_size)
            try:
                tile_jobs[tile.origin].append(tile)
            except KeyError:
                tile_jobs[tile.origin] = NdpiTileJob([tile])
        return list(tile_jobs.values())

    def _read(self, index: int) -> bytes:
        """Read frame bytes at index.

        Parameters
        ----------
        index: int
            Index of frame to read.

        Returns
        ----------
        bytes
            Frame bytes.
        """
        offset = self._page.dataoffsets[index]
        bytecount = self._page.databytecounts[index]
        return self._fh.read(offset, bytecount)


class NdpiOneFrameLevel(NdpiLevel):
    """Class for a ndpi level containing only one frame. The frame can be
    of any size (smaller or larger than the wanted tile size). The
    frame is padded to an even multipe of tile size. This is currently
    not lossless.
    """

    def _get_size_in_file(self) -> Size:
        """Return size of the single frame in file.

        Returns
        ----------
        Size
            The size of frame in the file.
        """
        return Size(self._page.shape[1], self._page.shape[0])

    def _get_frame_size(self, size_in_file: Size, tile_size: Size) -> Size:
        """Return frame size used for creating tiles.

        Parameters
        ----------
        size_in_file: Size
            Size of frame in file.
        tile_size: Size
            Requested tile size

        Returns
        ----------
        Size
            The size of frames to create when creating tiles.
        """

        return (size_in_file // tile_size + 1) * tile_size

    def _get_frame(self, tile_position: Point) -> bytes:
        """Return padded image covering tile coorindate as valid jpeg bytes.
        Includes header with the correct image size. Original restart markers
        are updated to get the proper incrementation. End of image tag is
        appended end.

        Parameters
        ----------
        tile_position: Point
            Tile position that should be covered by the stripe region.

        Returns
        ----------
        bytes
            Stitched image as jpeg bytes.
        """
        # This is not lossless!
        frame = self._read(0)
        frame_image = Image.open(io.BytesIO(frame))
        padded_frame = Image.new(
            'RGB',
            (self.frame_size.width, self.frame_size.height),
            (255, 255, 255)
        )
        padded_frame.paste(frame_image, (0, 0))
        with io.BytesIO() as buffer:
            padded_frame.save(buffer, format='jpeg')
            return buffer.getvalue()


class NdpiStripedLevel(NdpiLevel):
    """Class for a ndpi level containing stripes. A frame is constructed by
    concatenating multiple stripes, and from the frame one or more tiles can be
    produced by cropping. The procedure is lossless. Edge tiles were the tile
    is 'outside' a frame is not handled correctly.
    """

    @property
    def stripe_size(self) -> Size:
        """Size of the stripes."""
        return self._size_in_file

    @cached_property
    def striped_size(self) -> Size:
        """Number of stripes in columns and rows."""
        return Size(self._page.chunked[1], self._page.chunked[0])

    @cached_property
    def header(self) -> bytes:
        """Modified jpeg header for reading frames."""
        return self._update_header(self._page.jpegheader, self.frame_size)

    def _get_size_in_file(self) -> Size:
        """Return size of stripes in file.

        Returns
        ----------
        Size
            The size of stripes in the file.
        """
        (
            stripe_width,
            stripe_height,
            _, _
        ) = self._jpeg.decode_header(self._page.jpegheader)
        return Size(stripe_width, stripe_height)

    def _get_frame_size(self, size_in_file: Size, tile_size: Size) -> Size:
        """Return frame size used for creating tiles.

        Parameters
        ----------
        size_in_file: Size
            Size of stripes in file.
        tile_size: Size
            Requested tile size

        Returns
        ----------
        Size
            The size of frames to create when creating tiles.
        """
        return Size.max(tile_size, size_in_file)

    def _get_frame(self, tile_position: Point) -> bytes:
        """Return concatenated frame covering tile coorindate as valid jpeg
        bytes including header with correct image size. Original restart
        markers are updated to get the proper incrementation. End of image tag
        is appended.

        Parameters
        ----------
        tile_position: Point
            Tile position that should be covered by the stripe region.

        Returns
        ----------
        bytes
            Concatenated frame as jpeg bytes.
        """
        jpeg_data = self.header
        restart_marker_index = 0
        stripe_region = Region(
            (tile_position * self.tile_size) // self.stripe_size,
            Size.max(self.tile_size // self.stripe_size, Size(1, 1))
        )
        for stripe_coordiante in stripe_region.iterate_all():
            jpeg_data += self._get_stripe(stripe_coordiante)[:-1]
            jpeg_data += Tags.restart_mark(restart_marker_index)
            restart_marker_index += 1
        jpeg_data += Tags.end_of_image()
        return jpeg_data

    @staticmethod
    def _find_tag(
        header: bytes,
        tag: bytes
    ) -> Tuple[Optional[int], Optional[int]]:
        """Return first index and length of payload of tag in header.

        Parameters
        ----------
        heaer: bytes
            Header to search.
        tag: bytes
            Tag to search for.

        Returns
        ----------
        Tuple[Optional[int], Optional[int]]:
            Position of tag in header and length of payload.
        """
        index = header.find(tag)
        if index != -1:
            (length, ) = unpack('>H', header[index+2:index+4])
            return index, length
        return None, None

    @classmethod
    def _update_header(
        cls,
        header: bytes,
        size: Size,
    ) -> bytes:
        """Return manipulated header with changed pixel size (width, height).

        Parameters
        ----------
        heaer: bytes
            Header to manipulate.
        size: Size
            Pixel size to insert into header.

        Returns
        ----------
        bytes:
            Manupulated header.
        """
        header = bytearray(header)
        start_of_frame_index, length = cls._find_tag(
            header, Tags.start_of_frame()
        )
        if start_of_frame_index is None:
            raise ValueError("Start of scan tag not found in header")
        size_index = start_of_frame_index+5
        header[size_index:size_index+2] = struct.pack(">H", size.height)
        header[size_index+2:size_index+4] = struct.pack(">H", size.width)

        return bytes(header)

    def _stripe_position_to_index(self, position: Point) -> int:
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

    def _get_stripe(self, position: Point) -> bytes:
        """Return stripe bytes for stripe at point.

        Parameters
        ----------
        position: Point
            position of stripe to get.

        Returns
        ----------
        bytes
            Stripe as bytes.
        """
        index = self._stripe_position_to_index(position)
        try:
            data = self._read(index)
        except IndexError:
            raise IndexError(f"error reading tile {position} with index {index}")
        return data


class NdpiTiler:
    def __init__(
        self,
        tiff_series: TiffPageSeries,
        fh: NdpiFileHandle,
        tile_size: Tuple[int, int],
        turbo_path: Path = None
    ):
        """Cache for ndpi stripes, with functions to produce tiles of specified
        size.

        Parameters
        ----------
        tif: TiffFile
            Tiff file
        fh: NdpiFileHandle
            File handle to stripe data.
        series: int
            Series in tiff file
        tile_size: Tuple[int, int]
            Tile size to cache and produce. Must be multiple of 8.
        turbo_path: Path
            Path to turbojpeg (dll or so).

        """

        self._fh = fh
        self._tile_size = Size(*tile_size)
        self._tiff_series: TiffPageSeries = tiff_series
        if self.tile_size.width % 8 != 0 or self.tile_size.height % 8 != 0:
            raise ValueError(f"Tile size {self.tile_size} not divisable by 8")

        self.jpeg = TurboJPEG(turbo_path)
        self._levels: Dict[int, NdpiLevel] = {}

    @property
    def tile_size(self) -> Size:
        """The size of the tiles to generate."""
        return self._tile_size

    def get_tile(
        self,
        level: int,
        tile_position: Tuple[int, int]
    ) -> bytes:
        """Return tile for tile position x and y. If stripes for the tile
        is not cached, read them from disk and parse the jpeg data.

        Parameters
        ----------
        level: int
            Level of tile to get.
        tile_position: Tuple[int, int]
            Position of tile to get.

        Returns
        ----------
        bytes
            Produced tile at position, wrapped in header.
        """
        ndpi_level = self.get_level(level)
        return ndpi_level.get_tile(Point(*tile_position))

    def get_level(self, level: int) -> NdpiLevel:
        """Return level. Create level if not found.

        Parameters
        ----------
        level: int
            Level to get.

        Returns
        ----------
        NdpiLevel
            Requested level.
        """
        try:
            ndpi_level = self._levels[level]
        except KeyError:
            ndpi_level = self._create_level(level)
            self._levels[level] = ndpi_level
        return ndpi_level

    def _create_level(self, level: int) -> NdpiLevel:
        """Create a new level.

        Parameters
        ----------
        level: int
            Level to add

        Returns
        ----------
        NdpiLevel
            Created level.
        """
        page: TiffPage = self._tiff_series.levels[level].pages[0]
        if page.is_tiled:
            return NdpiStripedLevel(page, self._fh, self.tile_size, self.jpeg)
        return NdpiOneFrameLevel(page, self._fh, self.tile_size)
