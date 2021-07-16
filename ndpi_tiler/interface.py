import io
import struct
import threading
from abc import ABCMeta, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
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


@dataclass
class Point:
    x: int
    y: int

    def __str__(self):
        return f'{self.x},{self.y}'

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


class NdpiFileHandle:
    """A lockable file handle for reading stripes."""
    def __init__(self, fh: FileHandle):
        self._fh = fh
        self._lock = threading.Lock()

    def read(self, offset: int, bytecount: int) -> bytes:
        """Return bytes.

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


@dataclass
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
        frame_position = self._map_tile_to_frame(tile_position)
        self._origin = self._get_origin_tile(tile_position)
        self._left = frame_position.x
        self._top = frame_position.y
        self._width = tile_size.width
        self._height = tile_size.height

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
        ratio = self._frame_size / self._tile_size
        return tile_position - (tile_position % ratio)


@dataclass
class NdpiTileJob:
    """A list of tiles for a thread to parse. Tiles need to have the same
    origin."""
    def __init__(
        self,
        tile: NdpiTile
    ) -> None:
        """Create a tile job from given tile.

        Parameters
        ----------
        tile: NdpiTile
            Tile to base the tile job on.

        """
        self._origin = tile.origin
        self._tiles = [tile]

    @property
    def origin(self) -> Point:
        """The origin position of the tile job."""
        return self._origin

    def append(self, tile: NdpiTile) -> None:
        """Add a tile to the tile job."""
        if tile.origin != self.origin:
            raise ValueError(f"{tile} does not match {self} origin")
        self._tiles.append(tile)

    def __len__(self) -> int:
        """The number of tiles in the job."""
        return len(self._tiles)

    def __iter__(self) -> NdpiTile:
        """Iterator over the tile job yielding tiles."""
        for elem in self._tiles:
            yield elem


class NdpiLevel(metaclass=ABCMeta):
    """Metaclass for a ndpi level."""
    def __init__(
        self,
        page: TiffPage,
        fh: NdpiFileHandle,
        tile_size: Size,
        frame_size: Size,
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
        frame_size:
            Specified frame size.
        jpeg: TurboJpeg
            TurboJpeg instance to use.

        """
        self._page = page
        self._fh = fh
        self._tile_size = tile_size
        self._frame_size = frame_size
        self._jpeg = jpeg
        self._framed_size = Size(page.chunked[1], page.chunked[0])
        level_size = self.frame_size * self.framed_size
        self._tiled_size = level_size // tile_size
        self._tiles_per_frame = Size.max(
            self.frame_size // self.tile_size,
            Size(1, 1)
        )
        read_size = Size.max(
            self._tile_size,
            self._frame_size
            )
        self._header = self._update_header(self._page.jpegheader, read_size)
        self.tile_cache: Dict[Point, bytes] = {}
        self.frame_cache: Dict[Point, bytes] = {}

    @property
    def frame_size(self) -> Size:
        """The size of the stripes in the level."""
        return self._frame_size

    @property
    def framed_size(self) -> Size:
        """The level size when striped (columns and rows of stripes)."""
        return self._framed_size

    @property
    def tiled_size(self) -> Size:
        """The level size when tiled (coluns and rows of tiles)."""
        return self._tiled_size

    @property
    def tile_size(self) -> Size:
        """The size of the tiles to generate."""
        return self._tile_size

    @property
    def tiles_per_frame(self) -> Size:
        """The number of tiles created when parsing one frame."""
        return Size.max(
            self.frame_size // self.tile_size,
            Size(1, 1)
        )

    @abstractmethod
    def _get_frame(
        self,
        tile_position: Point
    ) -> bytes:
        raise NotImplementedError

    def create_batches(
        self,
        number_of_tiles: int
    ) -> Generator[List[Point], None, None]:
        level_region = Region(Point(0, 0), self.tiled_size)
        tiles = list(level_region.iterate_all())
        for index in range(0, len(tiles), number_of_tiles):
            yield tiles[index:index + number_of_tiles]

    def get_tile(self, tile_position: Point) -> bytes:
        """Return tile for tile position. Creates a new tile if the
        tile is not in cache.

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
            tile_job = NdpiTileJob(tile)
            # Create new tiles
            new_tiles = self._create_tiles(tile_job)
            # Add to tile cache
            self.tile_cache.update(new_tiles)
        return self.tile_cache[tile_position]

    def get_tiles(self, tile_positions: List[Point]) -> bytes:
        """Return tiles for tile positions. Sorts the requested tile posistions
        into tile jobs and uses a pool of threads to parse tile jobs. The
        results are concatenated.

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
            result = pool.map(self._parse_tile_job, tile_jobs)
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
        origin_coorindate = tile_job.origin
        try:
            frame = self.frame_cache[origin_coorindate]
        except KeyError:
            frame = self._get_frame(origin_coorindate)
            self.frame_cache = {}
            self.tile_cache = {}
            self.frame_cache[origin_coorindate] = frame
        tiles = self._crop_to_tiles(tile_job, frame)
        return tiles

    def _crop_to_tiles(
        self,
        tile_job: NdpiTileJob,
        jpeg_data: bytes
    ) -> Dict[Point, bytes]:
        """Crop jpeg data to tiles.

        Parameters
        ----------
        tile_job: NdpiTileJob
            Tile job defining the tiles to produce by cropping jpeg data.
        jpeg_data: bytes
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
                jpeg_data,
                tile.left,
                tile.top,
                tile.width,
                tile.height
            )
            for tile in tile_job
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
        origin_tiles: Dict[Point, NdpiTileJob] = {}
        for tile_position in tile_positions:
            tile = NdpiTile(tile_position, self.tile_size, self.frame_size)
            try:
                origin_tiles[tile.origin].append(tile)
            except KeyError:
                tile_job = NdpiTileJob(tile)
                origin_tiles[tile.origin] = tile_job
        return list(origin_tiles.values())

    def _parse_tile_job(self, tile_job: NdpiTileJob) -> bytes:
        """Return the concatenated bytes from a parsed tile job.

        Parameters
        ----------
        tile_job: NdpiTileJob
            Tile job to parse.

        Returns
        ----------
        bytes
            Concatenated bytes from tile job.

        """
        tiles = self._create_tiles(tile_job).values()
        return b"".join(tiles)

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


class NdpiOneFrameLevel(NdpiLevel):
    def __init__(
        self,
        page: TiffPage,
        fh: NdpiFileHandle,
        tile_size: Size,
        jpeg: TurboJPEG
    ):
        """Class for a ndpi level containing only one frame. The frame can be
        of any size (smaller or larger than the wanted tile size). The
        frame is padded to an even multipe of tile size. This is currently
        not lossless.

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
        original_size = Size(page.shape[1], page.shape[0])
        frame_size = (
            (original_size // tile_size + 1) * tile_size
        )
        super().__init__(
            page,
            fh,
            tile_size,
            frame_size,
            jpeg

        )

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
    def __init__(
        self,
        page: TiffPage,
        fh: NdpiFileHandle,
        tile_size: Size,
        jpeg: TurboJPEG
    ):
        (
            frame_width,
            frame_height, _, _
        ) = jpeg.decode_header(page.jpegheader)
        frame_size = Size(frame_width, frame_height)
        super().__init__(
            page,
            fh,
            tile_size,
            frame_size,
            jpeg
        )

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
        return position.x + position.y * self.framed_size.width

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
        return self._read(index)

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
        jpeg_data = self._header
        restart_marker_index = 0
        stripe_region = Region(
            (tile_position * self.tile_size) // self.frame_size,
            Size.max(self.tile_size // self.frame_size, Size(1, 1))
        )
        for stripe_coordiante in stripe_region.iterate_all():
            jpeg_data += self._get_stripe(stripe_coordiante)[:-1]
            jpeg_data += Tags.restart_mark(restart_marker_index)
            restart_marker_index += 1
        jpeg_data += Tags.end_of_image()
        return jpeg_data


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
