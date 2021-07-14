import concurrent.futures
import threading
import struct
from dataclasses import dataclass
from pathlib import Path
from struct import unpack
from typing import Dict, Generator, List, Optional, Tuple
from abc import ABCMeta, abstractmethod

from tifffile import FileHandle, TiffPage
from tifffile.tifffile import TiffPageSeries
from turbojpeg import TurboJPEG


@dataclass
class Size:
    width: int
    height: int

    def __str__(self):
        return f'{self.width}x{self.height}'

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
        """Return stripe bytes.

        Parameters
        ----------
        offset: int
            Offset in bytes to stripe.
        bytecount: int
            Length in bytes of stripe.

        Returns
        ----------
        bytes
            Stripe as bytes.
        """
        with self._lock:
            self._fh.seek(offset)
            stripe = self._fh.read(bytecount)
        return stripe


class NdpiLevel(metaclass=ABCMeta):
    def __init__(
        self,
        page: TiffPage,
        fh: NdpiFileHandle,
        jpeg: TurboJPEG,
        tile_size: Size
    ):
        self._page = page
        self._fh = fh
        self._tile_size = tile_size
        self._jpeg = jpeg
        self.tiles: Dict[Point, bytes] = {}

    @property
    def tile_size(self) -> Size:
        """The size of the tiles to generate."""
        return self._tile_size

    def _read_data(self, index: int) -> bytes:
        offset = self._page.dataoffsets[index]
        bytecount = self._page.databytecounts[index]
        return self._fh.read(offset, bytecount)

    @abstractmethod
    def get_tile(
        self,
        tile_position: Tuple[int, int]
    ) -> bytes:
        raise NotImplementedError

    @abstractmethod
    def _create_tiles(
        self,
        requested_tile: Point
    ) -> Dict[Point, bytes]:
        raise NotImplementedError


class NdpiLevelNonTiled(NdpiLevel):
    pass


class NdpiLevelTiled(NdpiLevel):
    def __init__(
        self,
        page: TiffPage,
        fh: NdpiFileHandle,
        jpeg: TurboJPEG,
        tile_size: Size
    ):
        super().__init__(page, fh, jpeg, tile_size)

        (
            stripe_width,
            stripe_height, _, _
        ) = self._jpeg.decode_header(self._page.jpegheader)

        self._stripe_size = Size(stripe_width, stripe_height)
        self._striped_size = Size(self._page.chunked[1], self._page.chunked[0])
        imaged_size = self.striped_size * self.stripe_size
        self._tiled_size = imaged_size // self.tile_size
        read_size = Size.max(
            self._tile_size,
            self._stripe_size
            )
        self._header = self._update_header(self._page.jpegheader, read_size)

    @property
    def stripe_size(self) -> Size:
        """The size of the stripes in the level."""
        return self._stripe_size

    @property
    def striped_size(self) -> Size:
        """The level size when striped (columns and rows of stripes)."""
        return self._striped_size

    @property
    def tiled_size(self) -> Size:
        """The level size when tiled (coluns and rows of tiles)."""
        return self._tiled_size

    def get_tile(
        self,
        tile_position: Tuple[int, int]
    ) -> bytes:
        """Return tile for tile position x and y. If stripes for the tile
        is not cached, read them from disk and parse the jpeg data.

        Parameters
        ----------
        tile_position: Tuple[int, int]
            Position of tile to get.

        Returns
        ----------
        bytes
            Produced tile at position, wrapped in header.
        """
        tile_point = Point(*tile_position)

        # Check if tile not in cached
        if tile_point not in self.tiles.keys():
            # Empty cache
            self.tiles = {}
            # Create tiles and add to tile cache
            self.tiles.update(self._create_tiles(tile_point))

        return self.tiles[tile_point]

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

    def _stripe_coordinate_to_index(self, coordinate: Point) -> int:
        return coordinate.x + coordinate.y * self.striped_size.width

    def _get_stripe(self, coordinate: Point) -> bytes:
        """Return stripe bytes for stripe at point.

        Parameters
        ----------
        coordinate: Point
            Coordinate of stripe to get.

        Returns
        ----------
        bytes
            Stripe as bytes.
        """
        index = self._stripe_coordinate_to_index(coordinate)
        return self._read_data(index)

    def _get_stitched_image(self, tile_coordinate: Point) -> bytes:
        """Return stitched image covering tile coorindate as valid jpeg bytes.
        Includes header with the correct image size. Original restart markers
        are updated to get the proper incrementation. End of image tag is
        appended end.

        Parameters
        ----------
        tile_coordinate: Point
            Tile coordinate that should be covered by the stripe region.

        Returns
        ----------
        bytes
            Stitched image as jpeg bytes.
        """
        jpeg_data = self._header
        restart_marker_index = 0
        stripe_region = Region(
            (tile_coordinate * self.tile_size) // self.stripe_size,
            Size.max(self.tile_size // self.stripe_size, Size(1, 1))
        )
        for stripe_coordiante in stripe_region.iterate_all():
            jpeg_data += self._get_stripe(stripe_coordiante)[:-1]
            jpeg_data += Tags.restart_mark(restart_marker_index)
            restart_marker_index += 1
        jpeg_data += Tags.end_of_image()
        return jpeg_data

    def _map_tile_to_image(self, tile_coordinate: Point) -> Point:
        """Map a tile coorindate to image coorindate.

        Parameters
        ----------
        tile_coordinate: Point
            Tile coordinate that should be map to image coordinate.

        Returns
        ----------
        Point
            Image coordiante for tile.
        """
        return tile_coordinate * self.tile_size

    def _create_tiles(
        self,
        requested_tile: Point
    ) -> Dict[Point, bytes]:
        """Return tiles created by parsing jpeg data. Additional tiles than the
        requested tile may be created if the stripes span multiple tiles.

        Parameters
        ----------
        jpeg_data: bytes
            Jpeg data covering the region to create tiles from.
        requested_tile: Point
            Coordinate of requested tile that should be created.

        Returns
        ----------
        Dict[Point, bytes]:
            Created tiles ordered by tile coordiante.
        """

        # Starting tile should be at stripe border
        ratio = self.stripe_size / self.tile_size
        starting_tile = requested_tile - (requested_tile % ratio)

        # Create jpeg data from stripes
        jpeg_data = self._get_stitched_image(starting_tile)

        tile_region = Region(
            starting_tile,
            Size.max(self.stripe_size // self.tile_size, Size(1, 1))
        )

        tiles = {}

        def crop_thread(tile: Point):
            tiles[tile] = self.jpeg.crop(
                jpeg_data,
                self._map_tile_to_image(tile).x % self.stripe_size.width,
                self._map_tile_to_image(tile).y % self.stripe_size.height,
                self.tile_size.width,
                self.tile_size.height
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            executor.map(crop_thread, tile_region.iterate_all())

        return tiles


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
        tile_position: Tuple[int, int]
            Position of tile to get.

        Returns
        ----------
        bytes
            Produced tile at position, wrapped in header.
        """
        try:
            ndpi_level = self._levels[level]
        except KeyError:
            ndpi_level = self._create_level(level)
            self._levels[level] = ndpi_level

        return ndpi_level.get_tile(*tile_position)

    def _create_level(self, level: int) -> NdpiLevel:
        page: TiffPage = self._tiff_series.levels[level].pages[0]
        if page.is_tiled:
            level_type = NdpiLevelTiled
        else:
            level_type = NdpiLevelNonTiled
        return level_type(page, self._fh, self.jpeg, self.tile_size)
