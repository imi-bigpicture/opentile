import math
import struct
import threading
from abc import ABCMeta, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from pathlib import Path
from struct import unpack
from typing import Dict, Iterator, List, Optional, Tuple

from tifffile import FileHandle, TiffPage
from tifffile.tifffile import TIFF
from wsidicom.geometry import Point, Region, Size, SizeMm

from opentile.interface import TifffileTiler, TiledPage
from opentile.turbojpeg_patch import TurboJPEG_patch as TurboJPEG


class Tags:
    TAGS = {
        'tag marker': 0xFF,
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
        return bytes([cls.TAGS['tag marker'], cls.TAGS['start of frame']])

    @classmethod
    def end_of_image(cls) -> bytes:
        """Return bytes representing a end of image tag."""
        return bytes([cls.TAGS['tag marker'], cls.TAGS['end of image']])

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

    def __str__(self) -> str:
        return f"NdpiCache of size {len(self)} and max size {self._size}"

    def __repr__(self) -> str:
        return f"NdpiCache({self._size})"

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

    def __str__(self) -> str:
        return f"NdpiFileHandle for FileHandle {self._fh}"

    def __repr__(self) -> str:
        return f"NdpiFileHandle({self._fh})"

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

    def close(self) -> None:
        self._fh.close()


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

    def __str__(self) -> str:
        return f"NdpiTile of position {self.position}"

    @property
    def position(self) -> Point:
        return self._tile_position

    @cached_property
    def origin(self) -> Point:
        """Return origin tile position for tile. The origin tile is the first
        tile (upper left) of the frame that containts the tile.

        Returns
        ----------
        Point
            The origin tile position for tile.

        """
        return (self.position // self.tiles_per_frame) * self.tiles_per_frame

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

    @property
    def frame_size(self) -> Size:
        return self._frame_size

    @property
    def tiles_per_frame(self) -> Size:
        return self._tiles_per_frame

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
        return (
            (tile_position * self._tile_size)
            % Size.max(self._frame_size, self._tile_size)
        )


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
        first_tile = tiles.pop(0)
        self._origin = first_tile.origin
        self._frame_size = first_tile.frame_size
        self._tiles: List[NdpiTile] = [first_tile]
        for tile in tiles:
            self.append(tile)

    def __repr__(self) -> str:
        return f"NdpiTileJob({self.tiles})"

    def __str__(self) -> str:
        return f"NdpiTileJob of tiles {self.tiles}"

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
    def frame_size(self) -> Size:
        return self._frame_size

    @property
    def tiles(self) -> List[NdpiTile]:
        return self._tiles

    @property
    def crop_parameters(self) -> List[Tuple[int, int, int, int]]:
        return [
            (tile.left, tile.top, tile.width, tile.height)
            for tile in self._tiles
        ]


class NdpiPage(TiledPage, metaclass=ABCMeta):
    def __init__(
        self,
        page: TiffPage,
        fh: NdpiFileHandle,
        base_shape: Size,
        tile_size: Size,
        jpeg: TurboJPEG,
        tile_cache: int = 10,
        frame_cache: int = 10
    ):
        """Metaclass for a ndpi page, should not be used.

        Parameters
        ----------
        page: TiffPage
            TiffPage defining the page.
        fh: NdpiFileHandle
            Filehandler to read data from.
        tile_size: Size
            Requested tile size.
        jpeg: TurboJpeg
            TurboJpeg instance to use.
        tile_cache: int:
            Number of created tiles to cache.
        frame_cache: int:
            Number of read frames to cache.
        """
        super().__init__(page, fh)
        # print(
        #     f"make pyramid index by: {base_shape, self.image_size}",
        #     int(
        #         math.log2(base_shape.width/self.image_size.width)
        #     )
        # )
        self._pyramid_index = int(
            math.log2(base_shape.width/self.image_size.width)
        )
        self._tile_size = tile_size
        self._jpeg = jpeg
        self._file_frame_size = self._get_file_frame_size()
        self._tile_cache = NdpiCache(tile_cache)
        self._frame_cache = NdpiCache(frame_cache)
        self._headers: Dict[Size, bytes] = {}

    @property
    def pyramid_index(self) -> int:
        return self._pyramid_index

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError

    @property
    def tile_size(self) -> Size:
        """The size of the tiles to generate."""
        return self._tile_size

    @property
    def pixel_spacing(self) -> SizeMm:
        return self.mpp * 1000.0

    @cached_property
    def mpp(self) -> SizeMm:
        return self._get_mpp_from_page()

    @cached_property
    def image_size(self) -> Size:
        """The size of the level."""
        return Size(self._page.shape[1], self._page.shape[0])

    @cached_property
    def frame_size(self) -> Size:
        """The default read size used for reading frames."""
        return Size.max(self.tile_size, self._file_frame_size)

    @cached_property
    def tiled_size(self) -> Size:
        """The level size when tiled (columns and rows of tiles)."""
        return (self.image_size / self.tile_size).ceil()

    @abstractmethod
    def _read_frame(self, tile_position: Point, frame_size: Size) -> bytes:
        """Read frame of frame size covering tile position."""
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
        tile_position: Point,
        z: float = 0,
        path: str = '0'
    ) -> bytes:
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
        if not self._check_if_tile_inside_image(tile_position):
            raise ValueError(
                f"Tile {tile_position} is outside "
                f"tiled size {self.tiled_size}"
            )
        # If tile not in cached
        if tile_position not in self._tile_cache.keys():
            # Create a tile job
            frame_size = self._get_frame_size_for_tile(tile_position)
            tile = NdpiTile(tile_position, self.tile_size, frame_size)
            tile_job = NdpiTileJob([tile])
            # Create tile
            new_tiles = self._create_tiles(tile_job)
            # Add to tile cache
            self._tile_cache.update(new_tiles)
        return self._tile_cache[tile_position]

    def get_tiles(self, tiles: List[Point]) -> Iterator[List[bytes]]:
        tile_jobs = self._sort_into_tile_jobs(tiles)
        with ThreadPoolExecutor() as pool:
            def thread(tile_job: NdpiTileJob) -> List[bytes]:
                return self._create_tiles(tile_job).values()
            return pool.map(thread, tile_jobs)

    def _create_tiles(
        self,
        tile_job: NdpiTileJob
    ) -> Dict[Point, bytes]:
        """Return tiles defined by tile job.

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
            frame = self._frame_cache[tile_job.origin]
        except KeyError:
            frame = self._read_frame(tile_job.origin, tile_job.frame_size)
            self._frame_cache[tile_job.origin] = frame
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
        tiles = self._jpeg.crop_multiple(
            frame,
            tile_job.crop_parameters
        )
        return {
            tile.position: tiles[i]
            for i, tile in enumerate(tile_job.tiles)
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
            if not self._check_if_tile_inside_image(tile_position):
                raise ValueError(
                    f"Tile {tile_position} is outside "
                    f"tiled size {self.tiled_size}"
                )
            frame_size = self._get_frame_size_for_tile(tile_position)
            tile = NdpiTile(tile_position, self.tile_size, frame_size)
            try:
                tile_jobs[tile.origin].append(tile)
            except KeyError:
                tile_jobs[tile.origin] = NdpiTileJob([tile])
        return list(tile_jobs.values())

    def _read(self, index: int) -> bytes:
        """Read frame bytes at index from file. Locks the filehandle while
        reading.

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
        return (
            tile_position.x < self.tiled_size.width and
            tile_position.y < self.tiled_size.height
        )

    def _get_mpp_from_page(self) -> SizeMm:
        x_resolution = self.page.tags['XResolution'].value[0]
        y_resolution = self.page.tags['YResolution'].value[0]
        resolution_unit = self.page.tags['ResolutionUnit'].value
        if resolution_unit != TIFF.RESUNIT.CENTIMETER:
            raise ValueError("Unkown resolution unit")

        mpp_x = 1/x_resolution
        mpp_y = 1/y_resolution
        return SizeMm(mpp_x, mpp_y)


class NdpiOneFramePage(NdpiPage):
    """Class for a ndpi page containing only one frame. The frame can be
    of any size (smaller or larger than the wanted tile size). The
    frame is padded to an even multipe of tile size.
    """
    def __repr__(self) -> str:
        return (
            f"NdpiOneFramePage({self._page}, {self._fh}, "
            f"{self.tile_size}, {self._jpeg})"
        )

    def __str__(self) -> str:
        return f"NdpiOneFramePage of page {self._page}"

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

    def _read_frame(self, origin: Point, frame_size: Size) -> bytes:
        """Return padded image covering tile coordiante as valid jpeg bytes.

        Parameters
        ----------
        origin: Point
            Upper left tile position that should be covered by the frame.
        frame_size: Size
            Size of the frame to read.

        Returns
        ----------
        bytes
            Frame
        """
        if origin != Point(0, 0):
            raise ValueError("Origin not (0, 0) for one frame level.")
        frame = self._read(0)
        # Use crop_multiple as it allows extending frame
        tile = self._jpeg.crop_multiple(
            frame,
            [(0, 0, frame_size.width, frame_size.height)]
        )[0]
        return tile


class NdpiStripedPage(NdpiPage):
    """Class for a ndpi page containing stripes. Frames are constructed by
    concatenating multiple stripes, and from the frame one or more tiles can be
    produced by lossless cropping.
    """

    def __repr__(self) -> str:
        return (
            f"NdpiStripedPage({self._page}, {self._fh}, "
            f"{self.tile_size}, {self._jpeg})"
        )

    def __str__(self) -> str:
        return f"NdpiStripedPage of page {self._page}"

    @property
    def stripe_size(self) -> Size:
        """Size of stripes."""
        return self._file_frame_size

    @cached_property
    def striped_size(self) -> Size:
        """Number of stripes in columns and rows."""
        return Size(self._page.chunked[1], self._page.chunked[0])

    def _get_file_frame_size(self) -> Size:
        """Return size of stripes in file. For striped levels this is parsed
        from the jpeg header.

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

    @staticmethod
    def _get_partial_frame_dimension(
        tile_position: int,
        striped_size: int,
        tile_size: int,
        stripe_size: int
    ) -> int:
        """Return frame dimension (either width or height) for edge tile
        at tile position, so that the frame does not extend beyond the image.

        Parameters
        ----------
        tile_position: int
            Tile position (x or y) for frame size calculation.
        striped_size: int
            Striped size for image (width or height).
        tile_size: int
            Requested tile size (width or height).
        stripe_size: int
            Stripe size (width or height).

        Returns
        ----------
        int
            Frame size (width or height) to be used at tile position.
        """

        return int(stripe_size * striped_size - tile_position * tile_size)

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

        is_partial_frame = self._is_partial_frame(tile_position)
        if is_partial_frame[0]:
            width = self._get_partial_frame_dimension(
                tile_position.x,
                self.striped_size.width,
                self.tile_size.width,
                self.stripe_size.width
            )
        else:
            width = self.frame_size.width

        if is_partial_frame[1]:
            height = self._get_partial_frame_dimension(
                tile_position.y,
                self.striped_size.height,
                self.tile_size.height,
                self.stripe_size.height
            )
        else:
            height = self.frame_size.height
        return Size(width, height)

    def _read_frame(self, origin: Point, frame_size: Size) -> bytes:
        """Return concatenated frame of frame size starting at origin tile.
        Returned frame is jpeg bytes including header with correct image size.
        Original restart markers are updated to get the proper incrementation.
        End of image tag is appended.

        Parameters
        ----------
        origin: Point
            Upper left tile position that should be covered by the frame.
        frame_size: Size
            Size of the frame to get.

        Returns
        ----------
        bytes
            Concatenated frame as jpeg bytes.
        """
        try:
            header = self._headers[frame_size]
        except KeyError:
            header = self._create_header(frame_size)
            self._headers[frame_size] = header
        jpeg_data = header
        restart_marker_index = 0

        stripe_region = Region(
            (origin * self.tile_size) // self.stripe_size,
            Size.max(frame_size // self.stripe_size, Size(1, 1))
        )
        for stripe_coordiante in stripe_region.iterate_all():
            jpeg_data += self._read_stripe(stripe_coordiante)[:-1]
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

    def _create_header(
        self,
        size: Size,
    ) -> bytes:
        """Return manipulated header with changed pixel size (width, height).

        Parameters
        ----------

        size: Size
            Pixel size to insert into header.

        Returns
        ----------
        bytes:
            Manupulated header.
        """

        header = bytearray(self._page.jpegheader)
        start_of_frame_index, length = self._find_tag(
            header, Tags.start_of_frame()
        )
        if start_of_frame_index is None:
            raise ValueError("Start of scan tag not found in header")
        size_index = start_of_frame_index+5
        header[size_index:size_index+2] = struct.pack(">H", size.height)
        header[size_index+2:size_index+4] = struct.pack(">H", size.width)

        return bytes(header)

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

    def _read_stripe(self, position: Point) -> bytes:
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
        index = self._get_stripe_position_to_index(position)
        try:
            data = self._read(index)
        except IndexError:
            raise IndexError(
                f"error reading stripe {position} with index {index}"
            )
        return data


class NdpiTiler(TifffileTiler):
    def __init__(
        self,
        filepath: str,
        tile_size: Tuple[int, int],
        turbo_path: Path
    ):
        """Cache for ndpi stripes, with functions to produce tiles of specified
        size.

        Parameters
        ----------
        filepath: str
            File path to ndpi file.
        tile_size: Tuple[int, int]
            Tile size to cache and produce. Must be multiple of 8.
        turbo_path: Path
            Path to turbojpeg (dll or so).

        """
        super().__init__(filepath)

        self._fh = NdpiFileHandle(self._tiff_file.filehandle)
        self._tile_size = Size(*tile_size)
        # Subsampling not accounted for!
        if self.tile_size.width % 8 != 0 or self.tile_size.height % 8 != 0:
            raise ValueError(f"Tile size {self.tile_size} not divisable by 8")
        self._turbo_path = turbo_path
        self._jpeg = TurboJPEG(self._turbo_path)
        # Keys are series, level, page
        self._pages: Dict[(int, int, int), NdpiPage] = {}

        self._volume_series_index = 0
        for series_index, series in enumerate(self.series):
            if series.name == 'Label':
                self._label_series_index = series_index
            elif series.name == 'Macro':
                self._overview_series_index = series_index

    def __repr__(self) -> str:
        return (
            f"NdpiTiler({self._filepath}, {self.tile_size.to_tuple}, "
            f"{self._turbo_path})"
        )

    def __str__(self) -> str:
        return f"NdpiTiler of Tifffile {self._tiff_file}"

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
        try:
            ndpi_page = self._pages[series, level, page]
        except KeyError:
            ndpi_page = self._create_page(series, level, page)
            self._pages[series, level, page] = ndpi_page
        return ndpi_page

    def _create_page(
        self,
        series: int,
        level: int,
        page: int,
    ) -> NdpiPage:
        """Create a new page.

        Parameters
        ----------
        level: int
            Level to add

        Returns
        ----------
        NdpiLevel
            Created level.
        """
        page: TiffPage = (
            self._tiff_file.series[series].levels[level].pages[page]
        )
        if page.is_tiled:
            return NdpiStripedPage(
                page,
                self._fh,
                self.base_size,
                self.tile_size,
                self._jpeg
            )
        return NdpiOneFramePage(
                page,
                self._fh,
                self.base_size,
                self.tile_size,
                self._jpeg
            )
