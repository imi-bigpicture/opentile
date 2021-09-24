import math
import struct
from abc import ABCMeta, abstractmethod
from functools import cached_property
from pathlib import Path
from struct import unpack
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
from tifffile import FileHandle, TiffFile, TiffPage
from tifffile.tifffile import TIFF

from opentile.common import OpenTilePage, Tiler
from opentile.geometry import Point, Region, Size, SizeMm
from opentile.turbojpeg_patch import TurboJPEG_patch as TurboJPEG
from opentile.utils import Jpeg


def get_value_from_ndpi_comments(
    comments: str,
    value_name: str,
    value_type: Type
) -> any:
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
        """Frame size required for reading tiles in NdpiTileJob."""
        return self._frame_size

    @property
    def tiles(self) -> List[NdpiTile]:
        """Tiles in NdpiTileJob."""
        return self._tiles

    @property
    def crop_parameters(self) -> List[Tuple[int, int, int, int]]:
        """Parameters for croping tiles in NdpiTileJob from frame."""
        return [
            (tile.left, tile.top, tile.width, tile.height)
            for tile in self._tiles
        ]


class NdpiPage(OpenTilePage):
    _pyramid_index = 0

    def __init__(
        self,
        page: TiffPage,
        fh: FileHandle,
        jpeg: TurboJPEG
    ):
        """Ndpi page that should not be tiled (e.g. overview or label).
        Image data is assumed to be jpeg.

        Parameters
        ----------
        page: TiffPage
            TiffPage defining the page.
        fh: FileHandle
            Filehandler to read data from.
        jpeg: TurboJpeg
            TurboJpeg instance to use.
        """
        super().__init__(page, fh)
        if self.compression != 'COMPRESSION.JPEG':
            raise NotImplementedError(
                f'{self.compression} is unsupported for ndpi '
                '(Only jpeg is supported)'
            )
        self._jpeg = jpeg

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._page}, {self._fh}, {self._jpeg}"
        )

    def get_tile(self, tile: Tuple[int, int]) -> bytes:
        if tile != (0, 0):
            raise ValueError
        return self._read_frame(0)

    @cached_property
    def focal_plane(self) -> List[float]:
        """Return focal plane (in um)."""
        # Return focal plane in um.
        try:
            # Defined in nm
            return self.page.ndpi_tags['ZOffsetFromSlideCenter'] / 1000.0
        except KeyError:
            return 0.0

    @property
    def pixel_spacing(self) -> SizeMm:
        """Return pixel spacing in mm per pixel."""
        return self.mpp * 1000.0

    @cached_property
    def mpp(self) -> SizeMm:
        """Return pixel spacing in um per pixel."""
        return self._get_mpp_from_page()

    @cached_property
    def properties(self) -> Dict[str, any]:
        """Return dictionary with ndpifile properties."""
        ndpi_tags = self.page.ndpi_tags
        manufacturer = ndpi_tags['Make']
        model = ndpi_tags['Model']
        software_versions = [ndpi_tags['Software']]
        device_serial_number = ndpi_tags['ScannerSerialNumber']
        aquisition_datatime = self._get_value_from_tiff_tags(
            self.page.tags, 'DateTime'
        )
        photometric_interpretation = self._get_value_from_tiff_tags(
            self.page.tags, 'PhotometricInterpretation'
        )
        return {
            'aquisition_datatime': aquisition_datatime,
            'device_serial_number': device_serial_number,
            'manufacturer': manufacturer,
            'model': model,
            'software_versions': software_versions,
            'photometric_interpretation': photometric_interpretation
        }

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

        mpp_x = 1/x_resolution
        mpp_y = 1/y_resolution
        return SizeMm(mpp_x, mpp_y)


class NdpiTiledPage(NdpiPage, metaclass=ABCMeta):
    def __init__(
        self,
        page: TiffPage,
        fh: FileHandle,
        base_shape: Size,
        tile_size: Size,
        jpeg: TurboJPEG,
        tile_cache: int = 10,
        frame_cache: int = 10
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
        jpeg: TurboJpeg
            TurboJpeg instance to use.
        tile_cache: int:
            Number of created tiles to cache.
        frame_cache: int:
            Number of read frames to cache.
        """
        super().__init__(page, fh, jpeg)
        self._base_shape = base_shape
        self._tile_size = tile_size
        self._file_frame_size = self._get_file_frame_size()
        self._pyramid_index = self._calculate_pyramidal_index(self._base_shape)
        self._tile_cache = NdpiCache(tile_cache)
        self._frame_cache = NdpiCache(frame_cache)
        self._headers: Dict[Size, bytes] = {}

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._page}, {self._fh}, "
            f"{self._base_shape}, {self.tile_size}, {self._jpeg}, "
            f"{self._tile_cache._size}, {self._frame_cache._size})"
        )

    @property
    def tile_size(self) -> Size:
        """The size of the tiles to generate."""
        return self._tile_size

    @cached_property
    def frame_size(self) -> Size:
        """The default read size used for reading frames."""
        return Size.max(self.tile_size, self._file_frame_size)

    @abstractmethod
    def _read_extended_frame(
        self, tile_position: Point,
        frame_size: Size
    ) -> bytes:
        """Read a frame of size frame_size covering tile_position."""
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
        """Return image bytes for tile at tile position. Caches created frame
        and tile.

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
            raise ValueError(
                f"Tile {tile_point} is outside "
                f"tiled size {self.tiled_size}"
            )
        # If tile not in cached
        if tile_point not in self._tile_cache:
            # Get frame size for reading out tile at tile point
            frame_size = self._get_frame_size_for_tile(tile_point)
            # Create a NdpiTile and add the single tile to a NdtiTileJob
            tile = NdpiTile(tile_point, self.tile_size, frame_size)
            tile_job = NdpiTileJob([tile])
            # Create the tile from the single NdpiTile in the NdpiTileJob
            new_tiles = self._create_tiles(tile_job)
            # Add the tile to tile cache
            self._tile_cache.update(new_tiles)
        return self._tile_cache[tile_point]

    def get_tiles(self, tile_positions: List[Tuple[int, int]]) -> List[bytes]:
        """Return list of image bytes for tile positions.

        Parameters
        ----------
        tile_positions: List[Tuple[int, int]]
            Tile positions to get.

        Returns
        ----------
        List[bytes]
            List of tile bytes.
        """
        tile_jobs = self._sort_into_tile_jobs(tile_positions)
        return [
            tile
            for tile_job in tile_jobs
            for tile in self._create_tiles(tile_job).values()
        ]

    def get_decoded_tiles(
        self, tile_positions: List[Tuple[int, int]]
    ) -> List[np.ndarray]:
        """Return list of decoded tiles for tiles at tile positions.

        Parameters
        ----------
        tile_positions: List[Tuple[int, int]]
            Tile positions to get.

        Returns
        ----------
        List[np.ndarray]
            List of decoded tiles.
        """
        tile_jobs = self._sort_into_tile_jobs(tile_positions)
        return [
            self._jpeg.decode(tile)
            for tile_job in tile_jobs
            for tile in self._create_tiles(tile_job).values()
        ]

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
        if tile_job.origin in self._frame_cache:
            frame = self._frame_cache[tile_job.origin]
        else:
            frame = self._read_extended_frame(
                tile_job.origin,
                tile_job.frame_size
            )
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
        tile_positions: List[Tuple[int, int]]
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
            tile_point = Point.from_tuple(tile_position)
            if not self._check_if_tile_inside_image(tile_point):
                raise ValueError(
                    f"Tile {tile_point} is outside "
                    f"tiled size {self.tiled_size}"
                )
            frame_size = self._get_frame_size_for_tile(tile_point)
            tile = NdpiTile(tile_point, self.tile_size, frame_size)
            if tile.origin in tile_jobs:
                tile_jobs[tile.origin].append(tile)
            else:
                tile_jobs[tile.origin] = NdpiTileJob([tile])
        return list(tile_jobs.values())


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

    def _read_extended_frame(self, origin: Point, frame_size: Size) -> bytes:
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
        frame = self._read_frame(0)
        # Use crop_multiple as it allows extending frame
        tile = self._jpeg.crop_multiple(
            frame,
            [(0, 0, frame_size.width, frame_size.height)]
        )[0]
        return tile


class NdpiStripedPage(NdpiTiledPage):
    """Class for a ndpi page containing stripes. Frames are constructed by
    concatenating multiple stripes, and from the frame one or more tiles can be
    produced by lossless cropping.
    """

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

    def _read_extended_frame(self, origin: Point, frame_size: Size) -> bytes:
        """Return extended frame of frame size starting at origin tile.
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
        if frame_size in self._headers:
            header = self._headers[frame_size]
        else:
            header = self._create_header(frame_size)
            self._headers[frame_size] = header
        jpeg_data = header
        restart_marker_index = 0

        stripe_region = Region(
            (origin * self.tile_size) // self.stripe_size,
            Size.max(frame_size // self.stripe_size, Size(1, 1))
        )
        for stripe_coordinate in stripe_region.iterate_all():
            index = self._get_stripe_position_to_index(stripe_coordinate)
            jpeg_data += self._read_frame(index)[:-1]
            jpeg_data += Jpeg.restart_mark(restart_marker_index)
            restart_marker_index += 1
        jpeg_data += Jpeg.end_of_image()
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
            header, Jpeg.start_of_frame()
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


class NdpiTiler(Tiler):
    def __init__(
        self,
        tiff_file: TiffFile,
        tile_size: Tuple[int, int],
        turbo_path: Path
    ):
        """Tiler for ndpi file, with functions to produce tiles of specified
        size.

        Parameters
        ----------
        tiff_file: TiffFile
            A ndpi-TiffFile.
        tile_size: Tuple[int, int]
            Tile size to cache and produce. Must be multiple of 8.
        turbo_path: Path
            Path to turbojpeg (dll or so).

        """
        super().__init__(tiff_file)

        self._fh = self._tiff_file.filehandle
        self._tile_size = Size(*tile_size)
        # Subsampling not accounted for!
        if self.tile_size.width % 8 != 0 or self.tile_size.height % 8 != 0:
            raise ValueError(f"Tile size {self.tile_size} not divisable by 8")
        self._turbo_path = turbo_path
        self._jpeg = TurboJPEG(self._turbo_path)
        # Keys are series, level, page
        self._pages: Dict[(int, int, int), NdpiPage] = {}

        self._level_series_index = 0
        for series_index, series in enumerate(self.series):
            if series.name == 'Label':
                self._label_series_index = series_index
            elif series.name == 'Macro':
                self._overview_series_index = series_index

    def __repr__(self) -> str:
        return (
            f"NdpiTiler({self._tiff_file.filename}, "
            f"{self.tile_size.to_tuple}, "
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
        """Return NdpiPage for series, level, page. NdpiPages holds a cache, so
        store created pages.
        """
        if (series, level, page) in self._pages:
            ndpi_page = self._pages[series, level, page]
        else:
            ndpi_page = self._create_page(series, level, page)
            self._pages[series, level, page] = ndpi_page
        return ndpi_page

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
        page: TiffPage = (
            self._tiff_file.series[series].levels[level].pages[page]
        )
        if page.is_tiled:  # Striped ndpi page
            return NdpiStripedPage(
                page,
                self._fh,
                self.base_size,
                self.tile_size,
                self._jpeg
            )
        if series == self._level_series_index:  # Single frame, force tiling
            return NdpiOneFramePage(
                page,
                self._fh,
                self.base_size,
                self.tile_size,
                self._jpeg
            )
        return NdpiPage(
            page,
            self._fh,
            self._jpeg
        )  # Single frame, do not tile
