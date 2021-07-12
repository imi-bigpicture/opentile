from typing import Dict, List, Tuple, Type, Optional

from struct import unpack
import struct
from .jpeg_tags import TAGS
from tifffile import FileHandle, TiffPage

from turbojpeg import TurboJPEG


class NdpiPageTiler:
    def __init__(
        self,
        fh: FileHandle,
        page: TiffPage,
        tile_width: int,
        tile_height: int,
    ):
        """Cache for ndpi stripes, with functions to produce tiles of specified
        with and height.

        Parameters
        ----------
        fh: FileHandle
            File handle to stripe data.
        page: TiffPage
            Page to cache and tile.
        tile_width: int
            Tile width to cache and produce.
        tile_height: int
            Tile heigh to cache and produce.

        """
        self._fh = fh
        self._page = page
        self.jpeg = TurboJPEG(r'C:\tools\libjpeg-turbo-vc64\bin\turbojpeg.dll')
        (
            self._stripe_width,
            self._stripe_height, _, _
        ) = self.jpeg.decode_header(page.jpegheader)

        self.stripe_cols = self._page.chunked[1]
        self._tile_width = tile_width
        self._tile_height = tile_height

        self.tiles: Dict[(int, int), bytes] = {}

    @property
    def stripe_size(self) -> Tuple[int, int]:
        return (self._stripe_width, self._stripe_height)

    @property
    def tile_size(self) -> Tuple[int, int]:
        return (self._tile_width, self._tile_height)

    @staticmethod
    def find_tag(
        header: bytes,
        tag: int
    ) -> Tuple[Optional[int], Optional[int]]:
        """Return first index and length of payload of tag in header."""
        index = header.find(tag.to_bytes(2, 'big'))
        if index != -1:
            (length, ) = unpack('>H', header[index+2:index+4])
            return index, length
        return None, None

    @classmethod
    def manupulated_header(
        cls,
        header: bytes,
        size: Tuple[int, int],
        remove_restart_interval: bool = False
    ) -> bytes:
        """Return manipulated header with changed pixel size (width, height)
        and removed reset interval marker.

        Parameters
        ----------
        heaer: bytes
            Header to manipulate.
        size: Tuple[int, int]
            Pixel size to insert into header.

        Returns
        ----------
        bytes:
            Manupulated header.
        """
        header = bytearray(header)
        start_of_scan_index, length = cls.find_tag(
            header, TAGS['start of frame']
        )
        if start_of_scan_index is None:
            raise ValueError("Start of scan tag not found in header")
        size_index = start_of_scan_index+5
        header[size_index:size_index+2] = struct.pack(">H", size[1])
        header[size_index+2:size_index+4] = struct.pack(">H", size[0])

        if remove_restart_interval:
            restart_interval_index, length = cls.find_tag(
                header, TAGS['restart interval']
            )
            if restart_interval_index is not None:
                del header[
                    restart_interval_index:restart_interval_index+length+2
                ]

        return bytes(header)

    def stripe_range(
        self,
        tile: int,
        stripe_size: int,
        tile_size: int
    ) -> range:
        """Return stripe coordinate range given a tile coordinate,
        stripe size and tile size.

        Parameters
        ----------
        tile: int
            Tile coordinate (x or y).
        stripe_size: int
            Stripe width or height.
        tile_size: int
            Tile width or height.

        Returns
        ----------
        range
            Range of stripes needed to cover tile.
        """
        start = (tile * tile_size) // stripe_size
        end = ((tile+1) * tile_size) // stripe_size
        if start == end:
            end += 1
        return range(start, end)

    def get_stripe(self, coordinate: Tuple[int, int]) -> bytes:
        index = coordinate[0] + coordinate[1] * self.stripe_cols
        offset = self._page.dataoffsets[index]
        bytecount = self._page.databytecounts[index]
        self._fh.seek(offset)
        return self._fh.read(bytecount)

    def get_tile(
        self,
        x: int,
        y: int
    ) -> bytes:
        """Produce a tile for tile position x and y. If stripes for the tile
        is not cached, read them from disk and parse the jpeg data.

        Parameters
        ----------
        x: int
            X tile position to get.
        y: int
            Y tile position to get.

        Returns
        ----------
        bytes
            Produced tile at x, y, wrapped in header.
        """
        # Check if tile not in cached
        if (x, y) not in self.tiles.keys():
            # Empty cache
            self.tiles = {}

            jpeg_data = self.manupulated_header(
                self._page.jpegheader,
                (self._stripe_width, self._tile_height)
            )
            restart_marker_index = 0
            for stripe_y in self.stripe_range(
                    y,
                    self._stripe_height,
                    self._tile_height
            ):
                for stripe_x in self.stripe_range(
                    x,
                    self._stripe_width,
                    self._tile_width
                ):
                    jpeg_data += self.get_stripe((stripe_x, stripe_y))[:-1]
                    jpeg_data += bytes([0xD0 + restart_marker_index % 8])
                    restart_marker_index += 1
            jpeg_data += bytes([0xFF, 0xD9])

            tile_range = {
                (tile_x, tile_y)
                for tile_y in self.stripe_range(
                    y * self._tile_height // self._stripe_height,
                    self._tile_height,
                    self._stripe_height
                )
                for tile_x in self.stripe_range(
                    x * self._tile_width // self._stripe_width,
                    self._tile_width,
                    self._stripe_width
                )
            }

            for (tile_x, tile_y) in tile_range:
                tile = self.jpeg.crop(
                    jpeg_data,
                    0,
                    0,
                    1024,
                    1024
                )
                self.tiles[x, y] = tile

        return self.tiles[x, y]
