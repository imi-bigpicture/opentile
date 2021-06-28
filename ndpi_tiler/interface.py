from typing import DefaultDict, Generator, Dict, Tuple, List

from collections import defaultdict

from bitarray import bitarray
from tifffile import FileHandle, TiffPage

from ndpi_tiler.jpeg import JpegHeader, JpegSegment, MCU_SIZE
from ndpi_tiler.jpeg_tags import TAGS


class Tile:
    def __init__(
        self,
        header: bytes,
        tile_width: int,
        tile_height: int,
        segment_width: int,
        segment_height: int,
        components: int
    ):
        self.header = header
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.segment_width = segment_width
        self.segment_height = segment_height
        self.components = components
        self.segments: Dict[(int, int), JpegSegment] = {}
        self.tile: bytes = None

    def add_segment(self, segment: JpegSegment, x: int, y: int):
        self.segments[(x, y)] = segment

    def get_tile(self) -> bytes:
        if self.tile is not None:
            return self.tile

        scan = bitarray()
        dc_offsets = [0] * self.components
        for y in range(self.tile_height//self.segment_width):
            for x in range(self.tile_width//self.segment_width):
                segment = self.segments[(x, y)]
                segment_previous_offset = segment.dc_offset
                first_mcu = segment.first
                # modify first_mcu according to dc_offset and previous offset
                # update dc_offsets for next segment
                rest = segment.rest
                scan += first_mcu + rest

        # Pad scan with ending 1
        padding_bits = 7 - (len(scan) - 1) % 8
        scan += bitarray(padding_bits*[1])

        # Convert to bytes
        scan_bytes = bytearray(scan.tobytes())

        # Add byte stuffing after 0xFF
        tag_index = None
        start_search = 0
        while tag_index != -1:
            tag_index = scan_bytes.find(TAGS['tag'], start_search)
            if tag_index != -1:
                scan_bytes.insert(tag_index+1, (TAGS['stuffing']))
                start_search = tag_index+1

        # Wrap scan with modified header and end of image tag.
        self.tile = JpegHeader.wrap_scan(
            self.header,
            scan_bytes,
            (self.tile_width, self.tile_height)
        )

        return self.tile


class NdpiPageTiler:
    def __init__(
        self,
        fh: FileHandle,
        page: TiffPage,
        tile_width: int,
        tile_height: int
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
        self._header = JpegHeader.from_bytes(self._page.jpegheader)
        self._stripe_width = self._header.width
        self._stripe_height = self._header.height
        self.stripe_cols = self._page.chunked[1]
        self._tile_width = tile_width
        self._tile_height = tile_height

        self.tiles: Dict[(int, int), Tile] = {}

    @property
    def stripe_size(self) -> Tuple[int, int]:
        return (self._header.width, self._header.height)

    @property
    def tile_size(self) -> Tuple[int, int]:
        return (self._tile_width, self._tile_height)

    def stripe_range(
        self,
        tile: int,
        stripe_size: int,
        tile_size: int
    ) -> range:
        start = (tile * tile_size) // stripe_size
        end = max(
            start + 1,
            ((tile+1) * tile_size) // stripe_size
        )
        return range(start, end)

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
        # Check if tile not is cached
        if (x, y) not in self.tiles.keys():
            x_range = self.stripe_range(x, self._stripe_width, self._tile_width)
            y_range = self.stripe_range(y, self._stripe_height, self._tile_height)
            # If first stripe is not in cached segments, get the stripes
            stripe_indices = [
                stripe_x + stripe_y * self.stripe_cols
                for stripe_y in y_range
                for stripe_x in x_range
            ]
            # Generator producing (stripe, stripe_index), requires patched
            # tifffile
            stripes: Generator[bytes, int] = self._fh.read_segments(
                offsets=self._page.dataoffsets,
                bytecounts=self._page.databytecounts,
                indices=stripe_indices,
            )
            # Loop over the stripes and get segments.
            for stripe, stripe_index in stripes:
                segments = self._header.get_segments(
                    stripe,
                    self._tile_width,
                )
                # For each segment, insert into segment cache at pixel (x y)
                # position
                for segment_index, segment in enumerate(segments):
                    segment_x = (
                        (stripe_index % self.stripe_cols) * self._stripe_width
                        + segment_index*self._tile_width
                    )
                    segment_y = (
                        (stripe_index // self.stripe_cols)
                        * self._stripe_height
                    )
                    tile_x = segment_x // self._tile_width
                    tile_y = segment_y // self._tile_height
                    # print(f"tile {tile_x, tile_y}")
                    if (tile_x, tile_y) not in self.tiles.keys():
                        tile = Tile(
                            self._page.jpegheader,
                            self._tile_width,
                            self._tile_height,
                            segment.count*MCU_SIZE, # should be a better way
                            MCU_SIZE,
                            3
                        )
                        self.tiles[(tile_x, tile_y)] = tile
                    else:
                        tile = self.tiles[(x, y)]

                    # print(f"segment x,y {segment_x, segment_y}")
                    tile.add_segment(
                        segment,
                        segment_x % self._tile_width,
                        segment_y % self._tile_height
                    )

        return self.tiles[x, y].get_tile()
