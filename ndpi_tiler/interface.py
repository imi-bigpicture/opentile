from collections import defaultdict
from typing import DefaultDict, Generator, Dict, Tuple, List

from tifffile import FileHandle, TiffPage
from bitarray import bitarray
from ndpi_tiler.jpeg import Component, JpegHeader, JpegSegment, Dc
from ndpi_tiler.jpeg_tags import BYTE_TAG, BYTE_TAG_STUFFING


class Tile:
    def __init__(
        self,
        header: bytes,
        tile_width: int,
        tile_height: int,
        components: Dict[str, Component]
    ):
        self.header = header
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.bits = bitarray()
        self.tile: bytes = None
        self.dc_offsets = Dc.zero(list(components.keys()))

    def add_segment(self, segment: JpegSegment):
        self.bits += segment.data
        # self.dc_offsets = segment.tile_offset

    def get_tile(self) -> bytes:
        if self.tile is not None:
            return self.tile

        # Pad scan with ending 1
        padding_bits = 7 - (len(self.bits) - 1) % 8
        self.bits += bitarray(padding_bits*[1])

        # Convert to bytes
        scan_bytes = bytearray(self.bits.tobytes())

        # Add byte stuffing after 0xFF
        scan_bytes = scan_bytes.replace(BYTE_TAG, BYTE_TAG_STUFFING)

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
        end = max(
            start + 1,
            ((tile+1) * tile_size) // stripe_size
        )
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
        # Check if tile not is cached
        if (x, y) not in self.tiles.keys():

            stripes = {
                (stripe_x, stripe_y): self.get_stripe((stripe_x, stripe_y))
                for stripe_y in self.stripe_range(
                    y,
                    self._stripe_height,
                    self._tile_height
                )
                for stripe_x in self.stripe_range(
                    x,
                    self._stripe_width,
                    self._tile_width
                )
            }
            # Loop over the stripes and get segments.
            for stripe_coordinate, stripe in stripes.items():
                # Calculate the tiles the segments in this stripe will span.
                tile_x_start = (
                    stripe_coordinate[0] *
                    self._stripe_width // self._tile_width
                )
                tile_y_start = (
                    stripe_coordinate[1] *
                    self._stripe_height // self._tile_height
                )
                tile_x_end = (
                    (stripe_coordinate[0] + 1)
                    * self._stripe_width // self._tile_width

                )
                tile_y_end = (
                    (stripe_coordinate[1] + 1)
                    * self._stripe_height // self._tile_height
                )

                tiles: List[Tile] = []
                for tile_y in range(tile_y_start, tile_y_end+1):
                    for tile_x in range(tile_x_start, tile_x_end):
                        try:
                            tiles.append(self.tiles[tile_x, tile_y])
                        except KeyError:
                            tile = Tile(
                                self._page.jpegheader,
                                self._tile_width,
                                self._tile_height,
                                self._header.components
                            )
                            self.tiles[tile_x, tile_y] = tile
                            tiles.append(tile)
                dc_offsets = [
                    tile.dc_offsets for tile in tiles
                ]
                print(tiles)
                # print(dc_offsets)
                # Send the tiles
                # as a list. get_segments can then insert the segment directly
                # into the tiles. The tile will give the current dc offset and
                # get_segments can update the tile dc_offset after insertion.

                segments = self._header.get_segments(
                    stripe,
                    self._tile_width,
                    dc_offsets
                )
                for tile_index, segment in enumerate(segments):
                    tiles[tile_index].add_segment(segment)

        return self.tiles[x, y].get_tile()
