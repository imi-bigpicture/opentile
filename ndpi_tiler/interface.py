from typing import Dict, Tuple, List

from tifffile import FileHandle, TiffPage
from bitarray import bitarray
from ndpi_tiler.jpeg import Component, JpegBuffer, JpegHeader, JpegSegment, Dc


class Tile:
    def __init__(
        self,
        header: JpegHeader,
        tile_width: int,
        tile_height: int,
        components: Dict[int, Component]
    ):
        self.header = header
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.bits = bitarray()
        self.tile: bytes = None
        self.dc_offsets = Dc.zero(list(components.keys()))

    def add_segment(self, segment: JpegSegment):
        self.bits += segment.data

    def get_tile(self) -> bytes:
        if self.tile is not None:
            return self.tile

        # Pad scan with ending 1, convert to bytes and add stuffing
        scan_bytes = JpegBuffer.convert_to_bytes(self.bits)

        # Wrap scan with modified header and end of image tag.
        self.tile = self.header.wrap_scan(
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
        self._header = JpegHeader(self._page.jpegheader)
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

            # Get the needed stripes
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
                # Tiles this stripe will cover.
                tile_range = {
                    (tile_x, tile_y)
                    for tile_y in self.stripe_range(
                        stripe_coordinate[1],
                        self._tile_height,
                        self._stripe_height
                    )
                    for tile_x in self.stripe_range(
                        stripe_coordinate[0],
                        self._tile_width,
                        self._stripe_width
                    )
                }

                # Get or make the tile(s) this stripe will cover
                tiles: List[Tile] = []
                for (tile_x, tile_y) in tile_range:
                    try:
                        tiles.append(self.tiles[tile_x, tile_y])
                    except KeyError:
                        tile = Tile(
                            self._header,
                            self._tile_width,
                            self._tile_height,
                            self._header.components
                        )
                        self.tiles[tile_x, tile_y] = tile
                        tiles.append(tile)

                # Get the segments, use the dc offsets of the tile(s)
                segments = self._header.get_segments(
                    stripe,
                    self._tile_width,
                    [tile.dc_offsets for tile in tiles]
                )

                # Insert the segments into the tile(s)
                for tile_index, segment in enumerate(segments):
                    tiles[tile_index].add_segment(segment)

        return self.tiles[x, y].get_tile()
