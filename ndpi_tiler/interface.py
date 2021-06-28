from typing import DefaultDict, Generator, Dict, Tuple, List

from collections import defaultdict

from bitarray import bitarray
from tifffile import FileHandle, TiffPage

from ndpi_tiler.jpeg import JpegHeader, JpegSegment
from ndpi_tiler.jpeg_tags import TAGS


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

        self.segments: Dict[(int, int), JpegSegment] = {}
        self.tile_dc_offsets: DefaultDict[(int, int), List[int]] = (
            defaultdict(list)
        )

    @property
    def stripe_size(self) -> Tuple[int, int]:
        return (self._header.width, self._header.height)

    @property
    def tile_size(self) -> Tuple[int, int]:
        return (self._tile_width, self._tile_height)

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
        # The range of stripes we need
        stripe_x_start = x * self._tile_width / self._stripe_width
        stripe_x_end = max(
            stripe_x_start + 1,
            (x+1) * self._tile_width // self._stripe_width
        )
        stripe_y_start = y * self._tile_height / self._stripe_height
        stripe_y_end = max(
            stripe_y_start + 1,
            (y+1) * self._tile_height // self._stripe_height
        )
        print(stripe_x_start)
        print(stripe_x_end)
        tile_x_start = self._tile_width * stripe_x_start
        tile_y_start = self._tile_height * stripe_y_start
        print(tile_x_start, tile_y_start)
        # If first stripe is not in cached segments, get the stripes
        if (tile_x_start, tile_y_start) not in self.segments.keys():
            print(f"fetching stripes starting at {stripe_x_start, stripe_y_start}")
            self.segments = {}
            stripe_indices = [
                x + y * self.stripe_cols
                for y in range(int(stripe_y_start), int(stripe_y_end))
                for x in range(int(stripe_x_start), int(stripe_x_end))
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
                    self.segments[(segment_x, segment_y)] = segment

        # bitarray for concatenating bit segments
        scan = bitarray()
        # Loop through the needed segments
        for segment_y in range(
            y*self._tile_height,
            (y+1)*self._tile_height,
            min(self._stripe_height, self._tile_width)
        ):
            for segment_x in range(
                x*self._tile_width,
                (x+1)*self._tile_width,
                min(self._stripe_width, self._tile_width)
            ):

                segment = self.segments[(segment_x, segment_y)]
                # Here the segment first DC values should be modified

                # Append segment to scan
                scan += segment.first + segment.rest

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
        tile = JpegHeader.wrap_scan(
            self._page.jpegheader,
            scan_bytes,
            (self._tile_width, self._tile_height)
        )
        return tile
