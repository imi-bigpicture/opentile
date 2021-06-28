import io
import struct
from struct import unpack
from pathlib import Path
from typing import Generator, List, Tuple, Dict, Optional

from bitarray import bitarray
from PIL import Image
from tifffile import FileHandle, TiffFile, TiffPage, TiffPageSeries

from ndpi_tiler.jpeg import JpegHeader, JpegSegment
from ndpi_tiler.jpeg_tags import TAGS


class NdpiPage:
    """Class for working with ndpi page (typically a level)"""
    def __init__(self, page: TiffPage, fh: FileHandle) -> None:
        """Initialize a NdpiPage from a TiffPage and Filehandle.

        Parameters
        ----------
        page: TiffPage
            Page to use.
        fh: FileHandle
            FileHandle to use

        """
        self._page = page
        self._fh = fh
        self._header = JpegHeader.from_bytes(self._page.jpegheader)

    def get_segments(
        self,
        x: int,
        y: int,
        scan_width: int
    ) -> List[JpegSegment]:
        stripe = self.get_encoded_stripe(x, y)
        segments = self._header.get_segments(stripe, scan_width)
        return segments

    def get_stripe_byte_range(self, x: int, y: int) -> Tuple[int, int]:
        """Return stripe offset and length from stripe position.

        Parameters
        ----------
        x: int
            X position of stripe.
        y: int
            Y position of stripe.

        Returns
        ----------
        Tuple[int, int]:
            stripe offset and length
        """
        cols = self._page.chunked[1]
        stripe = x + y * cols
        offset = self._page.dataoffsets[stripe]
        count = self._page.databytecounts[stripe]
        return (offset, count)

    def read_stripe(self, offset: int, length: int) -> bytes:
        """Read stripe scan data from page at offset.

        Parameters
        ----------
        offset: int
            Offset to stripe to read.
        length: int
            Length of stripe to read.

        Returns
        ----------
        bytes:
            Read stripe.
        """
        self._fh.seek(offset)
        stripe = self._fh.read(length)
        return stripe

    def wrap_scan(self, scan: bytes, size: Tuple[int, int]) -> bytes:
        """Wrap scan data with manipulated header and end of image tag.

        Parameters
        ----------
        scan: bytes
            Scan data to wrap.
        size: Tuple[int, int]
            Pixel size of scan.

        Returns
        ----------
        bytes:
            Scan wrapped in header as bytes.
        """
        if self._page.jpegheader is None:
            return scan

        image = self.manupulate_header(size)
        image += scan
        image += bytes([0xFF, 0xD9])
        return image

    def get_encoded_stripe(self, x: int, y: int) -> bytes:
        """Return stripe at position as bytes.

        Parameters
        ----------
        x: int
            X position of stripe.
        y: int
            Y position of stripe.

        Returns
        ----------
        bytes:
            stripe as bytes.
        """
        offset, count = self.get_stripe_byte_range(x, y)
        stripe = self.read_stripe(offset, count)
        return stripe

    @staticmethod
    def find_start_of_frame(header: bytes) -> int:
        """Return offset for start of frame tag in header.

        Parameters
        ----------
        header: bytes
            Header bytes.

        Returns
        ----------
        int:
            Offset to start of frame tag.
        """
        index = 0
        length = 1
        found_tag = False
        while index + length < len(header):
            if found_tag and header[index:index+length] == bytes(b'\xc0'):
                return index
            else:
                found_tag = False
            if header[index:index+length] == bytes(b'\xff'):  # JPEG tag
                found_tag = True
            index += length

    def find_tag(
        self,
        data: bytes,
        tag: int
    ) -> Tuple[Optional[int], Optional[int]]:
        """Return first index and length of payload of tag in buffer."""
        index = data.find(tag.to_bytes(2, 'big'))
        if index != -1:
            (length, ) = unpack('>H', data[index+2:index+4])
            return index, length
        return None, None

    def manupulate_header(self, size: Tuple[int, int]) -> bytearray:
        """Manipulate pixel size (width, height) of page header.

        Parameters
        ----------
        size: Tuple[int, int]
            Pixel size to insert into header.

        Returns
        ----------
        bytes:
            Manupulated header.
        """
        header = bytearray(self._page.jpegheader)
        start_of_scan_index, length = self.find_tag(
            header, TAGS['start of frame']
        )
        if start_of_scan_index is None:
            raise ValueError("Start of scan tag not found in header")
        size_index = start_of_scan_index+5
        header[size_index:size_index+2] = struct.pack(">H", size[1])
        header[size_index+2:size_index+4] = struct.pack(">H", size[0])

        reset_interval_index, length = self.find_tag(
            header, TAGS['restart interval']
        )
        if reset_interval_index is not None:
            del header[reset_interval_index:reset_interval_index+length+2]

        return header

    def stitch_tiles(
        self,
        pos: Tuple[int, int],
        size: Tuple[int, int]
    ) -> Image:
        """Stitch tiles (stripes) together to form image.

        Parameters
        ----------
        pos: Tuple[int, int]
            Position of stripe to start stitching from.
        size: Tuple[int, int]
            Number of stripe to stitch together.

        Returns
        ----------
        Image:
            Stitched image.
        """

        tile_width = self._page.tilewidth
        tile_height = self._page.tilelength

        x_pos = pos[0]
        y_pos = pos[1]
        width = size[0]
        height = size[1]

        image_size = (width*tile_width, height*tile_height)
        stripe_index = 0
        with io.BytesIO() as stripe_buffer:
            for x in range(x_pos, x_pos+width):
                for y in range(y_pos, y_pos+height):
                    stripe = self.get_encoded_stripe(x, y)
                    # Each stripe has a RST marker (0xFF, 0xDn, n 0-7) at end.
                    # Do not include last RST byte (0-7), use new from index
                    stripe_buffer.write(stripe[:-1])
                    last_rst_byte = struct.pack(">B", 208+stripe_index)
                    stripe_buffer.write(last_rst_byte)
                    stripe_index = (stripe_index + 1) % 8
            stripe = self.wrap_scan(stripe_buffer.getvalue(), image_size)

        return Image.open(io.BytesIO(stripe))


class NdpiStripCache:
    def __init__(
        self,
        page: NdpiPage,
        tile_width: int,
        tile_height: int
    ):
        """Cache for ndpi stripes, with functions to produce tiles of specified
        with and height.

        Parameters
        ----------
        page: NdpiPage
            Page to cache and tile.
        tile_width: int
            Tile width to cache and produce.
        tile_height: int
            Tile heigh to cache and produce.

        """
        self.page = page
        self.stripe_width = page._header.width
        self.stripe_height = page._header.height
        self.stripe_cols = self.page._page.chunked[1]
        self.tile_width = tile_width
        self.tile_height = tile_height

        self.segments: Dict[(int, int), JpegSegment] = {}

    def get_tile(self, x: int, y: int) -> bytes:
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
        stripe_x_start = x * self.tile_width / self.stripe_width
        stripe_x_end = max(
            stripe_x_start + 1,
            (x+1) * self.tile_width // self.stripe_width
        )
        stripe_y_start = y * self.tile_height / self.stripe_height
        stripe_y_end = max(
            stripe_y_start + 1,
            (y+1) * self.tile_height // self.stripe_height
        )

        # If first stripe is not in cached segments, get the stripes
        if (stripe_x_start, stripe_y_start) not in self.segments.keys():
            self.segments = {}
            stripe_indices = [
                x + y * self.stripe_cols
                for y in range(int(stripe_y_start), int(stripe_y_end))
                for x in range(int(stripe_x_start), int(stripe_x_end))
            ]

            # Generator producing (stripe, stripe_index), requires patched
            # tifffile
            stripes: Generator[bytes, int] = self.page._fh.read_segments(
                offsets=self.page._page.dataoffsets,
                bytecounts=self.page._page.databytecounts,
                indices=stripe_indices,
            )

            # Loop over the stripes and get segments.
            for stripe, stripe_index in stripes:
                segments = self.page._header.get_segments(
                    stripe,
                    self.tile_width
                )
                # For each segment, insert into segment cache at pixel (x y)
                # position
                for segment_index, segment in enumerate(segments):
                    segment_x = (
                        (stripe_index % self.stripe_cols) * self.stripe_width
                        + segment_index*self.tile_width
                    )
                    segment_y = (
                        (stripe_index // self.stripe_cols) * self.stripe_height
                    )
                    self.segments[(segment_x, segment_y)] = segment

        # bitarray for concatenating bit segments
        scan = bitarray()
        # Loop through the needed segments
        for segment_y in range(
            y*self.tile_height,
            (y+1)*self.tile_height,
            min(self.stripe_height, self.tile_width)
        ):
            for segment_x in range(
                x*self.tile_width,
                (x+1)*self.tile_width,
                min(self.stripe_width, self.tile_width)
            ):

                segment = self.segments[(segment_x, segment_y)]
                segment_bits = bitarray()
                segment_bits.frombytes(bytes(segment.data))
                # Here the segment first DC values should be modified

                # Append segment to scan
                scan += segment_bits[
                    segment.start.to_bits():segment.end.to_bits()
                ]

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
        tile = self.page.wrap_scan(
            scan_bytes,
            (self.tile_width, self.tile_height)
        )

        return tile


class NdpiTiler:
    """Class to convert stripes in a ndpi file, opened with TiffFile,
    into square tiles."""
    def __init__(self, path: Path) -> None:
        """Initialize by opening provided ndpi file in path as TiffFile.

        Parameters
        ----------
        path: Path
            Path to ndpi file to open

        """
        self.tif = TiffFile(path)
        self.__enter__()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback) -> None:
        self.close()

    def stitch_tiles(
        self,
        series: int,
        level: int,
        pos: Tuple[int, int],
        size: Tuple[int, int]
    ) -> Image:
        """Stitch tiles (stripes) together to form image.

        Parameters
        ----------
        series: int
            Series to stitch from.
        level: int
            Level to stitch from.
        pos: Tuple[int, int]
            Position of stripe to start stitching from.
        size: Tuple[int, int]
            Number of stripe to stitch together.

        Returns
        ----------
        Image:
            Stitched image.
        """

        tiff_series: TiffPageSeries = self.tif.series[series]
        tiff_level: TiffPageSeries = tiff_series.levels[level]
        page = NdpiPage(tiff_level.pages[0], self.tif.filehandle)
        return page.stitch_tiles(pos, size)

    def close(self) -> None:
        self.tif.close()
