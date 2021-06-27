import io
import struct
from struct import unpack
from pathlib import Path
from typing import Generator, List, Tuple, Dict, Optional

from bitarray import bitarray
from PIL import Image
from tifffile import FileHandle, TiffFile, TiffPage, TiffPageSeries

from ndpi_tiler.jpeg import JpegHeader, JpegScan, JpegSegment
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
        offset, length = self.get_stripe_byte_range(x, y)
        scan = JpegScan(self._header, self._fh, offset, length, scan_width)
        segments = scan.segments
        scan.close()
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
            Stripe offset and length
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

    def get_encoded_strip(self, x: int, y: int) -> bytes:
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
            Stripe as bytes.
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
        strip_index = 0
        with io.BytesIO() as strip_buffer:
            for x in range(x_pos, x_pos+width):
                for y in range(y_pos, y_pos+height):
                    stripe = self.get_encoded_strip(x, y)
                    # Each strip has a RST marker (0xFF, 0xDn, n 0-7) at end.
                    # Do not include last RST byte (0-7), use new from index
                    strip_buffer.write(stripe[:-1])
                    last_rst_byte = struct.pack(">B", 208+strip_index)
                    strip_buffer.write(last_rst_byte)
                    strip_index = (strip_index + 1) % 8
            stripe = self.wrap_scan(strip_buffer.getvalue(), image_size)

        return Image.open(io.BytesIO(stripe))


class NdpiStripCache:
    def __init__(
        self,
        page: NdpiPage,
        strip_width: int,
        strip_height: int,
        tile_width: int,
        tile_height: int
    ):
        self.page = page
        self.strip_width = strip_width
        self.strip_height = strip_height
        self.tile_width = tile_width
        self.tile_height = tile_height

        self.segments: Dict[(int, int), JpegSegment] = {}

    def get_tile(self, x: int, y: int) -> bytes:
        strip_x_start = x * self.tile_width / self.strip_width
        strip_x_end = max(
            strip_x_start + 1,
            (x+1) * self.tile_width // self.strip_width
        )
        strip_y_start = y * self.tile_height / self.strip_height
        strip_y_end = max(
            strip_y_start + 1,
            (y+1) * self.tile_height // self.strip_height
        )

        test_strip_position = (strip_x_start, strip_y_start)

        if test_strip_position not in self.segments.keys():
            self.segments = {}
            cols = self.page._page.chunked[1]
            strip_indices = [
                x + y * cols
                for y in range(int(strip_y_start), int(strip_y_end))
                for x in range(int(strip_x_start), int(strip_x_end))
            ]
            strip_byte_range = [
                self.page.get_stripe_byte_range(x, y)
                for y in range(int(strip_y_start), int(strip_y_end))
                for x in range(int(strip_x_start), int(strip_x_end))
            ]

            strips: Generator[bytes, int] = self.page._fh.read_segments(
                offsets=self.page._page.dataoffsets,
                bytecounts=self.page._page.databytecounts,
                indices=strip_indices,
                sort=False
            )
            for strip, strip_index in strips:
                scan = JpegScan(self.page._header, strip)
                segments = scan.segments
                for segment_index, segment in enumerate(segments):
                    segment_x = (
                        (strip_index % cols) * self.strip_width
                        + segment_index*self.tile_width
                    )
                    segment_y = (strip_index // cols) * self.strip_height
                    self.segments[(segment_x, segment_y)] = segment
        print(self.segments.keys())
        scan = bitarray()
        for segment_x in range(
            x*self.tile_width,
            (x+1)*self.tile_width,
            min(self.strip_width, self.tile_width)
        ):
            for segment_y in range(
                y*self.tile_height,
                (y+1)*self.tile_height,
                min(self.strip_height, self.tile_width)
            ):
                segment = self.segments[(segment_x, segment_y)]
                segment_bits = bitarray()
                segment_bits.frombytes(bytes(segment.data))
                # Modify segment bits before appending
                scan += segment_bits[0:segment.length.to_bits()]

        padding_bits = 7 - (len(scan) - 1) % 8
        scan += bitarray(padding_bits*[1])
        scan_bytes = bytearray(scan.tobytes())

        tag_index = None
        start_search = 0
        while tag_index != -1:
            tag_index = scan_bytes.find(0xFF, start_search)
            if tag_index != -1:
                scan_bytes.insert(tag_index+1, 0x00)
                start_search = tag_index+1

        tile = self.page.wrap_scan(
            scan_bytes,
            (self.tile_width, self.tile_height)
        )

        f = open("scan.jpeg", "wb")
        f.write(tile)
        f.close()
        # print(len(image))
        # print(image.hex())
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
