import io
import struct
from pathlib import Path
from typing import List, Tuple

from PIL import Image
from tifffile import FileHandle, TiffFile, TiffPage, TiffPageSeries

from ndpi_tiler.jpeg import JpegHeader, JpegScan


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
        with io.BytesIO() as buffer:
            buffer.write(self.manupulate_header(size))
            buffer.write(scan)
            buffer.write(bytes([0xFF, 0xD9]))  # End of Image Tag
            return buffer.getvalue()

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

    def manupulate_header(self, size: Tuple[int, int]) -> bytes:
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
        index = self.find_start_of_frame(self._page.jpegheader)
        with io.BytesIO() as buffer:
            buffer.write(self._page.jpegheader)
            buffer.seek(index+4)
            buffer.write(struct.pack(">H", size[1]))
            buffer.write(struct.pack(">H", size[0]))
            manupulated_header = buffer.getvalue()
        return manupulated_header

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


class NdpiStrip:
    def __init__(self):
        pass


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

        self.strips: List[NdpiStrip] = []


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
