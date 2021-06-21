import io
from pathlib import Path
from typing import Dict, List, Optional, OrderedDict, Set, Tuple, Union
import struct

from PIL import Image
from tifffile import FileHandle, TiffFile, TiffPage, TiffPageSeries
from .jpeg import JpegHeader, JpegScan


class NdpiPage:
    def __init__(self, page: TiffPage, fh: FileHandle):
        self._page = page
        self._fh = fh

    def get_stripe_byte_range(self, x, y) -> Tuple[int, int]:
        cols = self._page.chunked[1]
        stripe = x + y * cols
        offset = self._page.dataoffsets[stripe]
        count = self._page.databytecounts[stripe]
        return (offset, count)

    def read_stripe(self, offset, count) -> bytes:
        self._fh.seek(offset)
        stripe = self._fh.read(count)
        return stripe

    def wrap_scan(self, stripe: bytes, size: Tuple[int, int]) -> bytes:
        if self._page.jpegheader is None:
            return stripe
        # header = JpegHeader(self.manupulate_header(size))
        # print(self._page.jpegheader.hex())
        # JpegScan(header, stripe)
        with io.BytesIO() as buffer:
            buffer.write(self.manupulate_header(size))
            buffer.write(stripe)
            buffer.write(bytes([0xFF, 0xD9]))  # End of Image
            return buffer.getvalue()

    def get_encoded_strip(self, x, y) -> bytes:
        offset, count = self.get_stripe_byte_range(x, y)
        stripe = self.read_stripe(offset, count)
        # print(stripe.hex())
        # stripe = self.wrap_scan(stripe)
        return stripe

    @staticmethod
    def find_start_of_frame(header: bytes) -> int:
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
        index = self.find_start_of_frame(self._page.jpegheader)
        # print(f"orginal header {self._page.jpegheader.hex()}")

        with io.BytesIO() as buffer:
            buffer.write(self._page.jpegheader)
            buffer.seek(index+4)
            buffer.write(struct.pack(">H", size[1]))
            buffer.write(struct.pack(">H", size[0]))
            manupulated_header = buffer.getvalue()
        # print(f"manupulated header {manupulated_header.hex()}")
        return manupulated_header

    def stitch_tiles(
        self,
        pos: Tuple[int, int],
        size: Tuple[int, int]
    ) -> Image:
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

        # print(f"tile bytes {tile_bytes.hex()}")
        return Image.open(io.BytesIO(stripe))


class NdpiTiler:
    def __init__(self, path: Path) -> None:
        self.tif = TiffFile(path)
        self.__enter__()

    def __enter__(self):
        return self

    def __exit__(self) -> None:
        self.tif.close()

    def stitch_tiles(
        self,
        series: int,
        level: int,
        pos: Tuple[int, int],
        size: Tuple[int, int]
    ) -> Image:
        tiff_series: TiffPageSeries = self.tif.series[series]
        tiff_level: TiffPageSeries = tiff_series.levels[level]
        page = NdpiPage(tiff_level.pages[0], self.tif.filehandle)
        return page.stitch_tiles(pos, size)

    def close(self) -> None:
        self.tif.close()