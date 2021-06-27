from dataclasses import dataclass
from struct import unpack
from bitstring import BitArray
import io

from tifffile.tifffile import FileHandle

from ndpi_tiler.jpeg_tags import TAGS


@dataclass
class StreamPosition:
    byte: int
    bit: int

    def to_bits(self) -> int:
        return 8 * self.byte + self.bit


class Stream:
    """Convenience class for reading bits from bytes"""
    def __init__(
        self,
        data: bytes
    ) -> None:
        """Create a Stream from data. Offers read function for single,
        multiple or range of bits.

        Parameters
        ----------
        data: bytes
            Byte data to stream

        """
        self.data = self.remove_stuffing(bytearray(data))
        self._buffer = io.BytesIO(self.data)
        self._byte = self._read_byte()
        self._bit_pos = 0

    @staticmethod
    def remove_stuffing(data: bytearray) -> bytearray:
        stuffing_removed = False
        search_start = 0
        while not stuffing_removed:
            tag_index = data.find(bytes([0xFF, 0x00]), search_start)
            if tag_index != -1:
                del data[tag_index+1]
                search_start = tag_index+1
            else:
                stuffing_removed = True
        return data

    @property
    def pos(self) -> StreamPosition:
        """The current steam position."""
        return StreamPosition(self.bit_pos // 8, self.bit_pos % 8)

    @property
    def byte_pos(self) -> int:
        """The current byte position (buffer read is one byte ahead)"""
        return self._buffer.tell() - 1

    @property
    def bit_pos(self) -> int:
        """The current bit posiont."""
        return 8 * self.byte_pos + self._bit_pos

    def _read_byte(self) -> int:
        return unpack('B', self._buffer.read(1))[0]

    def _read_bit(self) -> int:
        """Return a bit from the buffer. If passing a byte, and the next byte
        stuffed, skip it. Also check if the next byte is a tag and flag for
        skip."""

        bit = (self._byte >> (7-self._bit_pos)) & 0b1
        self._bit_pos += 1
        if self._bit_pos == 8:
            self._bit_pos = 0
            self._byte = self._read_byte()
        return bit

    def read(self, count: int = 1) -> int:
        """Read count bits and return the unsigned integer interpretation"""
        value = 0
        for i in range(count):
            value = 2*value + self._read_bit()
        return value

    def seek(self, position: int) -> None:
        """Seek to bit posiion in stream."""
        if position // 8 != self.byte_pos:
            self._buffer.seek(position // 8)
            self._byte = self._read_byte()
        self._bit_pos = position % 8

    def skip(self, skip_length: int) -> None:
        """Skip length of bits."""
        skip_to = self.bit_pos + skip_length
        # Are we skipping to another byte?
        self.seek(skip_to)

    def close(self) -> None:
        self._buffer.close()
