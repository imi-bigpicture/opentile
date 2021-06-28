from dataclasses import dataclass
from ndpi_tiler.huffman import HuffmanTable

from ndpi_tiler.jpeg_tags import BYTE_TAG_STUFFING


@dataclass
class BufferPosition:
    byte: int
    bit: int

    def to_bits(self) -> int:
        return 8 * self.byte + self.bit


class JpegBuffer:
    """Convenience class for reading bits from Jpeg data."""
    def __init__(
        self,
        data: bytes
    ) -> None:
        """Create a JpegBuffer from data. Offers read function for single,
        multiple or range of bits.

        Parameters
        ----------
        data: bytes
            Byte data to buffer.

        """
        self._data = self.remove_stuffing(data)
        self._bit_pos = 0
        self._byte_pos = 0
        self._byte = self._read_byte()

    @staticmethod
    def remove_stuffing(data: bytes) -> bytes:
        data = bytearray(data)
        stuffing_removed = False
        search_start = 0
        while not stuffing_removed:
            tag_index = data.find(BYTE_TAG_STUFFING, search_start)
            if tag_index != -1:
                del data[tag_index+1]
                search_start = tag_index+1
            else:
                stuffing_removed = True
        return bytes(data)

    @property
    def pos(self) -> BufferPosition:
        """The current buffer position."""
        return BufferPosition(self.bit_pos // 8, self.bit_pos % 8)

    @property
    def byte_pos(self) -> int:
        """The current byte position (buffer read is one byte ahead)"""
        return self._byte_pos - 1

    @property
    def bit_pos(self) -> int:
        """The current bit posiont."""
        return 8 * self.byte_pos + self._bit_pos

    def _read_byte(self) -> int:
        byte = self._data[self._byte_pos]
        self._byte_pos += 1
        return byte

    def _read_bit(self) -> int:
        """Return a bit from the buffer."""

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

    def read_variable_length(self, table: HuffmanTable) -> int:
        """Read variable length using huffman table"""
        symbol = self.read()
        length = 1
        # We should check for length here, max is 16?
        code = table.decode(symbol, length)
        while code is None:
            symbol = 2*symbol + self.read()
            length += 1
            code = table.decode(symbol, length)
        return code

    def seek(self, position: int) -> None:
        """Seek to bit posiion in stream."""
        if position // 8 != self.byte_pos:
            self._byte_pos = position // 8
            self._byte = self._read_byte()
        self._bit_pos = position % 8

    def skip(self, skip_length: int) -> None:
        """Skip length of bits."""
        skip_to = self.bit_pos + skip_length
        # Are we skipping to another byte?
        self.seek(skip_to)

    def close(self) -> None:
        pass
        # self._buffer.close()
