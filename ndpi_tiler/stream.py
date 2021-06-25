import io
from struct import unpack
from bitstring import BitArray, Bits, ConstBitStream

from ndpi_tiler.jpeg_tags import TAGS


class Stream:
    """Convenience class for reading bits from byte stuffed bytes."""
    def __init__(self, data: bytes) -> None:
        """Create a Stream from data. Offers read function for single,
        multiple or range of bits.

        Parameters
        ----------
        data: bytes
            Bytes to stream.

        """
        self._buffer = io.BytesIO(data)
        self._bit_pos = 0
        self._next_byte_is_stuffed = False
        self._byte = self._read_byte()

    @property
    def byte_pos(self) -> int:
        return self._buffer.tell() - 1

    @property
    def pos(self) -> int:
        """The current bit position."""
        return 8 * self.byte_pos + self._bit_pos

    def _read_byte(self) -> int:
        if self._next_byte_is_stuffed:
            self._buffer.seek(1, 1)
        byte = unpack('B', self._buffer.read(1))[0]
        if byte == TAGS['tag']:
            tag = unpack('B', self._buffer.read(1))[0]
            if tag != TAGS['stuffing']:
                raise ValueError(f"tag {hex(tag)} at position {self.pos}")
            self._buffer.seek(-1, 1)
            self._next_byte_is_stuffed = True
        else:
            self._next_byte_is_stuffed = False
        return byte

    def _read_bit(self) -> Bits:
        """Return a bit from the buffer. If passing a byte, and the next byte
        stuffed, skip it. Also check if the next byte is a tag and flag for
        skip."""

        if self._bit_pos == 8:
            self._byte = self._read_byte()
            self._bit_pos = 0
        bit = (self._byte >> (7-self._bit_pos)) & 0b1
        self._bit_pos += 1

        return bit

    def read(self, count: int = 1) -> int:
        """Read count bits and return the unsigned integer interpretation"""
        value = 0
        for i in range(count):
            value = 2*value + self._read_bit()
        return value

    def seek(self, position: int) -> None:
        """Seek to bit posiion in stream. Does handle byte stuffing?."""
        if position // 8 != self.byte_pos:
            self._buffer.seek(position // 8)
            self._byte = self._read_byte()
        self._bit_pos = position % 8

    def skip(self, skip_length: int) -> None:
        """Skip length of bits. Handles byte stuffing."""
        skip_to = self.pos + skip_length
        # Are we skipping to another byte?
        self.seek(skip_to)

    def read_segment(
        self,
        start: int,
        end: int
    ) -> BitArray:
        buffer_pos = self.pos
        self.seek(start)
        segment_bits = BitArray([self._read_bit() for bit in range(end-start-8)])
        self.seek(buffer_pos)
        return segment_bits

    @classmethod
    def to_bytes(
        bit_data: BitArray
    ) -> bytes:
        padding_bits = 8 - len(bit_data) % 8
        return bit_data.append(Bits(f'{padding_bits}*0b1'))

