import io
from typing import Tuple

from bitstring import ConstBitStream
from ndpi_tiler.jpeg_tags import BYTE_TAG, BYTE_STUFFING

class Stream:
    """Convenience class for reading bits from byte stuffed bytes."""
    def __init__(self, data: bytes) -> None:
        """Create a Stream from data. Reads byte by byte from buffer to check
        for tags and byte stuffing. Offers read function for single or multiple
        bits.

        Parameters
        ----------
        data: bytes
            Bytes to stream.

        """
        self._buffer = io.BytesIO(data)
        self._byte = ConstBitStream()
        self._total_read_bits: int = 0

    # @property
    # def pos(self) -> Tuple[int, int]:
    #     """The current bit position."""
    #     return (self._buffer.tell(), self._byte.pos)

    @property
    def pos(self) -> Tuple[int, int]:
        bytes = self._total_read_bits // 8
        bits = self._total_read_bits - bytes * 8
        return (
            bytes,
            bits
        )

    def _read_byte(self) -> bytes:
        """Return next byte from buffer. If read byte is a tag (0xFF), check if
        the coming byte is byte stuffing (0x00) or a tag. Raise ValueError when
        encountering a tag, but this will change."""
        next_byte = self._buffer.read(1)
        # print(f"read byte {next_byte.hex()}")
        if next_byte == BYTE_TAG:
            tag = self._buffer.read(1)
            print(f"tag {tag.hex()}")
            if tag != BYTE_STUFFING:
                raise ValueError(f"tag at position {self.pos}")
        return next_byte

    def read_bit(self) -> int:
        """Return a bit from the buffer. If the stored byte has been read, read
        a new one."""
        if self._byte.pos == 0:
            self._byte = ConstBitStream(self._read_byte())

        bit = self._byte.read('uint:1')
        if self._byte.pos == 8:
            self._byte.pos = 0
        self._total_read_bits += 1
        return bit

    def read_bits(self, count: int) -> int:
        """Read multiple bits from the buffer."""
        bits = [self.read_bit() for i in range(count)]
        value = 0
        for bit in bits:
            value += 2*value + bit
        return value