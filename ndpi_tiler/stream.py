from struct import unpack

from tifffile.tifffile import FileHandle

from ndpi_tiler.jpeg_tags import TAGS


class Stream:
    """Convenience class for reading bits from byte stuffed bytes."""
    def __init__(self, fh: FileHandle, offset: int) -> None:
        """Create a Stream from data. Offers read function for single,
        multiple or range of bits.

        Parameters
        ----------
        data: bytes
            Bytes to stream.

        """
        print(f"offset is {offset}")
        self._buffer = fh
        self._offset = offset
        self._bit_pos = 0
        self._next_byte_is_stuffed = False
        self.seek(offset*8)
        print(f"pos is {self.pos} byte is {hex(self._byte)}")

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

    def _read_bit(self) -> int:
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
        # position += 8 * self._offset
        if position // 8 != self.byte_pos:
            self._buffer.seek(position // 8)
            self._byte = self._read_byte()
        self._bit_pos = position % 8

    def skip(self, skip_length: int) -> None:
        """Skip length of bits. Handles byte stuffing."""
        skip_to = self.pos + skip_length
        # Are we skipping to another byte?
        self.seek(skip_to)
