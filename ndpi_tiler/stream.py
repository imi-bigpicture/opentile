from dataclasses import dataclass
from typing import List

from bitstring import BitArray, Bits, ConstBitStream

from ndpi_tiler.jpeg_tags import TAGS


@dataclass
class McuBlock:
    """A component block of a Mcu, with bit position and dc amplitude"""
    position: int
    amplitude: int


@dataclass
class Mcu:
    """A Mcu, consisting of one or more blocks (components)"""
    blocks: List[McuBlock]

    @property
    def start(self) -> int:
        return self.blocks[0].position


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
        self._buffer = ConstBitStream(data)
        self._next_byte_is_stuffed = False

    @property
    def pos(self) -> int:
        """The current bit position."""
        return self._buffer.pos

    def _is_stuffed(self, position: int) -> None:
        """Check if next byte from position is stuffed."""
        next_byte_is_stuffed = False
        restore_position = self.pos
        byte_pos = 8 * (position // 8)
        self._buffer.pos = byte_pos
        next_byte = self._buffer.read('uint:8')
        if next_byte == TAGS['tag']:
            tag = self._buffer.read('uint:8')
            if tag != TAGS['stuffing']:
                raise ValueError(f"tag at position {self.pos}")
            next_byte_is_stuffed = True
        self._buffer.pos = restore_position
        return next_byte_is_stuffed

    def _read_bit(self) -> Bits:
        """Return a bit from the buffer. If passing a byte, and the next byte
        stuffed, skip it. Also check if the next byte is a tag and flag for
        skip."""
        if self.pos % 8 == 0:
            if self._next_byte_is_stuffed:
                self._buffer.pos += 8
                self._next_byte_is_stuffed = False
            self._next_byte_is_stuffed = self._is_stuffed(self.pos)
        bit = Bits(self._buffer.read('bits:1'))
        return bit

    def read(self, count: int = 1) -> int:
        """Read count bits and return the unsigned integer interpretation"""
        if count == 0:
            return 0
        bits = BitArray([self._read_bit() for i in range(count)])
        return bits.uint

    def seek(self, position: int) -> None:
        """Seek to bit posiion in stream. Does not handle byte stuffing."""
        self._buffer.pos = position

    def skip(self, skip_length: int) -> None:
        """Skip length of bits. Handles byte stuffing."""
        skip_to = self.pos + skip_length
        # Are we skipping to another byte?
        if skip_to // 8 != self.pos // 8:
            self._next_byte_is_stuffed = False
            # If not first byte
            if skip_to >= 8:
                # Check if this byte is stuffed
                if self._is_stuffed(skip_to - 8):
                    skip_to += 8
            if skip_to % 8 != 0:
                # Byte stuffing is checked on read if at first bit
                # The byte we skip to has been checked for stuffing, so
                # only check the next byte if we are not at first bit
                self._next_byte_is_stuffed = self._is_stuffed(skip_to)
        self.seek(skip_to)

    def read_segment(
        self,
        start: int,
        end: int
    ) -> BitArray:
        buffer_pos = self.pos
        self.seek(start)
        segment_bits = BitArray(self._buffer.read(end-start))
        self._buffer.pos = buffer_pos
        return segment_bits

    def create_segment_bytes(
        self,
        first_mcu: Mcu,
        start: int,
        end: int
    ) -> bytes:
        segment_bits = self._read_segment(first_mcu.start, end)
        rest_of_scan_bits = self._read_segment(start, end)
        segment_bits.append(rest_of_scan_bits)
        padding_bits = 8 - len(segment_bits) % 8
        segment_bits.append(Bits(f'{padding_bits}*0b1'))
        return segment_bits.bytes
