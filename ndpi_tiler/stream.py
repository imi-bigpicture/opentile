from dataclasses import dataclass
from typing import List, Tuple

from bitstring import BitArray, Bits, ConstBitStream

from ndpi_tiler.jpeg_tags import TAGS


@dataclass
class McuBlock:
    position: int
    amplitude: int


@dataclass
class Mcu:
    blocks: List[McuBlock]
    @property
    def start(self) -> int:
        return self.blocks[0].position

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
        self._buffer = ConstBitStream(data)
        self._total_read_bits: int = 0
        self._next_byte_is_stuffed = False

    @property
    def pos(self) -> int:
        """The current bit position."""
        return self._buffer.pos

    # @property
    # def pos(self) -> StreamPosition:
    #     return StreamPosition.from_bits(self._total_read_bits)

    def _check_for_tag(self):
        position = self._buffer.pos
        next_byte = self._buffer.read('uint:8')
        if next_byte == TAGS['tag']:
            tag = self._buffer.read('uint:8')
            if tag != TAGS['stuffing']:
                raise ValueError(f"tag at position {self.pos}")
            self._next_byte_is_stuffed = True
        self._buffer.pos = position


    def read_bit(self) -> int:
        """Return a bit from the buffer. If the stored byte has been read, read
        a new one."""
        if self._buffer.pos % 8 == 0:
            if self._next_byte_is_stuffed:
                self._buffer.pos += 8
                self._next_byte_is_stuffed = False
            self._check_for_tag()
        bit = self._buffer.read('uint:1')
        return bit

    def read_bits(self, count: int) -> int:
        """Read multiple bits from the buffer."""
        bits = [self.read_bit() for i in range(count)]
        value = 0
        for bit in bits:
            value = 2*value + bit
        return value

    def seek(self, position: int) -> None:
        byte_pos = max(8 * (position // 8), 8)
        bit_pos = position - byte_pos
        self._buffer.pos = byte_pos - 8
        previous_byte = self._buffer.read('uint:8')
        if previous_byte == TAGS['tag']:
            tag = self._buffer.read('uint:8')
            if tag != TAGS['stuffing']:
                raise ValueError(f"tag at position {self.pos}")

            self._buffer.pos += bit_pos
        else:
            self._buffer.pos = position


    def skip(self, skip_length: int) -> None:
        skip_to = self.pos + skip_length
        self.seek(skip_to)


    def _read_segment(
        self,
        start: int,
        end: int
    ) -> BitArray:
        buffer_pos = self._buffer.pos
        self.seek(start)
        segment_bits = BitArray(self._buffer.read(end-start))
        self._buffer.pos = buffer_pos
        return segment_bits

    def mcu_to_bits(
        self,
        mcu: Mcu,
        mcu_end: int
    ):
        pass


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
