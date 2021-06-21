from dataclasses import dataclass
import io
from typing import Tuple, List
import math
from bitstring import BitStream, ConstBitArray, ConstBitStream, Bits, BitArray
from numpy import byte
from ndpi_tiler.jpeg_tags import BYTE_TAG, BYTE_STUFFING

@dataclass
class StreamPosition:
    byte: int
    bit: int

    @classmethod
    def from_bits(cls, bits: int) -> 'StreamPosition':
        return cls(
            bits // 8,
            bits % 8
        )

    def __add__(self, value) -> 'StreamPosition':
        if isinstance(value, StreamPosition):
            position_to_add = self.from_bits(self.bit + value.bit)
            print(f"add {value} to {self} {position_to_add} {self.byte} {value.byte}")
            return StreamPosition(
                byte=self.byte + value.byte + position_to_add.byte,
                bit=position_to_add.bit
            )
@dataclass
class McuBlock:
    position: StreamPosition
    amplitude: int


@dataclass
class Mcu:
    blocks: List[McuBlock]
    @property
    def start(self) -> StreamPosition:
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
        self._buffer = io.BufferedReader(io.BytesIO(data))
        self._byte = ConstBitStream()
        self._total_read_bits: int = 0

        self._read_next_byte()


    @property
    def pos(self) -> StreamPosition:
        """The current bit position."""
        return StreamPosition(self._buffer.tell() - 1, self._byte.pos)

    # @property
    # def pos(self) -> StreamPosition:
    #     return StreamPosition.from_bits(self._total_read_bits)


    def _read_next_byte(self):
        """Read new byte. If read byte is a tag (0xFF), check if
        the coming byte is byte stuffing (0x00) or a tag. Raise ValueError when
        encountering a tag, but this will change."""
        next_byte = self._buffer.read(1)
        self._byte = ConstBitStream(next_byte)
        if next_byte == BYTE_TAG:
            tag = self._buffer.read(1)
            if tag != BYTE_STUFFING:
                raise ValueError(f"tag at position {self.pos}")

    def read_bit(self) -> int:
        """Return a bit from the buffer. If the stored byte has been read, read
        a new one."""
        bit = self._byte.read('uint:1')
        if self._byte.pos == 8:
            self._read_next_byte()
        self._total_read_bits += 1
        print(bit)
        return bit

    def read_bits(self, count: int) -> int:
        """Read multiple bits from the buffer."""
        bits = [self.read_bit() for i in range(count)]
        value = 0
        for bit in bits:
            value = 2*value + bit
        return value

    def seek(self, position: StreamPosition) -> None:
        if(position.byte != self.pos.byte):
            print(f"{position.byte} not same as {self.pos.byte}")
            if position.byte > 0:
                print(f"{position.byte} larger than 0")

                self._buffer.seek(position.byte-1)
                previous_byte = self._buffer.read(1)
                print(f"previous byte {previous_byte.hex()}")

                if previous_byte == BYTE_TAG:
                    print("is a tag")
                    tag = self._buffer.read(1)
                    print(f"next byte is {tag.hex()}")
                    if tag != BYTE_STUFFING:
                        print("is stuffing")
                        raise ValueError("Unexpected tag")
            else:
                print(f"{position.byte} is 0")
                self._buffer.seek(position.byte)

            print(f"reading new byte at {self._buffer.tell()}")
            self._read_next_byte()
            print(f"buffer is now at new byte at {self._buffer.tell()}")


        self._byte.pos = position.bit


    def skip(self, bit_length: int) -> None:
        # print(f"bits to skip {StreamPosition.from_bits(bit_length)}")
        # print(f"skip from {self.pos}")
        skip_to = self.pos + StreamPosition.from_bits(bit_length)
        print(f"skip to {skip_to}")

        self.seek(skip_to)
        # print(f"now at {self.pos}")



    def _read_segment(
        self,
        start: StreamPosition,
        end: StreamPosition
    ) -> BitArray:
        buffer_pos = self._buffer.tell()
        self._buffer.seek(start.byte)
        bits_to_read = end.byte-start.byte + math.ceil(end.bit/8)
        segment_bytes = self._buffer.read(bits_to_read)
        bit_buffer = ConstBitStream(segment_bytes)
        bit_buffer.pos = start.bit
        bits_to_read = (end.byte - start.byte) * 8 + (end.bit - start.bit)
        segment_bits = BitArray(bit_buffer.read(f'bits:{bits_to_read}'))
        self._buffer.seek(buffer_pos)
        return segment_bits

    def mcu_to_bits(
        self,
        mcu: Mcu,
        mcu_end: StreamPosition
    ):
        pass


    def create_segment_bytes(
        self,
        first_mcu: Mcu,
        start: StreamPosition,
        end: StreamPosition
    ) -> bytes:
        segment_bits = self._read_segment(first_mcu.start, end)
        rest_of_scan_bits = self._read_segment(start, end)
        segment_bits.append(rest_of_scan_bits)
        padding_bits = 8 - len(segment_bits) % 8
        segment_bits.append(Bits(f'{padding_bits}*0b1'))
        return segment_bits.bytes
