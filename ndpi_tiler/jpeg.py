import io
import struct
from dataclasses import dataclass
from struct import unpack
from typing import Dict, List, Optional, Tuple
from functools import lru_cache

from bitarray import bitarray, util

from ndpi_tiler.huffman import HuffmanTable, HuffmanTableIdentifier
from ndpi_tiler.jpeg_tags import BYTE_TAG, BYTE_TAG_STUFFING, TAGS
from ndpi_tiler.utils import split_byte_into_nibbles

MCU_SIZE = 8


@dataclass
class Component:
    name: str
    dc_table_id: HuffmanTableIdentifier
    ac_table_id: HuffmanTableIdentifier
    dc_table: HuffmanTable = None
    ac_table: HuffmanTable = None


class Dc:
    def __init__(self, values: Dict[str, int]):
        self.offsets = values

    def add(self, values: 'Dc') -> None:
        for name in self.names:
            self.offsets[name] += values[name]

    def remove(self, values: 'Dc') -> None:
        for name in self.names:
            self.offsets[name] -= values[name]

    def __getitem__(self, name: str) -> int:
        return self.offsets[name]

    def __setitem__(self, name: str, value: int) -> None:
        self.offsets[name] = value

    def __eq__(self, other) -> bool:
        if isinstance(other, Dc):
            for key in self.offsets:
                if not (key in other.offsets and self[key] == other[key]):
                    return False
            return True
        return NotImplemented

    def __str__(self) -> str:
        return str(self.offsets)

    def __repr__(self) -> str:
        return str(self)

    @classmethod
    def zero(cls, names: List[str]) -> 'Dc':
        return cls({name: 0 for name in names})

    @property
    def names(self) -> List[str]:
        return self.offsets.keys()


@dataclass
class JpegSegment:
    data: bitarray
    segment_delta: Dc
    tile_offset: Dc = None


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
        self._data = bitarray()
        self._data.frombytes(self.remove_stuffing(data))
        self._bit_pos = 0

    @staticmethod
    def remove_stuffing(data: bytes) -> bytes:
        return data.replace(BYTE_TAG_STUFFING, BYTE_TAG)

    @property
    def pos(self) -> int:
        """The current buffer position."""
        return self._bit_pos

    def _read_bit(self) -> int:
        """Return a bit from the buffer."""
        bit = self._data[self._bit_pos]
        self._bit_pos += 1
        return bit

    def read(self, count: int = 1) -> int:
        """Read count bits and return the unsigned integer interpretation"""
        # return util.ba2int(
        #     self._data[self._bit_pos:self._bit_pos+count],
        #     signed=False
        # )
        sum = 0
        for value in self._data[self._bit_pos:self._bit_pos+count]:
            sum = 2*sum + value
        self._bit_pos += count

        return sum

    def read_variable_length(self, table: HuffmanTable) -> int:
        """Read variable length using huffman table"""
        # node = table.root
        # while isinstance(node, HuffmanNode):
        #     node = node._nodes[self.read()]
        # return node.value
        symbol = self.read()
        length = 1
        code = table.decode(symbol, length)
        while code is None:
            symbol = 2*symbol + self.read()
            length += 1
            code = table.decode(symbol, length)
        return code

    def read_to_bitarray(
        self,
        start: int,
        end: int
    ) -> bitarray:
        return self._data[start:end]

    def seek(self, position: int) -> None:
        """Seek to bit posiion in stream."""
        self._bit_pos = position

    def skip(self, skip_length: int) -> None:
        """Skip length of bits."""
        skip_to = self.pos + skip_length
        self.seek(skip_to)


class JpegHeader:
    """Class for minimal parsing of jpeg header"""
    def __init__(
        self,
        width: int,
        height: int,
        components: List[Component],
        restart_interval: int = None
    ) -> None:

        self._components = {
            component.name: component for component in components
        }
        self._width = width
        self._height = height
        self._restart_interval = restart_interval
        self._mcu_count = height * width // (MCU_SIZE * MCU_SIZE)

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def components(self) -> Dict[str, Component]:
        return self._components

    @property
    def mcu_count(self) -> int:
        return self._mcu_count

    @property
    def restart_interval(self) -> int:
        if self._restart_interval is not None:
            return self._restart_interval
        return self.width * self.height // (MCU_SIZE*MCU_SIZE)

    @classmethod
    def from_bytes(cls, data: bytes) -> 'JpegHeader':
        """Parse jpeg header. Read markers from data and parse payload if
        huffman table(s) or start of frame. Ignore other markers (for now)

        Parameters
        ----------
        data: bytes
            Jpeg header in bytes.

        Returns
        ----------
        JpegHeader
            JpegHeader created from data.

        """
        width: int
        height: int
        components: List[Component] = []
        huffman_tables: Dict[HuffmanTableIdentifier, HuffmanTable] = {}

        with io.BytesIO(data) as buffer:
            marker = cls.read_marker(buffer)
            if not marker == TAGS['start of image']:
                raise ValueError("Expected start of image marker")
            marker = cls.read_marker(buffer)
            while marker is not None:
                if (
                    marker == TAGS['start of image'] or
                    marker == TAGS['end of image']
                ):
                    raise ValueError("Unexpected marker")
                payload = cls.read_payload(buffer)
                if marker == TAGS['huffman table']:
                    huffman_tables = cls.parse_huffman(payload, huffman_tables)
                elif marker == TAGS['start of frame']:
                    (width, height) = cls.parse_start_of_frame(payload)
                elif marker == TAGS['start of scan']:
                    components = cls.parse_start_of_scan(payload)
                elif marker == TAGS['restart interval']:
                    restart_interval = cls.parse_restart_interval(payload)
                else:
                    pass

                marker = cls.read_marker(buffer)
        if (
            huffman_tables == {} or
            width is None or height is None or
            components == []
        ):
            raise ValueError("missing tags")

        for component in components:
            component.dc_table = huffman_tables[component.dc_table_id]
            component.ac_table = huffman_tables[component.ac_table_id]

        return cls(
            width,
            height,
            components,
            restart_interval
        )

    @staticmethod
    def find_tag(
        header: bytes,
        tag: int
    ) -> Tuple[Optional[int], Optional[int]]:
        """Return first index and length of payload of tag in header."""
        index = header.find(tag.to_bytes(2, 'big'))
        if index != -1:
            (length, ) = unpack('>H', header[index+2:index+4])
            return index, length
        return None, None

    @classmethod
    def manupulate_header(cls, header: bytes, size: Tuple[int, int]) -> bytes:
        """Manipulate pixel size (width, height) of page header and
        remove reset interval marker.

        Parameters
        ----------
        size: Tuple[int, int]
            Pixel size to insert into header.

        Returns
        ----------
        bytes:
            Manupulated header.
        """
        start_of_scan_index, length = cls.find_tag(
            header, TAGS['start of frame']
        )
        if start_of_scan_index is None:
            raise ValueError("Start of scan tag not found in header")
        size_index = start_of_scan_index+5
        header = bytearray(header)
        header[size_index:size_index+2] = struct.pack(">H", size[1])
        header[size_index+2:size_index+4] = struct.pack(">H", size[0])

        reset_interval_index, length = cls.find_tag(
            header, TAGS['restart interval']
        )
        if reset_interval_index is not None:
            del header[reset_interval_index:reset_interval_index+length+2]

        return bytes(header)

    @classmethod
    def wrap_scan(
        cls,
        header: bytes,
        scan: bytes,
        size: Tuple[int, int]
    ) -> bytes:
        """Wrap scan data with manipulated header and end of image tag.

        Parameters
        ----------
        header: bytes
            Header to use.
        scan: bytes
            Scan data to wrap.
        size: Tuple[int, int]
            Pixel size of scan.

        Returns
        ----------
        bytes:
            Scan wrapped in header as bytes.
        """

        image = cls.manupulate_header(header, size)
        image += scan
        image += bytes([0xFF, 0xD9])
        return image

    @staticmethod
    def parse_restart_interval(payload: bytes) -> int:
        """Parse restart interval payload.

        Parameters
        ----------
        payload: bytes
            Huffman table in bytes.

        Returns
        ----------
        int
            Restart interval in number of MCUs
        """
        return unpack('>H', payload)[0]

    @staticmethod
    def parse_start_of_scan(
        payload: bytes
    ) -> List[Component]:
        """Parse start of scan paylaod. Only Huffman table selections are
        extracted.

        Parameters
        ----------
        payload: bytes
            Start of scan in bytes.

        Returns
        ----------
        List[Component]
            Huffman table selection with component name as key.

        """
        with io.BytesIO(payload) as buffer:
            number_of_components: int = unpack('B', buffer.read(1))[0]
            components: List[Component] = []
            for component in range(number_of_components):
                identifier, table_selection = unpack('BB', buffer.read(2))
                dc_table, ac_table = split_byte_into_nibbles(table_selection)
                if identifier == 0:
                    name = 'Y'
                elif identifier == 1:
                    name = 'Cb'
                elif identifier == 2:
                    name = 'Cr'
                else:
                    raise ValueError("Incorrect component identifier")
                components.append(Component(
                    name=name,
                    dc_table_id=HuffmanTableIdentifier('DC', dc_table),
                    ac_table_id=HuffmanTableIdentifier('AC', ac_table)
                ))
        return components

    @staticmethod
    def parse_huffman(
        payload: bytes,
        tables: Dict[HuffmanTableIdentifier, HuffmanTable]
    ) -> Dict[HuffmanTableIdentifier, HuffmanTable]:
        """Parse huffman table(s) in payload. Multiple tables can be defined in
        the same tag. Each table is stored in a dict with header as key.

        Parameters
        ----------
        payload: bytes
            Huffman table in bytes.
        tables: Dict[HuffmanTableIdentifier, HuffmanTable]
            Dict of revious tables to append read tables to.

        Returns
        ----------
        Dict[HuffmanTableIdentifier, HuffmanTable]
            Dict of Huffman tables.
        """
        table_start = 0
        while(table_start < len(payload)):
            (table, byte_read) = HuffmanTable.from_data(payload[table_start:])
            tables[table.identifier] = table
            table_start += byte_read
        return tables

    @staticmethod
    def parse_start_of_frame(payload: bytes) -> Tuple[int, int]:
        """Parse start of frame paylaod. Only height and width of frame is
        important.

        Parameters
        ----------
        payload: bytes
            Start of frame in bytes.

        Returns
        ----------
        Tuple[int, int]
            Height and width of frame.
        """
        _, height, width, _ = unpack('>BHHB', payload[0:6])
        return (width, height)

    @staticmethod
    def read_marker(buffer: io.BytesIO) -> Optional[int]:
        """Read a marker from buffer.

        Parameters
        ----------
        buffer: io.BytesIO
            Buffer to read marker from.

        Returns
        ----------
        Optional[int]
            Int representatin of marker.

        """
        try:
            (marker,) = unpack('>H', buffer.read(2))
            return marker
        except struct.error:
            return None

    @staticmethod
    def read_payload(buffer: io.BytesIO) -> bytes:
        """Read a payload from buffer.

        Parameters
        ----------
        buffer: io.BytesIO
            Buffer to read payload from.

        Returns
        ----------
        bytes
            Read payload.

        """
        payload_length = unpack('>H', buffer.read(2))[0]
        # Payload length includes length bytes
        return buffer.read(payload_length-2)

    def get_segments(
         self,
         data: bytes,
         scan_width: int,
         tiles_dc_offsets: List[Dc]
    ) -> List[JpegSegment]:
        """Parse MCUs in jpeg scan data into segments of max scan width size.
        Insert the segments into the supplied tile(s).

        Parameters
        ----------
        data: bytes
            Jpeg scan data to parse.
        scan_width: int
            Maximum number of pixels per segment.
        tiles: List[Tile]:
            Tiles to insert the segments into.

        """
        segments: List[JpegSegment] = []
        mcu_scan_width = scan_width // MCU_SIZE
        mcus_left = self.mcu_count
        scan = JpegScan(data, self.components)
        segment_dc_offset = Dc.zero(list(self.components.keys()))
        tile_index = 0
        while mcus_left > 0:
            mcu_to_scan = min(mcus_left, mcu_scan_width)
            segment = scan.read_and_modify_segment(
                mcu_to_scan,
                segment_dc_offset,
                tiles_dc_offsets[tile_index]
            )

            segments.append(segment)
            mcus_left -= mcu_to_scan
            tile_index += 1

        return segments


class JpegScan:
    """Class for minimal decoding of jpeg scan data"""
    def __init__(
        self,
        data: bytes,
        components: List[Component],
    ):
        """Parse jpeg scan using info in header.

        Parameters
        ----------
        header: JpegHeader
            Header.
        data: bytes
            Jpeg scan data, excluding start of scan tag.
        scan_width: int
            Maximum widht of produced segments.

        """

        self._components = components
        self._buffer = JpegBuffer(data)

    @property
    def components(self) -> Dict[str, Component]:
        return self._components

    @property
    def number_of_components(self) -> int:
        return len(self.components)

    def read_segment(
        self,
        count: int,
    ) -> List[int]:
        """Read a segment of count number of Mcus from stream

        Parameters
        ----------
        count: int
            Number of MCUs to extract.

        Returns
        ----------
        int:
            DC diff across segment.
        """
        start = self._buffer.pos
        dc_offsets = self._read_multiple_mcus(count)
        end = self._buffer.pos
        return JpegSegment(
            data=self._buffer.read_to_bitarray(start, end),
            segment_delta=dc_offsets
        )

    def _read_multiple_mcus(
        self,
        count: int,
    ) -> Dc:
        """Read count number of Mcus from stream. Only DC amplitudes are
        decoded. Accumulate the DC values of each component.

        Parameters
        ----------
        count: int
            Number of MCUs to read.

        Returns
        ----------
        Dict[str, int]
            Cumulative sums DC in MCUs per component.
        """
        dc_offsets = Dc.zero(list(self.components.keys()))
        for index in range(count):
            dc_offsets = self._read_mcu(dc_offsets)
        return dc_offsets

    def _read_mcu(self, dc_sums: Dc) -> Dc:
        """Parse MCU and return cumulative DC per component.

        Parameters
        ----------
        Dc
            Cumulative sums of previous MCUs DC per component.

        Returns
        ----------
        Dc
            Cumulative sums of previous MCUs DC, including this MCU, per
            component

        """
        mcu_dc = Dc({
                name: self._read_mcu_component(component)
                for name, component in self.components.items()
            })
        dc_sums.add(mcu_dc)
        return dc_sums

    def _read_mcu_component(
        self,
        component: Component,
    ) -> int:
        """Read single component of a MCU.

        Parameters
        ----------
        component: Component
            Huffman table selection.
        dc_sum: int
            Cumulative sum of previous MCUs DC for this component

        Returns
        ----------
        int
            Cumulative DC sum for this component including this MCU.

        """
        dc_amplitude = self._read_dc(component.dc_table)
        self._skip_ac(component.ac_table)
        return dc_amplitude

    def _read_dc(self, table: HuffmanTable) -> int:
        """Return DC amplitude for MCU block read from stream.

        Parameters
        ----------
        table: HuffmanTable
            Huffman table to use.

        Returns
        ----------
        Int
            DC amplitude for read MCU block.
        """
        length = self._buffer.read_variable_length(table)
        value = self._buffer.read(length)
        return self._decode_value(length, value)

    def _skip_ac(self, table: HuffmanTable) -> None:
        """Skip the ac part of MCU block read from stream.

        Parameters
        ----------
        table: HuffmanTable
            Huffman table to use.

        """
        mcu_length = 1  # DC amplitude is first value
        while mcu_length < 64:
            code = self._buffer.read_variable_length(table)

            if code == 0:  # End of block
                break
            else:
                zeros, length = split_byte_into_nibbles(code)
                self._buffer.skip(length)
                mcu_length += 1 + zeros

    @staticmethod
    @lru_cache(maxsize=None)
    def _decode_value(length: int, code: int) -> int:
        if length == 0:
            return 0
        # Smallest positive value for this length
        smallest_value = (1 << (length - 1))
        # If code is larger, value is positive
        if code >= smallest_value:
            return code
        # Negative value starts at negative largest value for this level
        largest_value = (smallest_value << 1) - 1
        return code - largest_value

    @staticmethod
    @lru_cache(maxsize=None)
    def _code_value(value: int) -> Tuple[int, int]:
        # Zero is coded as 0 length, no value
        if value == 0:
            return 0, 0
        # Length needed to code the value
        length = value.bit_length()
        # If value is negative, subtract 1
        if value < 0:
            value -= 1
        # Take out the lower bits according to length
        code = value & ((1 << length) - 1)
        return length, code

    @staticmethod
    @lru_cache(maxsize=None)
    def _to_bits(value: int, length: int) -> bitarray:
        if length == 0:
            return bitarray()
        return util.int2ba(value, length=length)

    def read_and_modify_segment(
        self,
        count: int,
        stripe_dc_offsets: Dc,
        tile_dc_offsets: Dc
    ) -> JpegSegment:

        output_segment = bitarray()
        # print("------------read and modify segment----------")
        # print(f"tile: {id(tile_dc_offsets)} {tile_dc_offsets}")
        # print(f"stripe: {id(stripe_dc_offsets)} {stripe_dc_offsets}")
        segment_dc_delta = Dc.zero(list(tile_dc_offsets.names))
        for name, component in self.components.items():
            current_dc = self._read_dc(component.dc_table)
            current_dc_non_diffed = stripe_dc_offsets[name] + current_dc
            new_dc = (
                current_dc_non_diffed - tile_dc_offsets[name]
            )  # ???
            # print(
            #     f"c {current_dc}, s {stripe_dc_offsets[name]} "
            #     f"t {tile_dc_offsets[name]}, new {new_dc}"
            # )l

            segment_dc_delta[name] += current_dc_non_diffed
            stripe_dc_offsets[name] += current_dc
            tile_dc_offsets[name] += new_dc
            # Code the value into length and code
            length, code = self._code_value(new_dc)
            # Encode length using huffman table and write to output segment
            output_segment += component.dc_table.encode_into_bits(length)
            # If length is non-zero, write code (in bits) to output segment
            output_segment += self._to_bits(code, length)

            # Get start and end of ac and write to output_segment
            ac_start = self._buffer.pos
            self._skip_ac(component.ac_table)
            ac_end = self._buffer.pos
            output_segment += self._buffer.read_to_bitarray(
                ac_start,
                ac_end
            )

        # Get start and end of rest of mcus and add to segment delta
        rest_start = self._buffer.pos
        rest_of_segment_dc_delta = self._read_multiple_mcus(count - 1)
        # print(
        #     f"rest of segment delta {id(rest_of_segment_dc_delta)} "
        #     f"{rest_of_segment_dc_delta}"
        # )
        segment_dc_delta.add(rest_of_segment_dc_delta)
        stripe_dc_offsets.add(rest_of_segment_dc_delta)
        tile_dc_offsets.add(rest_of_segment_dc_delta)
        end = self._buffer.pos
        output_segment += self._buffer.read_to_bitarray(
            rest_start,
            end
        )
        # print(f"segment delta {id(segment_dc_delta)} {segment_dc_delta}")
        # print(f"tile: {id(tile_dc_offsets)} {tile_dc_offsets}")
        # print(f"stripe: {id(stripe_dc_offsets)} {stripe_dc_offsets}")
        return JpegSegment(
            data=output_segment,
            segment_delta=segment_dc_delta,
            tile_offset=tile_dc_offsets
        )
