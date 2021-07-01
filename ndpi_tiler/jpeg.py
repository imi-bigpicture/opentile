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
    """Holds Huffman tables for a component."""
    identifier: int
    dc_table: HuffmanTable
    ac_table: HuffmanTable


@dataclass
class Dc:
    """Holds DC amplitude offsets for components identified by component id."""
    offsets: Dict[int, int]

    def add(self, values: 'Dc') -> None:
        """Add another offset to this offset."""
        for identifier in self.identifiers:
            self.offsets[identifier] += values[identifier]

    def remove(self, values: 'Dc') -> None:
        """Remove another offset to this offset."""
        for identifier in self.identifiers:
            self.offsets[identifier] -= values[identifier]

    def __getitem__(self, identifier: int) -> int:
        return self.offsets[identifier]

    def __setitem__(self, identifier: int, value: int) -> None:
        self.offsets[identifier] = value

    def __eq__(self, other) -> bool:
        if isinstance(other, Dc):
            for key in self.offsets:
                if not (key in other.offsets and self[key] == other[key]):
                    return False
            return True
        return NotImplemented

    @classmethod
    def zero(cls, identifiers: List[int]) -> 'Dc':
        """Return a object initialized with 0."""
        return cls({identifier: 0 for identifier in identifiers})

    @property
    def identifiers(self) -> List[int]:
        return self.offsets.keys()


@dataclass
class JpegSegment:
    """A segment of Jpeg data read and possibly modified from buffer."""
    data: bitarray
    segment_delta: Dc = None


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
    def remove_stuffing(data: bytes) -> bytearray:
        """Remove byte stuffing (0x00 following 0xFF)"""
        return data.replace(BYTE_TAG_STUFFING, BYTE_TAG)

    @property
    def position(self) -> int:
        """The current buffer position."""
        return self._bit_pos

    def read(self, count: int = 1) -> int:
        """Read count bits and return the unsigned integer interpretation."""
        sum = 0
        for value in self._data[self._bit_pos:self._bit_pos+count]:
            sum = 2*sum + value

        self._bit_pos += count
        return sum

    def read_variable_length(self, table: HuffmanTable) -> int:
        """Read variable length using huffman table"""
        length = 1
        symbol = self.read()
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
        """Return bitarray from start to end of buffer."""
        return bitarray(self._data[start:end])

    def seek(self, position: int) -> None:
        """Seek to bit posiion in buffer."""
        self._bit_pos = position

    def skip(self, skip_length: int) -> None:
        """Skip length of bits in buffer."""
        skip_to = self.position + skip_length
        self.seek(skip_to)

    @classmethod
    def convert_to_bytes(cls, bits: bitarray) -> bytes:
        padding_bits = 7 - (len(bits) - 1) % 8
        bits += bitarray(padding_bits*[1])

        # Convert to bytes
        data = bytearray(bits.tobytes())

        # Add byte stuffing after 0xFF
        data = data.replace(BYTE_TAG, BYTE_TAG_STUFFING)

        return data


class JpegHeader:
    """Class for minimal parsing of jpeg header"""
    def __init__(
        self,
        data: bytes,
    ) -> None:
        """Parse jpeg header. Read markers from data and parse payload if
        huffman table(s) or start of frame.

        Parameters
        ----------
        data: bytes
            Jpeg header in bytes.

        Returns
        ----------
        JpegHeader
            JpegHeader created from data.
        """
        self.header_data = data
        width: int
        height: int
        components_stubs: List[Tuple[int, int, int]] = []
        huffman_tables: Dict[HuffmanTableIdentifier, HuffmanTable] = {}

        restart_interval = None
        with io.BytesIO(data) as buffer:
            marker = self.read_marker(buffer)
            if not marker == TAGS['start of image']:
                raise ValueError("Expected start of image marker")
            marker = self.read_marker(buffer)
            while marker is not None:
                if (
                    marker == TAGS['start of image'] or
                    marker == TAGS['end of image']
                ):
                    raise ValueError("Unexpected marker")
                payload = self.read_payload(buffer)
                if marker == TAGS['huffman table']:
                    huffman_tables = self.parse_huffman(
                        payload,
                        huffman_tables
                    )
                elif marker == TAGS['start of frame']:
                    (width, height) = self.parse_start_of_frame(payload)
                elif marker == TAGS['start of scan']:
                    components_stubs = self.parse_start_of_scan(payload)
                elif marker == TAGS['restart interval']:
                    restart_interval = self.parse_restart_interval(payload)
                else:
                    pass

                marker = self.read_marker(buffer)
        if (
            huffman_tables == {} or
            width is None or height is None or
            components_stubs == []
        ):
            raise ValueError("missing tags")

        self._components = {
            component_id: Component(
                component_id,
                huffman_tables[HuffmanTableIdentifier('DC', dc_id)],
                huffman_tables[HuffmanTableIdentifier('AC', ac_id)]
            )
            for (component_id, dc_id, ac_id) in components_stubs
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
    def components(self) -> Dict[int, Component]:
        return self._components

    @property
    def mcu_count(self) -> int:
        return self._mcu_count

    @property
    def restart_interval(self) -> int:
        if self._restart_interval is not None:
            return self._restart_interval
        return self.width * self.height // (MCU_SIZE*MCU_SIZE)

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
    def manupulated_header(
        cls,
        header: bytes,
        size: Tuple[int, int]
    ) -> bytes:
        """Return manipulated header with changed pixel size (width, height)
        of removed reset interval marker.

        Parameters
        ----------
        heaer: bytes
            Header to manipulate.
        size: Tuple[int, int]
            Pixel size to insert into header.

        Returns
        ----------
        bytes:
            Manupulated header.
        """
        header = bytearray(header)
        start_of_scan_index, length = cls.find_tag(
            header, TAGS['start of frame']
        )
        if start_of_scan_index is None:
            raise ValueError("Start of scan tag not found in header")
        size_index = start_of_scan_index+5
        header[size_index:size_index+2] = struct.pack(">H", size[1])
        header[size_index+2:size_index+4] = struct.pack(">H", size[0])

        reset_interval_index, length = cls.find_tag(
            header, TAGS['restart interval']
        )
        if reset_interval_index is not None:
            del header[reset_interval_index:reset_interval_index+length+2]

        return bytes(header)

    def wrap_scan(
        self,
        scan: bytes,
        size: Tuple[int, int]
    ) -> bytes:
        """Wrap scan data with manipulated header and end of image tag.

        Parameters
        ----------
        scan: bytes
            Scan data to wrap.
        size: Tuple[int, int]
            Pixel size of scan.

        Returns
        ----------
        bytes:
            Scan wrapped in header as bytes.
        """

        image = self.manupulated_header(self.header_data, size)
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
    ) -> List[Tuple[int, int, int]]:
        """Parse start of scan paylaod. Only Huffman table selections are
        extracted.

        Parameters
        ----------
        payload: bytes
            Start of scan in bytes.

        Returns
        ----------
        List[Tuple[int, int, int]]
            List of component id, dc table id and ac table id

        """
        with io.BytesIO(payload) as buffer:
            number_of_components: int = unpack('B', buffer.read(1))[0]
            components: List[Tuple[int, int, int]] = []
            for component in range(number_of_components):
                identifier, table_selection = unpack('BB', buffer.read(2))
                dc_table, ac_table = split_byte_into_nibbles(table_selection)
                components.append((identifier, dc_table, ac_table))
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
        Modify the DC values in the segments according to the tile dc offset
        the segment is to be inserted into.

        Parameters
        ----------
        data: bytes
            Jpeg scan data to parse.
        scan_width: int
            Maximum number of pixels per segment.
        tiles_dc_offsets: List[Dc]:
            Dc offsets of tiles the segment are to be inserted into.

        Returns
        ----------
        List[JpegSegment]
            List of created JpegSegments.
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
        """Parse jpeg data using information in components.

        Parameters
        ----------
        data: bytes
            Jpeg scan data, excluding start of scan tag.
        components: List[Components]
            List of components in the scan.

        """
        self._components = components
        self._buffer = JpegBuffer(data)

    @property
    def components(self) -> Dict[int, Component]:
        return self._components

    @property
    def number_of_components(self) -> int:
        return len(self.components)

    def read_segment(
        self,
        count: int,
    ) -> List[int]:
        """Read a segment of count number of Mcus from buffer.

        Parameters
        ----------
        count: int
            Number of MCUs to extract.

        Returns
        ----------
        JpegSegment:
            Read segment.
        """
        start = self._buffer.position
        dc_offsets = self._read_multiple_mcus(count)
        end = self._buffer.position
        return JpegSegment(
            data=self._buffer.read_to_bitarray(start, end),
            segment_delta=dc_offsets
        )

    def _read_multiple_mcus(
        self,
        count: int,
    ) -> Dc:
        """Read count number of Mcus from buffer. Only DC amplitudes are
        decoded. Accumulate the DC values of each component.

        Parameters
        ----------
        count: int
            Number of MCUs to read.

        Returns
        ----------
        Dc
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
                identifier: self._read_mcu_component(component)
                for identifier, component in self.components.items()
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
            Component to read.

        Returns
        ----------
        int
            Cumulative DC sum for this component including this MCU.

        """
        dc_amplitude = self._read_dc(component.dc_table)
        self._skip_ac(component.ac_table)
        return dc_amplitude

    def _read_dc(self, table: HuffmanTable) -> int:
        """Return DC amplitude for MCU block read from buffer.

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
        """Read length of each AC of the 63 component of the block,
        and skip ahead that length. End block if end of block is read.

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
        """Decode code based on length into a integer."""
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
        """Code integer value into code and length."""
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
        """Convert a value to a bitarry of specified length."""
        if length == 0:
            return bitarray()
        return util.int2ba(value, length=length)

    def read_and_modify_segment(
        self,
        count: int,
        stripe_dc_offsets: Dc,
        tile_dc_offsets: Dc
    ) -> JpegSegment:
        """Read and modify segment of count number of Mcus from buffer. The DC
        values of the first MCU is modified in according to the supplied
        tile dc offsets

        Parameters
        ----------
        count: int
            Number of MCUs to extract.
        stripe_dc_offsets: Dc
            Offsets of the stripe until this segment.
        tile_dc_offsets: Dc
            Offset of the tile until this segment.

        Returns
        ----------
        JpegSegment:
            Read and modified segment.
        """

        output_segment = bitarray()

        for id, component in self.components.items():
            current_dc = self._read_dc(component.dc_table)
            new_dc = (
                stripe_dc_offsets[id] + current_dc - tile_dc_offsets[id]
            )

            stripe_dc_offsets[id] += current_dc
            tile_dc_offsets[id] += new_dc
            # Code the value into length and code
            length, code = self._code_value(new_dc)
            # Encode length using huffman table and write to output segment
            output_segment += component.dc_table.encode_into_bits(length)
            # Write code to output segment
            output_segment += self._to_bits(code, length)

            # Get start and end of ac and write to output_segment
            block_ac_start = self._buffer.position
            self._skip_ac(component.ac_table)
            block_ac_end = self._buffer.position
            output_segment += self._buffer.read_to_bitarray(
                block_ac_start,
                block_ac_end
            )

        # Get start and end of rest of mcus and add to segment delta
        rest_of_segment_start = self._buffer.position
        rest_of_segment_dc_delta = self._read_multiple_mcus(count - 1)

        stripe_dc_offsets.add(rest_of_segment_dc_delta)
        tile_dc_offsets.add(rest_of_segment_dc_delta)
        segment_end = self._buffer.position
        output_segment += self._buffer.read_to_bitarray(
            rest_of_segment_start,
            segment_end
        )

        return JpegSegment(data=output_segment)
