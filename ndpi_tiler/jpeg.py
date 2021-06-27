import io
import struct
from dataclasses import dataclass
from struct import unpack
from typing import Dict, List, Optional, Tuple

from bitstring import BitArray

from ndpi_tiler.huffman import (HuffmanTable, HuffmanTableIdentifier)
from ndpi_tiler.jpeg_tags import TAGS
from ndpi_tiler.stream import Stream, StreamPosition
from ndpi_tiler.utils import split_byte_into_nibbles

MCU_SIZE = 8


@dataclass
class Component:
    name: str
    dc_table_id: HuffmanTableIdentifier
    ac_table_id: HuffmanTableIdentifier
    dc_table: HuffmanTable = None
    ac_table: HuffmanTable = None


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
        hdr, height, width, components = unpack('>BHHB', payload[0:6])
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


@dataclass
class JpegSegment:
    data: bytearray
    length: StreamPosition
    count: int
    dc_offset: Dict[str, int]
    dc_sum: Dict[str, int]
    modified: bool = False


class JpegScan:
    """Class for minimal decoding of jpeg scan data"""
    def __init__(
        self,
        header: JpegHeader,
        data: bytes,
        scan_width: int = None
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
        self._header = header
        self._mcu_count = header.height * header.width // (MCU_SIZE * MCU_SIZE)
        if scan_width is not None:
            self._scan_width = scan_width
        else:
            self._scan_width = self._mcu_count * MCU_SIZE
        self._stream = Stream(data)
        self.segments = self._get_segments(self._scan_width)

    @property
    def components(self) -> Dict[str, Component]:
        return self._header.components

    @property
    def huffman_tables(self) -> List[HuffmanTable]:
        return self._header.huffman_tables

    @property
    def mcu_count(self) -> int:
        return self._mcu_count

    @property
    def restart_interval(self) -> int:
        return self._header.restart_interval

    @property
    def number_of_components(self) -> int:
        return len(self.components)

    def _get_segments(self, scan_width: int) -> List[JpegSegment]:
        """Parse MCUs in jpeg scan data produce segments. Each segment
        contains max scan width (in pixels) number of MCUs and the cumulative
        DC amplitude (per component).

        Parameters
        ----------
        scan_width: int
            Maximum number of pixels per segment.

        Returns
        ----------
        List[JpegSegment]
            Segments of MCUs.
        """
        mcu_scan_width = scan_width // MCU_SIZE
        segments: List[JpegSegment] = []
        mcus_left = self.mcu_count
        # print(f"mcu count {self.mcu_count} mcu scan width {mcu_scan_width}")
        dc_offset = {name: 0 for name in self.components.keys()}
        while mcus_left > 0:
            # print(f"mcus left {mcus_left}")
            mcu_to_scan = min(mcus_left, mcu_scan_width)
            segment = self._extract_segment(mcu_to_scan, dc_offset)
            segments.append(segment)
            mcus_left -= mcu_to_scan
            dc_offset = segment.dc_sum

        return segments

    def _extract_segment(
        self,
        count: int,
        dc_offset: Dict[str, int]
    ) -> JpegSegment:
        """Extract a segment of count number of Mcus from stream

        Parameters
        ----------
        count: int
            Number of MCUs to extract.
        dc_offset: Dict[str, int]
            DC of previous segment.

        Returns
        ----------
        JpegSegment
            Segment of MCUs read.
        """
        dc_sum = self._read_multiple_mcus(count)
        scan_end = self._stream.pos
        return JpegSegment(
            data=self._stream.data,
            length=scan_end,
            count=count,
            dc_offset=dc_offset,
            dc_sum=dc_sum
        )

    def _read_multiple_mcus(self, count: int) -> List[int]:
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
        dc_sums = {name: 0 for name in self.components.keys()}
        for index in range(count):
            # print(f"mcu index {index} buffer pos {self._stream.pos}")
            dc_sums = self._read_mcu(dc_sums)
        return dc_sums

    def _read_mcu(self, dc_sums: Dict[str, int]) -> Dict[str, int]:
        """Parse MCU and return cumulative DC per component.

        Parameters
        ----------
        dc_sums: Dict[str, int]
            Cumulative sums of previous MCUs DC per component.

        Returns
        ----------
        Dict[str, int]
            Cumulative sums of previous MCUs DC, including this MCU, per
            component

        """
        return {
            name: self._read_mcu_component(component, dc_sums[name])
            for name, component in self.components.items()
        }

    def _read_mcu_component(
        self,
        component: Component,
        dc_sum: int
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
        return dc_sum + dc_amplitude

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
        length = table.decode(self._stream)
        value = self._stream.read(length)
        # print(f"reading {length}")
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
            code = table.decode(self._stream)
            if code == 0:  # End of block
                break
            else:
                zeros, length = split_byte_into_nibbles(code)
                # value = self._stream.read(length)
                # print(f"skipping {length}")
                self._stream.skip(length)
                mcu_length += 1 + zeros

    @staticmethod
    def _decode_value(length: int, code: int) -> int:
        # Smallest positive value for this length
        smallest_value = 2 ** (length - 1)
        # If code is larger, value is positive
        if code >= smallest_value:
            return code
        # Negative value starts at negative largest value for this level
        largest_value = 2 * smallest_value - 1
        return code - largest_value

    @staticmethod
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
        code = value & (2**length - 1)
        return length, code

    def close(self) -> None:
        self._stream.close()
