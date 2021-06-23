import io
import struct
from dataclasses import dataclass
from struct import unpack
from typing import Dict, List, Optional, Tuple

from bitstring import BitArray

from ndpi_tiler.huffman import (HuffmanTable, HuffmanTableIdentifier,
                                HuffmanTableSelection)
from ndpi_tiler.jpeg_tags import TAGS
from ndpi_tiler.stream import Stream
from ndpi_tiler.utils import split_byte_into_nibbles

MCU_SIZE = 8


class JpegHeader:
    """Class for minimal parsing of jpeg header"""

    def __init__(
        self,
        huffman_tables: List[HuffmanTable],
        width: int,
        height: int,
        table_selections: Dict[int, HuffmanTableSelection],
        restart_interval: int = None
    ) -> None:

        self._huffman_tables = {
            table.identifier: table for table in huffman_tables
        }
        self._width = width
        self._height = height
        self._table_selections = table_selections
        self._restart_interval = restart_interval

    @property
    def huffman_tables(self) -> List[HuffmanTable]:
        return self._huffman_tables

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def table_selections(self) -> Dict[int, HuffmanTableSelection]:
        return self._table_selections

    @property
    def restart_interval(self) -> int:
        if self._restart_interval is not None:
            return self._restart_interval
        return self.width * self.height // (8*8)

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
        huffman_tables: List[HuffmanTable] = []
        width: int
        height: int
        table_selections: Dict[int, HuffmanTableSelection] = {}

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
                    huffman_tables += cls.parse_huffman(payload)
                elif marker == TAGS['start of frame']:
                    (width, height) = cls.parse_start_of_frame(payload)
                elif marker == TAGS['start of scan']:
                    table_selections = cls.parse_start_of_scan(payload)
                elif marker == TAGS['restart interval']:
                    restart_interval = cls.parse_restart_interval(payload)
                else:
                    pass

                marker = cls.read_marker(buffer)
        if (
            huffman_tables == [] or
            width is None or height is None or
            table_selections == {}
        ):
            raise ValueError("missing tags")
        return cls(
            huffman_tables,
            width,
            height,
            table_selections,
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
    ) -> Dict[int, HuffmanTableSelection]:
        """Parse start of scan paylaod. Only Huffman table selections are
        extracted.

        Parameters
        ----------
        payload: bytes
            Start of scan in bytes.

        Returns
        ----------
        Dict[int, HuffmanTableSelection]
            Huffman table selection with component identifier as key.

        """
        with io.BytesIO(payload) as buffer:
            components: int = unpack('B', buffer.read(1))[0]
            table_selections: Dict[int, Tuple[int, int]] = {}
            for component in range(components):
                identifier, table_selection = unpack('BB', buffer.read(2))
                dc_table, ac_table = split_byte_into_nibbles(table_selection)
                table_selections[identifier] = HuffmanTableSelection(
                    dc=dc_table,
                    ac=ac_table
                )
        return table_selections

    @staticmethod
    def parse_huffman(payload: bytes) -> List[HuffmanTable]:
        """Parse huffman table(s) in payload. Multiple tables can be defined in
        the same tag. Each table is stored in a dict with header as key.

        Parameters
        ----------
        payload: bytes
            Huffman table in bytes.

        Returns
        ----------
        List[HuffmanTable]
            List of Huffman tables.
        """
        tables: List[HuffmanTable] = []
        table_start = 0
        while(table_start < len(payload)):
            (table, byte_read) = HuffmanTable.from_data(payload[table_start:])
            tables.append(table)
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
class Mcu:
    """A Mcu, consisting of one or more blocks (components)"""
    dc_amplitudes: List[int]


@dataclass
class JpegSegment:
    data: BitArray
    length: int
    dc_sum: int
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
            Header containing
        data: bytes
            Jpeg scan data, excluding start of scan tag

        """
        self._header = header
        self._mcu_count = header.height * header.width // (MCU_SIZE * MCU_SIZE)
        if scan_width is not None:
            self._scan_width = scan_width
        else:
            self._scan_width = self._mcu_count // MCU_SIZE
        self.segments = self._get_segments(data)

    @property
    def table_selections(self) -> Dict[int, HuffmanTableSelection]:
        return self._header.table_selections

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
        return len(self.table_selections)

    def _get_segments(
        self,
        data: bytes,
    ) -> List[JpegSegment]:
        # Result should be a list with jpeg segments, each segments has
        # byte data and a mcu length
        # From the avaiable mcus we need to scan mcu_to_scan until all mcus are
        # scanned
        # For each segment, we need the bits for the first mcu blocks separete,
        # the start and end of the rest of the scan and the cumulative DC
        # component
        # For the first mcu, we need to modifiy the DC component of each block
        # We then joing together the first mcus modified blocks and the rest of
        # the scan
        stream = Stream(data)
        segments: List[JpegSegment] = []
        mcus_left = self.mcu_count
        while mcus_left > 0:
            mcu_to_scan = max(mcus_left, self._scan_width // MCU_SIZE)
            segment = self._extract_segment(stream, mcu_to_scan)
            segments.append(segment)
            mcus_left -= mcu_to_scan

        return segments

    def _extract_segment(
        self,
        stream: Stream,
        count: int,
    ) -> JpegSegment:
        scan_start = stream.pos
        # mcus = [
        #     self._read_mcu(stream)
        #     for mcu in range(count)
        # ]
        mcus = []
        for mcu in range(count):
            print(f"mcu {mcu} position {stream.pos}")
            mcus.append(self._read_mcu(stream))
        cumulative_dc: List[int] = [0] * self.number_of_components
        for mcu in mcus:
            for index, amplitude in enumerate(mcu.dc_amplitudes):
                cumulative_dc[index] += amplitude
        scan_end = stream.pos

        return JpegSegment(
            data=stream.read_segment(scan_start, scan_end),
            length=count,
            dc_sum=cumulative_dc
        )

        # padding_bits = 8 - len(segment_bits) % 8
        # segment_bits.append(Bits(f'{padding_bits}*0b1'))
        # segment_bits.append('0xFFD9')

    def _read_mcu(
        self,
        stream
    ) -> Mcu:
        """Return mcu (position and ac amplitudes) read from stream.

        Parameters
        ----------
        stream: Stream
            Stream of jpeg scan data

        Returns
        ----------
        Mcu
            Mcu containing position and ac amplitudes.
        """
        return Mcu([
            self._read_mcu_block(stream, table_selection)
            for table_selection in self.table_selections.values()
        ])

    @staticmethod
    def _decode_value(length: int, value: int) -> int:
        """Decode value by magic. Need to check how this works."""
        magic = 2 ** (length - 1)
        if value < magic:
            value -= (2 * magic - 1)
        return int(value)

    def _read_dc(
        self,
        stream: Stream,
        table_identifier: HuffmanTableIdentifier
    ) -> int:
        """Return DC amplitude for mcu block read from stream.

        Parameters
        ----------
        stream: Stream
            Stream of jpeg scan data
        table_identifier: HuffmanTableIdentifier
            Identifier for Huffman table to use.

        Returns
        ----------
        Int
            DC amplitude for read mcu block.
        """
        table: HuffmanTable = self.huffman_tables[table_identifier]
        length = table.decode(stream)
        value = stream.read(length)
        return self._decode_value(length, value)

    def _skip_ac(
        self,
        stream: Stream,
        table_identifier: HuffmanTableIdentifier
    ) -> None:
        """Skip the ac part of mcu block read from stream.

        Parameters
        ----------
        stream: Stream
            Stream of jpeg scan data
        table_identifier: HuffmanTableIdentifier
            Identifier for Huffman table to use.

        """

        table: HuffmanTable = self.huffman_tables[table_identifier]
        mcu_length = 1  # DC amplitude is first value
        while mcu_length < 64:
            code = table.decode(stream)
            if code == 0:  # End of block
                break
            else:
                zeros, length = split_byte_into_nibbles(code)
                # value = stream.read(length)
                stream.skip(length)
                mcu_length += 1 + zeros

    def _read_mcu_block(
        self,
        stream: Stream,
        table_selection: Tuple[int, int]
    ) -> int:
        """Read single block (component) of a MCU.

        Parameters
        ----------
        stream: Stream
            Stream of jpeg scan data
        table_selection: Tuple[int, int]
            Huffman table selection for DC and AC
        """
        dc_amplitude = self._read_dc(
            stream,
            HuffmanTableIdentifier('DC', table_selection.dc)
        )
        self._skip_ac(
            stream,
            HuffmanTableIdentifier('AC', table_selection.ac)
        )
        return dc_amplitude
