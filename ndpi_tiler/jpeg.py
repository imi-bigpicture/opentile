import io
import struct
from dataclasses import dataclass
from struct import unpack
from typing import Dict, List, Optional, Tuple


from bitarray import bitarray

from .huffman import HuffmanTable, HuffmanTableIdentifier
from .jpeg_tags import TAGS
from .utils import split_byte_into_nibbles


MCU_SIZE = 8


@dataclass
class Component:
    """Holds Huffman tables for a component."""
    identifier: int
    dc_table: HuffmanTable
    ac_table: HuffmanTable


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

    def create_constant(self, y: int, cb: int, cr: int) -> bytes:
        """Create jpeg byte scan data with constant color of the same size as
        defined in the jpeg header and with the same huffman tables. Usefull
        for padding input data when cropping edge tiles."""
        output = bitarray()
        dc_values = [y, cb, cr]
        for mcu in range(self.mcu_count):
            for index, component in enumerate(self.components.values()):
                output += component.dc_table.encode_value(4 * dc_values[index])
                output += component.ac_table.encode_value(0)
            dc_values = [0, 0, 0]
        return output.tobytes()



