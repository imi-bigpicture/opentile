import io
import unittest
from dataclasses import dataclass
from struct import unpack
from typing import Dict, List, Tuple, Type

import pytest
from bitarray import bitarray, util
from ndpi_tiler.huffman import HuffmanTable
from ndpi_tiler.jpeg import (Dc, JpegBuffer, JpegBufferBitBinary,
                             JpegBufferBitBit, JpegBufferBitDict,
                             JpegBufferByteBinary, JpegBufferByteDict,
                             JpegBufferIntBinary, JpegBufferIntDict,
                             JpegBufferIntList, JpegHeader, JpegScan,
                             JpegSegment)
from ndpi_tiler.jpeg_tags import MARER_MAPPINGS
from tifffile import TiffFile

from .create_jpeg_data import create_large_set, create_small_set, open_tif


@dataclass
class Mcu:
    position: int
    dc_sum: List[int]


@dataclass
class Segment:
    reset: bool
    start: int
    end: int
    count: int
    segment: JpegSegment


@pytest.mark.unittest
class NdpiTilerJpegTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.large_header: JpegHeader
        self.large_scan: JpegScan
        self.small_header: JpegHeader
        self.small_scan: JpegScan
        self.tif: TiffFile

    @classmethod
    def setUp(cls):
        cls.large_scan._buffer.reset()
        cls.small_scan._buffer.reset()

    @classmethod
    def setUpClass(cls):
        cls.tif = open_tif()
        cls.large_header, cls.large_scan, cls.large_data = create_large_set(
            cls.tif
        )
        cls.small_header, cls.small_scan, cls.small_data = create_small_set()

    @classmethod
    def tearDownClass(cls):
        cls.tif.close()

    def test_image_size(self):
        self.assertEqual(16, self.small_header.width)
        self.assertEqual(8, self.small_header.height)
        self.assertEqual(4096, self.large_header.width)
        self.assertEqual(8, self.large_header.height)

    def test_read_payload(self):
        payload = io.BytesIO(bytes([0x00, 0x03, 0x01]))
        self.assertEqual(
            1,
            unpack('B', JpegHeader.read_payload(payload))[0]
        )

    def test_read_marker(self):
        for marker in MARER_MAPPINGS.keys():
            self.assertEqual(
                marker,
                JpegHeader.read_marker(io.BytesIO(
                    bytes([marker >> 8, marker & 0xFF])
                ))
            )

    def test_small_scan_read_segments(self):
        start = 8*0+0
        end = 8*7+2
        true_segment = JpegSegment(
            data=self.small_scan._buffer.read_to_bitarray(start, end),
            segment_delta=Dc({1: 508, 2: 0, 3: 0})
        )
        read_segment = self.small_scan.read_segment(2)
        print(true_segment)
        print(read_segment)
        self.assertEqual(true_segment, read_segment)

    def test_large_scan_read_segments(self):
        starts = [0, 0, 8*267+7, 8*550+0, 8*824+4]
        ends = [8*1135+6, 8*267+7, 8*550+0, 8*824+4, 8*1135+6]
        data = [
            self.large_scan._buffer.read_to_bitarray(starts[i], ends[i])
            for i in range(len(starts))
        ]
        tests: List[Segment] = [
            Segment(
                reset=True,
                start=starts[0],
                end=ends[0],
                count=512,
                segment=JpegSegment(
                    data=data[0],
                    segment_delta=Dc({0: 81, 1: 2, 2: 0})
                )),
            Segment(
                reset=True,
                start=starts[1],
                end=ends[1],
                count=128,
                segment=JpegSegment(
                    data=data[1],
                    segment_delta=Dc({0: 80, 1: 2, 2: 0})
                )),
            Segment(
                reset=False,
                start=starts[2],
                end=ends[2],
                count=128,
                segment=JpegSegment(
                    data=data[2],
                    segment_delta=Dc({0: 0, 1: 0, 2: 0})
                )),
            Segment(
                reset=False,
                start=starts[3],
                end=ends[3],
                count=128,
                segment=JpegSegment(
                    data=data[3],
                    segment_delta=Dc({0: 1, 1: 0, 2: 0})
                )),
            Segment(
                reset=False,
                start=starts[4],
                end=ends[4],
                count=128,
                segment=JpegSegment(
                    data=data[4],
                    segment_delta=Dc({0: 0, 1: 0, 2: 0})
                ))
        ]
        for test in tests:
            if test.reset:
                self.large_scan._buffer.seek(0)
            print(self.large_scan._buffer.position)
            read_segment = self.large_scan.read_segment(test.count)

            self.assertEqual(test.segment, read_segment)

    def test_large_scan_read_mcus(self):
        # Header offset, as positions are readed from jpeg
        header_offset = 0x294

        true_mcus: Dict[int, Mcu] = {
            0: Mcu(
                position=8*(0x294-header_offset) + 0,
                dc_sum=Dc({0: 80, 1: 2, 2: 0})
            ),
            1: Mcu(
                position=8*(0x297-header_offset) + 2,
                dc_sum=Dc({0: 1, 1: 0, 2: 0})
            ),
            150: Mcu(
                position=8*(0x3D4-header_offset) + 5,
                dc_sum=Dc({0: 0, 1: 0, 2: 0})
            ),
            151: Mcu(
                position=8*(0x3D6-header_offset) + 3,
                dc_sum=Dc({0: 0, 1: 1, 2: 0})
            ),
            510: Mcu(
                position=8*(0x700-header_offset) + 0,
                dc_sum=Dc({0: -1, 1: 0, 2: 0})
            ),
            511: Mcu(
                position=8*(0x702-header_offset) + 0,
                dc_sum=Dc({0: 0, 1: 0, 2: 0})
            )
        }
        self.large_scan._buffer.seek(0)
        read_mcus = {
            index: Mcu(
                position=self.large_scan._buffer.position,
                dc_sum=self.large_scan._read_mcu(
                    Dc({0: 0, 1: 0, 2: 0})
                )
            )
            for index in range(self.large_header.mcu_count)
        }

        for index, value in true_mcus.items():
            self.assertEqual(value, read_mcus[index])

    def test_code_decode(self):
        for value in range(-2047, 2047):
            length, code = JpegScan._code_value(value)
            decode = JpegScan._decode_value(length, code)
            self.assertEqual(value, decode)

    def test_write_and_read_bitarray(self):
        bits = bitarray()
        # Write huffman coded length and symbol for each possible value
        # in huffman tables.
        for component in self.large_scan.components.values():
            for table in [component.ac_table, component.dc_table]:
                for value in table.encode_dict.keys():
                    length, code = self.large_scan._code_value(value)
                    length_bits = table.encode_into_bits(length)
                    bits += length_bits
                    code_bits = self.large_scan._to_bits(code, length)
                    bits += code_bits

        # Read huffman coded length and symbol and compare to each possible
        # value in huffman tables
        bit_position = 0
        for component in self.large_scan.components.values():
            for table in [component.ac_table, component.dc_table]:
                for value in table.encode_dict.keys():
                    decoded_length, read_bits = table.decode_from_bits(
                        bits[bit_position:]
                    )
                    bit_position = bit_position + read_bits
                    if decoded_length == 0:
                        decoded_value = 0
                    else:
                        decoded_value = self.large_scan._decode_value(
                            decoded_length,
                            util.ba2int(
                                bits[bit_position:bit_position+decoded_length]
                            )
                        )
                    bit_position += decoded_length
                    self.assertEqual(value, decoded_value)
        # Check that all bits have been read
        self.assertEqual(len(bits), bit_position)

    def test_encode_know_sequence(self):
        know_bits = bitarray(
            '1111 1100 1111 1111 1110 0010\
             1010 1111 1110 1111 1111 0011\
             0001 0101 01'
        )
        # length symbol, value, table
        values: List[Tuple[int, int, HuffmanTable]] = [
            (10, -512, self.small_scan.components[1].dc_table),
            (0, 0, self.small_scan.components[1].ac_table),
            (0, 0, self.small_scan.components[2].dc_table),
            (0, 0, self.small_scan.components[2].ac_table),
            (0, 0, self.small_scan.components[3].dc_table),
            (0, 0, self.small_scan.components[3].ac_table),
            (10, 1020, self.small_scan.components[1].dc_table),
            (0, 0, self.small_scan.components[1].ac_table),
            (0, 0, self.small_scan.components[2].dc_table),
            (0, 0, self.small_scan.components[2].ac_table),
            (0, 0, self.small_scan.components[3].dc_table),
            (0, 0, self.small_scan.components[3].ac_table),
        ]
        bits = bitarray()
        for value in values:
            block = bitarray()
            table = value[2]
            length, code = self.small_scan._code_value(value[1])
            length_bits = table.encode_into_bits(length)
            block += length_bits
            code_bits = self.small_scan._to_bits(code, length)
            block += code_bits
            bits += block
        self.assertEqual(know_bits, bits)

    def test_decode_know_sequence(self):
        know_bits = bytes([
            0b11111100, 0b11111111, 0b11100010,
            0b10101111, 0b11101111, 0b11110011,
            0b00010101, 0b01110000
        ])
        buffers: List[Type[JpegBuffer]] = [
            JpegBufferByteBinary, JpegBufferByteDict,
            JpegBufferBitBit, JpegBufferBitDict, JpegBufferBitBinary,
            JpegBufferIntBinary, JpegBufferIntDict, JpegBufferIntList
        ]
        for buffer_type in buffers:
            buffer = buffer_type(know_bits)
            tables = [
                self.small_scan.components[1].dc_table,
                self.small_scan.components[1].ac_table,
                self.small_scan.components[2].dc_table,
                self.small_scan.components[2].ac_table,
                self.small_scan.components[3].dc_table,
                self.small_scan.components[3].ac_table,
            ]
            for block in range(2):
                for table in tables:
                    # print(table.decode_tree_dict)
                    length = buffer.read_variable_length(table)
                    code = buffer.read(length)

                    decoded = JpegScan._decode_value(length, code)
                    print(length, code, decoded)

    def test_read_and_modify_segment_no_modify(self):
        starts = [0, 0, 8*267+7, 8*550+0, 8*824+4]
        ends = [8*1135+6, 8*267+7, 8*550+0, 8*824+4, 8*1135+6]
        data = [
            self.large_scan._buffer.read_to_bitarray(starts[i], ends[i])
            for i in range(len(starts))
        ]
        tests: List[Segment] = [
            Segment(
                reset=True,
                start=starts[0],
                end=ends[0],
                count=512,
                segment=JpegSegment(
                    data=data[0]
                )),
            Segment(
                reset=True,
                start=starts[1],
                end=ends[1],
                count=128,
                segment=JpegSegment(
                    data=data[1],
                )),
            Segment(
                reset=False,
                start=starts[2],
                end=ends[2],
                count=128,
                segment=JpegSegment(
                    data=data[2]
                )),
            Segment(
                reset=False,
                start=starts[3],
                end=ends[3],
                count=128,
                segment=JpegSegment(
                    data=data[3]
                )),
            Segment(
                reset=False,
                start=starts[4],
                end=ends[4],
                count=128,
                segment=JpegSegment(
                    data=data[4]
                ))
        ]
        for test in tests:
            if test.reset:
                self.large_scan._buffer.reset()

            read_segment = self.large_scan.read_and_modify_segment(
                test.count,
                Dc.zero(self.large_scan.components.keys()),
                Dc.zero(self.large_scan.components.keys())
            )
            print(test.segment)
            print(read_segment)
            self.assertEqual(test.segment, read_segment)
