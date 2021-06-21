import io
import os
import unittest
from struct import unpack

import pytest
from bitstring import ConstBitStream
from ndpi_tiler.jpeg import (HuffmanLeaf, HuffmanNode, HuffmanTable,
                             JpegHeader, JpegScan, Stream, marker_mapping, Mcu)
from tifffile import TiffFile, TiffPage

tif_test_data_dir = os.environ.get("TIF_TESTDIR", "C:/temp/tif")
tif_test_file_name = "test.ndpi"
tif_test_file_path = tif_test_data_dir + '/' + tif_test_file_name


def create_small_header() -> JpegHeader:
    table_0 = HuffmanTable(
        0,
        [
            [],
            [],
            [0x04, 0x05, 0x03, 0x02, 0x06, 0x01, 0x00],
            [0x07],
            [0x08],
            [0x09],
            [0x0A],
            [0x0B]
        ])
    table_1 = HuffmanTable(
        1,
        [
            [],
            [0x01, 0x00],
            [0x02, 0x03],
            [0x04, 0x05, 0x06],
            [0x07],
            [0x08],
            [0x09],
            [0x0A],
            [0x0B]
        ])
    table_16 = HuffmanTable(
        16,
        [
            [],
            [0x01, 0x02],
            [0x03],
            [0x11, 0x04, 0x00],
            [0x05, 0x21, 0x12],
            [0x31, 0x41],
            [0x51, 0x06, 0x13, 0x61],
            [0x22, 0x71],
            [0x81, 0x14, 0x32, 0x91, 0xA1, 0x07],
            [0x15, 0xB1, 0x42, 0x23, 0xC1, 0x52, 0xD1],
            [0xE1, 0x33, 0x16],
            [0x62, 0xF0, 0x24, 0x72],
            [0x82, 0xF1],
            [0x25, 0x43, 0x34, 0x53, 0x92, 0xA2],
            [0xB2, 0x63],
            [
                0x73, 0xC2, 0x35, 0x44, 0x27, 0x93, 0xA3, 0xB3, 0x36, 0x17,
                0x54, 0x64, 0x74, 0xC3, 0xD2, 0xE2, 0x08, 0x26, 0x83, 0x09,
                0x0A, 0x18, 0x19, 0x84, 0x94, 0x45, 0x46, 0xA4, 0xB4, 0x56,
                0xD3, 0x55, 0x28, 0x1A, 0xF2, 0xE3, 0xF3, 0xC4, 0xD4, 0xE4,
                0xF4, 0x65, 0x75, 0x85, 0x95, 0xA5, 0xB5, 0xC5, 0xD5, 0xE5,
                0xF5, 0x66, 0x76, 0x86, 0x96, 0xA6, 0xB6, 0xC6, 0xD6, 0xE6,
                0xF6, 0x37, 0x47, 0x57, 0x67, 0x77, 0x87, 0x97, 0xA7, 0xB7,
                0xC7, 0xD7, 0xE7, 0xF7, 0x38, 0x48, 0x58, 0x68, 0x78, 0x88,
                0x98, 0xA8, 0xB8, 0xC8, 0xD8, 0xE8, 0xF8, 0x29, 0x39, 0x49,
                0x59, 0x69, 0x79, 0x89, 0x99, 0xA9, 0xB9, 0xC9, 0xD9, 0xE9,
                0xF9, 0x2A, 0x3A, 0x4A, 0x5A, 0x6A, 0x7A, 0x8A, 0x9A, 0xAA,
                0xBA, 0xCA, 0xDA, 0xEA, 0xFA
            ]
        ])
    table_17 = HuffmanTable(
        17,
        [
            [],
            [0x01, 0x00],
            [0x02, 0x11],
            [0x03],
            [0x04, 0x21],
            [0x12, 0x31, 0x41],
            [0x05, 0x51, 0x13, 0x61, 0x22],
            [0x06, 0x71, 0x81, 0x91, 0x32],
            [0xA1, 0xB1, 0xF0, 0x14],
            [0xC1, 0xD1, 0xE1, 0x23, 0x42],
            [0x15, 0x52, 0x62, 0x72, 0xF1, 0x33],
            [0x24, 0x34, 0x43, 0x82],
            [0x16, 0x92, 0x53, 0x25, 0xA2, 0x63, 0xB2, 0xC2],
            [0x07, 0x73, 0xD2],
            [0x35, 0xE2, 0x44],
            [
                0x83, 0x17, 0x54, 0x93, 0x08, 0x09, 0x0A, 0x18, 0x19, 0x26,
                0x36, 0x45, 0x1A, 0x27, 0x64, 0x74, 0x55, 0x37, 0xF2, 0xA3,
                0xB3, 0xC3, 0x28, 0x29, 0xD3, 0xE3, 0xF3, 0x84, 0x94, 0xA4,
                0xB4, 0xC4, 0xD4, 0xE4, 0xF4, 0x65, 0x75, 0x85, 0x95, 0xA5,
                0xB5, 0xC5, 0xD5, 0xE5, 0xF5, 0x46, 0x56, 0x66, 0x76, 0x86,
                0x96, 0xA6, 0xB6, 0xC6, 0xD6, 0xE6, 0xF6, 0x47, 0x57, 0x67,
                0x77, 0x87, 0x97, 0xA7, 0xB7, 0xC7, 0xD7, 0xE7, 0xF7, 0x38,
                0x48, 0x58, 0x68, 0x78, 0x88, 0x98, 0xA8, 0xB8, 0xC8, 0xD8,
                0xE8, 0xF8, 0x39, 0x49, 0x59, 0x69, 0x79, 0x89, 0x99, 0xA9,
                0xB9, 0xC9, 0xD9, 0xE9, 0xF9, 0x2A, 0x3A, 0x4A, 0x5A, 0x6A,
                0x7A, 0x8A, 0x9A, 0xAA, 0xBA, 0xCA, 0xDA, 0xEA, 0xFA
            ]
        ])

    return JpegHeader(
        huffman_tables=[table_0, table_1, table_16, table_17],
        width=16,
        height=8,
        table_selections={0: (0, 0), 1: (1, 1), 2: (1, 1)}
    )


def create_small_scan(header: JpegHeader) -> JpegScan:
    jpeg_bytes = bytes(
        [0xFC, 0xFF, 0x00, 0xE2, 0xAF, 0xEF, 0xF3, 0x15, 0x7F, 0xFF, 0xD9]
    )
    return JpegScan(header, jpeg_bytes)


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
    def setUpClass(cls):
        cls.tif = TiffFile(tif_test_file_path)
        page: TiffPage = cls.tif.series[0].levels[0].pages[0]
        cls.large_header = JpegHeader.from_bytes(page.jpegheader)
        stripe_offset = page.dataoffsets[0]
        stripe_length = page.databytecounts[0]
        cls.tif.filehandle.seek(stripe_offset)
        stripe: bytes = cls.tif.filehandle.read(stripe_length)
        cls.large_scan = JpegScan(cls.large_header, stripe)
        # f = open(tif_test_data_dir+'/'+"slide.jpeg", "wb")
        # f.write(page.jpegheader)
        # f.write(stripe)
        # f.close()
        cls.small_header = create_small_header()
        cls.small_scan = create_small_scan(cls.small_header)

    @classmethod
    def tearDownClass(cls):
        cls.tif.close()

    def test_huffman_node(self):
        root = HuffmanNode(0)
        self.assertEqual(root.insert(HuffmanLeaf(1), 0), 0b0)
        self.assertEqual(root.insert(HuffmanLeaf(2), 0), 0b1)
        self.assertEqual(root.insert(HuffmanLeaf(3), 0), None)
        self.assertEqual(root.insert(HuffmanLeaf(4), 1), None)

        root = HuffmanNode(0)
        self.assertEqual(root.insert(HuffmanLeaf(1), 0), 0b0)
        self.assertEqual(root.insert(HuffmanLeaf(2), 1), 0b01)
        self.assertEqual(root.insert(HuffmanLeaf(3), 1), 0b11)
        self.assertEqual(root.insert(HuffmanLeaf(4), 1), None)

        root = HuffmanNode(0)
        self.assertEqual(root.insert(HuffmanLeaf(1), 0), 0b0)
        self.assertEqual(root.insert(HuffmanLeaf(2), 1), 0b01)
        self.assertEqual(root.insert(HuffmanLeaf(3), 2), 0b011)
        self.assertEqual(root.insert(HuffmanLeaf(4), 2), 0b111)
        self.assertEqual(root.insert(HuffmanLeaf(5), 2), None)

    def test_huffman_node_full(self):
        root = HuffmanNode(0)
        self.assertFalse(root.full)
        root.insert(HuffmanLeaf(1), 0)
        self.assertFalse(root.full)
        root.insert(HuffmanLeaf(1), 0)
        self.assertTrue(root.full)

    def test_huffman_node_insert_into_self(self):
        root = HuffmanNode(0)
        self.assertEqual(root._insert_into_self(HuffmanLeaf(1), 0), 0b0)
        self.assertEqual(root._insert_into_self(HuffmanLeaf(2), 1), None)
        self.assertEqual(root._insert_into_self(HuffmanLeaf(3), 0), 0b1)
        self.assertEqual(root._insert_into_self(HuffmanLeaf(4), 0), None)

    def test_huffman_node_insert_into_child(self):
        root = HuffmanNode(0)
        root._nodes = [HuffmanNode(1), HuffmanNode(1)]
        self.assertEqual(root._insert_into_child(HuffmanLeaf(1), 1), 0b00)
        self.assertEqual(root._insert_into_child(HuffmanLeaf(2), 1), 0b10)
        self.assertEqual(root._insert_into_child(HuffmanLeaf(3), 1), 0b01)
        self.assertEqual(root._insert_into_child(HuffmanLeaf(4), 1), 0b11)
        self.assertEqual(root._insert_into_child(HuffmanLeaf(5), 1), None)

    def test_huffman_node_insert_into_new_child(self):
        root = HuffmanNode(0)
        self.assertEqual(root._insert_into_new_child(HuffmanLeaf(1), 1), 0b0)
        self.assertEqual(root._insert_into_new_child(HuffmanLeaf(2), 1), 0b1)
        self.assertEqual(root._insert_into_new_child(HuffmanLeaf(3), 1), None)

    def test_huffman(self):
        table_0 = self.large_header.huffman_tables[0]
        codes = {
            0x00: ConstBitStream('0b00'),
            0x01: ConstBitStream('0b010'),
            0x02: ConstBitStream('0b011'),
            0x03: ConstBitStream('0b100'),
            0x04: ConstBitStream('0b101'),
            0x05: ConstBitStream('0b110'),
            0x06: ConstBitStream('0b1110'),
            0x07: ConstBitStream('0b11110'),
            0x08: ConstBitStream('0b111110'),
            0x09: ConstBitStream('0b1111110'),
            0x0A: ConstBitStream('0b11111110'),
            0x0B: ConstBitStream('0b111111110')
        }

        for truth, code in codes.items():
            decoded = table_0.decode_from_bits(code)
            self.assertEqual(truth, decoded)
        print(self.large_header.huffman_tables.keys())
        table_16 = self.large_header.huffman_tables[16]
        codes = {
            0x01: ConstBitStream('0b00'),
            0x02: ConstBitStream('0b01'),
            0x03: ConstBitStream('0b100'),
            0x00: ConstBitStream('0b1010'),
            0x04: ConstBitStream('0b1011'),
            0x11: ConstBitStream('0b1100'),
            0x05: ConstBitStream('0b11010'),
            0x12: ConstBitStream('0b11011'),
            0x21: ConstBitStream('0b11100'),
            0x31: ConstBitStream('0b111010'),
            0x41: ConstBitStream('0b111011'),
            0x06: ConstBitStream('0b1111000'),
            0x13: ConstBitStream('0b1111001'),
            0x51: ConstBitStream('0b1111010'),
            0x61: ConstBitStream('0b1111011'),
            0x24: ConstBitStream('0b111111110100'),
            0x33: ConstBitStream('0b111111110101'),
            0x62: ConstBitStream('0b111111110110'),
            0x72: ConstBitStream('0b111111110111')
        }
        for truth, code in codes.items():
            decoded = table_16.decode_from_bits(code)

            self.assertEqual(truth, decoded)

    def test_mcu_positions(self):
        header_offset = 0x294
        mcu_positions = {
            0: (0x294, 0),
            1: (0x297, 2),
            2: (0x299, 5),
            3: (0x29C, 2),
            22: (0x2C1, 6),
            64: (0x319, 0)
        }
        mcu_positions_without_offset = {
            index: (mcu_position[0]-header_offset, mcu_position[1])
            for index, mcu_position in mcu_positions.items()
        }
        selected_mcu_positions = {
            index: mcu.position
            for index, mcu in enumerate(self.large_scan.mcus)
            if (index in mcu_positions_without_offset.keys())
        }

        self.assertEqual(
            mcu_positions_without_offset,
            selected_mcu_positions
        )

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
        for marker in marker_mapping.keys():
            self.assertEqual(
                marker,
                JpegHeader.read_marker(io.BytesIO(
                    bytes([marker >> 8, marker & 0xFF])
                ))
            )

    def test_small_scan_huffman_table(self):
        data = {
            0: bytes([254]),
            1: bytes([254]),
            16: bytes([248]),
            17: bytes([248])
        }
        decoded_values = [
            table.decode(Stream(data[index]))
            for index, table in self.small_header.huffman_tables.items()
        ]
        actual_values = [0x0B, 0x0A, 0x22, 0x81]
        self.assertEqual(actual_values, decoded_values)

    def test_small_scan_mcus(self):
        actual_mcus = [
            Mcu(position=(0, 0), dc_amplitudes=[9841, 0, 0]),
            Mcu(position=(3, 5), dc_amplitudes=[29520, 0, 0])
        ]
        self.assertEqual(actual_mcus, self.small_scan.mcus)

    def test_table_selection(self):
        selection = self.small_header.table_selections
        self.assertEqual({0: (0, 0), 1: (1, 1), 2: (1, 1)}, selection)

        selection = self.large_header.table_selections
        self.assertEqual({0: (0, 0), 1: (1, 1), 2: (1, 1)}, selection)
