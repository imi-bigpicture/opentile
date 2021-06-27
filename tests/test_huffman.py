import unittest

import pytest
from bitstring import Bits
from ndpi_tiler.huffman import HuffmanLeaf, HuffmanNode
from ndpi_tiler.jpeg import JpegHeader, JpegScan
from tifffile import TiffFile

from .create_jpeg_data import create_large_set, create_small_set, open_tif


@pytest.mark.unittest
@pytest.mark.huffman
class NdpiTilerHuffmanTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.large_header: JpegHeader
        self.large_scan: JpegScan
        self.small_header: JpegHeader
        self.small_scan: JpegScan
        self.tif: TiffFile

    @classmethod
    def setUp(cls):
        cls.large_scan._stream.seek(0)
        cls.small_scan._stream.seek(0)

    @classmethod
    def setUpClass(cls):
        cls.tif = open_tif()
        (
            cls.large_header,
            cls.large_scan,
            cls.large_offset,
            cls.large_length
        ) = create_large_set(cls.tif)
        (
            cls.small_header,
            cls.small_scan,
            cls.small_offset,
            cls.small_length
        ) = create_small_set()

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
        self.assertEqual(root.insert(HuffmanLeaf(2), 1), 0b10)
        self.assertEqual(root.insert(HuffmanLeaf(3), 1), 0b11)
        self.assertEqual(root.insert(HuffmanLeaf(4), 1), None)

        root = HuffmanNode(0)
        self.assertEqual(root.insert(HuffmanLeaf(1), 0), 0b0)
        self.assertEqual(root.insert(HuffmanLeaf(2), 1), 0b10)
        self.assertEqual(root.insert(HuffmanLeaf(3), 2), 0b110)
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
        self.assertEqual(root._insert_into_child(HuffmanLeaf(2), 1), 0b01)
        self.assertEqual(root._insert_into_child(HuffmanLeaf(3), 1), 0b10)
        self.assertEqual(root._insert_into_child(HuffmanLeaf(4), 1), 0b11)
        self.assertEqual(root._insert_into_child(HuffmanLeaf(5), 1), None)

    def test_huffman_node_insert_into_new_child(self):
        root = HuffmanNode(0)
        self.assertEqual(root._insert_into_new_child(HuffmanLeaf(1), 1), 0b00)
        self.assertEqual(root._insert_into_new_child(HuffmanLeaf(2), 1), 0b10)
        self.assertEqual(root._insert_into_new_child(HuffmanLeaf(3), 1), None)

    def test_huffman(self):
        DC_0 = self.large_header.components['Y'].dc_table
        codes = {
            0x00: Bits('0b00'),
            0x01: Bits('0b010'),
            0x02: Bits('0b011'),
            0x03: Bits('0b100'),
            0x04: Bits('0b101'),
            0x05: Bits('0b110'),
            0x06: Bits('0b1110'),
            0x07: Bits('0b11110'),
            0x08: Bits('0b111110'),
            0x09: Bits('0b1111110'),
            0x0A: Bits('0b11111110'),
            0x0B: Bits('0b111111110')
        }

        for truth, code in codes.items():
            decoded = DC_0.decode_from_bits(code)
            self.assertEqual(truth, decoded)

        AC_0 = self.large_header.components['Y'].ac_table

        codes = {
            0x01: Bits('0b00'),
            0x02: Bits('0b01'),
            0x03: Bits('0b100'),
            0x00: Bits('0b1010'),
            0x04: Bits('0b1011'),
            0x11: Bits('0b1100'),
            0x05: Bits('0b11010'),
            0x12: Bits('0b11011'),
            0x21: Bits('0b11100'),
            0x31: Bits('0b111010'),
            0x41: Bits('0b111011'),
            0x06: Bits('0b1111000'),
            0x13: Bits('0b1111001'),
            0x51: Bits('0b1111010'),
            0x61: Bits('0b1111011'),
            0x24: Bits('0b111111110100'),
            0x33: Bits('0b111111110101'),
            0x62: Bits('0b111111110110'),
            0x72: Bits('0b111111110111')
        }
        for truth, code in codes.items():
            decoded = AC_0.decode_from_bits(code)

            self.assertEqual(truth, decoded)

    # def test_small_scan_huffman_table(self):
    #     data = {
    #         HuffmanTableIdentifier('DC', 0): bytes([254]),
    #         HuffmanTableIdentifier('DC', 1): bytes([254]),
    #         HuffmanTableIdentifier('AC', 0): bytes([248]),
    #         HuffmanTableIdentifier('AC', 1): bytes([248])
    #     }
    #     decoded_values = [
    #         table.decode(Stream(data[index]))
    #         for index, table in self.small_header.huffman_tables.items()
    #     ]
    #     actual_values = [0x0B, 0x0A, 0x22, 0x81]
    #     self.assertEqual(actual_values, decoded_values)

    def test_code_decode(self):
        for header in [self.small_header, self.large_header]:
            for component in header.components.values():
                for table in [component.dc_table, component.ac_table]:
                    for value in table.encode_dict.keys():
                        symbol = table.encode_into_bits(value)
                        decoded = table.decode_from_bits(symbol)
                        self.assertEqual(value, decoded)
