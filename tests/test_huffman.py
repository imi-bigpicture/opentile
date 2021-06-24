import unittest

import pytest
from bitstring import ConstBitStream
from ndpi_tiler.huffman import (HuffmanLeaf, HuffmanNode,
                                HuffmanTableIdentifier, HuffmanTableSelection)
from ndpi_tiler.jpeg import JpegHeader, JpegScan
from ndpi_tiler.stream import Stream
from tifffile import TiffFile

from .create_jpeg_data import (create_large_header, create_large_scan,
                               create_large_scan_data, create_small_header,
                               create_small_scan, get_page, open_tif)


@pytest.mark.unittest
class NdpiTilerHuffmanTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.large_header: JpegHeader
        self.large_scan: JpegScan
        self.small_header: JpegHeader
        self.small_scan: JpegScan
        self.tif: TiffFile

    @classmethod
    def setUpClass(cls):
        cls.tif = open_tif()
        cls.large_header = create_large_header(get_page(cls.tif))
        cls.large_scan_data = create_large_scan_data(cls.tif)
        cls.large_scan = create_large_scan(
            cls.large_header,
            cls.large_scan_data
        )
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
        DC_0 = self.large_header.huffman_tables[
            HuffmanTableIdentifier('DC', 0)
        ]
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
            decoded = DC_0.decode_from_bits(code)
            self.assertEqual(truth, decoded)

        print(self.large_header.huffman_tables.keys())
        AC_0 = self.large_header.huffman_tables[
            HuffmanTableIdentifier('AC', 0)
        ]
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
            decoded = AC_0.decode_from_bits(code)

            self.assertEqual(truth, decoded)

    def test_small_scan_huffman_table(self):
        data = {
            HuffmanTableIdentifier('DC', 0): bytes([254]),
            HuffmanTableIdentifier('DC', 1): bytes([254]),
            HuffmanTableIdentifier('AC', 0): bytes([248]),
            HuffmanTableIdentifier('AC', 1): bytes([248])
        }
        decoded_values = [
            table.decode(Stream(data[index]))
            for index, table in self.small_header.huffman_tables.items()
        ]
        actual_values = [0x0B, 0x0A, 0x22, 0x81]
        self.assertEqual(actual_values, decoded_values)

    def test_components(self):
        actual_components = {
                'Y': HuffmanTableSelection(dc=0, ac=0),
                'Cb': HuffmanTableSelection(dc=1, ac=1),
                'Cr': HuffmanTableSelection(dc=1, ac=1)
            }
        components = self.small_header.components
        self.assertEqual(actual_components, components)

        components = self.large_header.components
        self.assertEqual(actual_components, components)
