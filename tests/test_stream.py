import unittest
from struct import unpack

import pytest
from ndpi_tiler.stream import Stream
from .create_jpeg_data import create_small_scan_data

@pytest.mark.unittest
class NdpiTilerStreamTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
         super().__init__(*args, **kwargs)
         self.stream: Stream

    @classmethod
    def setUp(cls):
        data = create_small_scan_data()
        print(data.hex())
        cls.stream = Stream(data)

    @classmethod
    def tearDown(cls):
        pass

    def test_read_bit(self):
        expected_results = [
            1, 1, 1, 1, 1, 1, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1,
            # 0, 0, 0, 0, 0, 0, 0, 0 # bit stuffing
            1, 1, 1, 0, 0, 0, 1, 0,
            1, 0, 1, 0, 1, 1, 1, 1
        ]
        for expected_result in expected_results:
            self.assertEqual(self.stream.read_bit(), expected_result)

    def test_read_bits(self):
        self.assertEqual(self.stream.read_bits(4), 0b1111)
        self.assertEqual(self.stream.read_bits(4), 0b1100)
        self.assertEqual(self.stream.read_bits(5), 0b11111)
        self.assertEqual(self.stream.read_bits(7), 0b1111110)
        self.assertEqual(self.stream.read_bits(12), 0b001010101111)

    def test_seek(self):
        self.stream.seek(18)
        self.assertEqual(self.stream.pos, 18+8)

        self.stream.seek(7)
        self.assertEqual(self.stream.pos, 7)

        self.stream.seek(25)
        self.assertEqual(self.stream.pos, 25)

        self.stream.seek(32)
        self.assertEqual(self.stream.pos, 32)

        self.stream.seek(31)
        self.assertEqual(self.stream.pos, 31)

    def test_seek_read(self):
        self.stream.read_bits(4)
        self.stream.seek(0)
        self.assertEqual(self.stream.read_bits(4), 0b1111)

        self.stream.seek(16)
        self.assertEqual(self.stream.read_bits(6), 0b111000)

        self.stream.seek(18)
        self.assertEqual(self.stream.read_bits(5), 0b10001)

        self.stream.seek(14)
        self.assertEqual(self.stream.read_bits(2), 0b11)

    def test_skip_pos(self):
        self.stream.skip(2)
        self.assertEqual(self.stream.pos, 2)

        self.stream.skip(6)
        self.assertEqual(self.stream.pos, 8)


    def test_skip_read(self):
        self.stream.skip(8)
        self.assertEqual(self.stream.read_bits(5), 0b11111)
        print(self.stream.pos)

        self.stream.skip(1)
        self.assertEqual(self.stream.read_bits(3), 0b111)

        self.stream.skip(2)
        self.assertEqual(self.stream.read_bits(3), 0b000)