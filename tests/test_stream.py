import unittest
import io


import pytest
from ndpi_tiler.stream import Stream

from .create_jpeg_data import create_small_scan_data


@pytest.mark.unittest
@pytest.mark.stream
class NdpiTilerStreamTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stream: Stream

    @classmethod
    def setUp(cls):
        fh, offset = create_small_scan_data()
        cls.stream = Stream(fh, offset)

    @classmethod
    def tearDown(cls):
        pass

    def test_read_bit(self):
        expected_results = [
            1, 1, 1, 1, 1, 1, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1,
            # 0, 0, 0, 0, 0, 0, 0, 0 # bit stuffing
            1, 1, 1, 0, 0, 0, 1, 0,
            1, 0, 1, 0, 1, 1, 1, 1,
            1, 1, 1, 0, 1, 1, 1, 1,
            1, 1, 1, 1, 0, 0, 1, 1,
            0, 0, 0, 1, 0, 1, 0, 1,
            0, 1, 1, 1, 1, 1, 1, 1
        ]
        for index, expected_result in enumerate(expected_results):
            print(index)
            self.assertEqual(self.stream.read(), expected_result)

    def test_read_bits(self):
        self.assertEqual(self.stream.read(4), 0b1111)
        self.assertEqual(self.stream.read(4), 0b1100)
        self.assertEqual(self.stream.read(5), 0b11111)
        self.assertEqual(self.stream.read(7), 0b1111110)
        self.assertEqual(self.stream.read(12), 0b001010101111)

    def test_seek(self):
        self.stream.seek(18)
        self.assertEqual(self.stream.pos, 18)

        self.stream.seek(7)
        self.assertEqual(self.stream.pos, 7)

        self.stream.seek(25)
        self.assertEqual(self.stream.pos, 25)

        self.stream.seek(32)
        self.assertEqual(self.stream.pos, 32)

        self.stream.seek(31)
        self.assertEqual(self.stream.pos, 31)

    def test_seek_read(self):
        self.stream.read(4)
        self.stream.seek(0)
        self.assertEqual(self.stream.read(4), 0b1111)

        self.stream.seek(24)
        self.assertEqual(self.stream.read(6), 0b111000)

        self.stream.seek(30)
        self.assertEqual(self.stream.read(5), 0b10101)

        self.stream.seek(14)
        self.assertEqual(self.stream.read(2), 0b11)

    def test_skip_pos(self):
        self.stream.skip(2)
        self.assertEqual(self.stream.pos, 2)

        self.stream.skip(6)
        self.assertEqual(self.stream.pos, 8)

    def test_skip_read(self):
        self.stream.skip(8)
        self.assertEqual(self.stream.read(5), 0b11111)

        self.stream.skip(1)
        self.assertEqual(self.stream.read(3), 0b111)
        print(self.stream.pos)

        self.stream.skip(2)
        self.assertEqual(self.stream.read(3), 0b000)
