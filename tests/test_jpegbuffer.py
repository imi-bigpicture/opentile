import unittest

import pytest
from ndpi_tiler.jpeg import JpegBuffer

from .create_jpeg_data import create_small_scan_data


@pytest.mark.unittest
@pytest.mark.jpegbuffer
class NdpiTilerJpegBufferTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer: JpegBuffer

    @classmethod
    def setUp(cls):
        cls.buffer = JpegBuffer(create_small_scan_data())

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
            self.assertEqual(self.buffer.read(), expected_result)

    def test_read_bits(self):
        self.assertEqual(self.buffer.read(4), 0b1111)
        self.assertEqual(self.buffer.read(4), 0b1100)
        self.assertEqual(self.buffer.read(5), 0b11111)
        self.assertEqual(self.buffer.read(7), 0b1111110)
        self.assertEqual(self.buffer.read(12), 0b001010101111)

    def test_seek(self):
        self.buffer.seek(18)
        self.assertEqual(self.buffer.pos, 18)

        self.buffer.seek(7)
        self.assertEqual(self.buffer.pos, 7)

        self.buffer.seek(25)
        self.assertEqual(self.buffer.pos, 25)

        self.buffer.seek(32)
        self.assertEqual(self.buffer.pos, 32)

        self.buffer.seek(31)
        self.assertEqual(self.buffer.pos, 31)

    def test_seek_read(self):
        self.buffer.read(4)
        self.buffer.seek(0)
        self.assertEqual(self.buffer.read(4), 0b1111)

        self.buffer.seek(24)
        self.assertEqual(self.buffer.read(6), 0b101011)

        self.buffer.seek(30)
        self.assertEqual(self.buffer.read(5), 0b11111)

        self.buffer.seek(14)
        self.assertEqual(self.buffer.read(2), 0b11)

    def test_skip_pos(self):
        self.buffer.skip(2)
        self.assertEqual(self.buffer.pos, 2)

        self.buffer.skip(6)
        self.assertEqual(self.buffer.pos, 8)

    def test_skip_read(self):
        self.buffer.skip(8)
        self.assertEqual(self.buffer.read(5), 0b11111)

        self.buffer.skip(1)
        self.assertEqual(self.buffer.read(3), 0b111)
        print(self.buffer.pos)

        self.buffer.skip(2)
        self.assertEqual(self.buffer.read(3), 0b000)
