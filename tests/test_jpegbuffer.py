import unittest

import pytest
from ndpi_tiler.jpeg import (JpegBuffer, JpegBufferBitarrayBinary,
                             JpegBufferBitarrayBit, JpegBufferBitarrayDict,
                             JpegBufferBitstringBinary,
                             JpegBufferBitstringDict, JpegBufferByteBinary,
                             JpegBufferByteDict, JpegBufferIntBinary,
                             JpegBufferIntDict, JpegBufferIntList)

from .create_jpeg_data import create_small_scan_data


@pytest.mark.unittest
@pytest.mark.jpegbuffer
class NdpiTilerJpegBufferTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer: JpegBuffer

    @classmethod
    def setUp(cls):
        data = create_small_scan_data()
        buffer_types = [
            JpegBufferByteBinary, JpegBufferByteDict,
            JpegBufferBitarrayBit, JpegBufferBitarrayDict,
            JpegBufferBitarrayBinary, JpegBufferBitstringBinary,
            JpegBufferIntBinary, JpegBufferIntDict, JpegBufferIntList,
            JpegBufferBitstringDict
        ]
        cls.buffers = [
            buffer_type(data) for buffer_type in buffer_types
        ]
        cls.buffer = JpegBufferByteDict(create_small_scan_data())

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
        for buffer in self.buffers:
            for index, expected_result in enumerate(expected_results):
                self.assertEqual(buffer.read(), expected_result)

    def test_read_bits(self):
        for buffer in self.buffers:
            self.assertEqual(buffer.read(4), 0b1111)
            self.assertEqual(buffer.read(4), 0b1100)
            self.assertEqual(buffer.read(5), 0b11111)
            self.assertEqual(buffer.read(7), 0b1111110)
            self.assertEqual(buffer.read(12), 0b001010101111)

    def test_skip_pos(self):
        for buffer in self.buffers:
            buffer.skip(2)
            self.assertEqual(buffer.position, 2)

            buffer.skip(6)
            self.assertEqual(buffer.position, 8)

    def test_skip_read(self):
        for buffer in self.buffers:
            buffer.skip(8)
            self.assertEqual(buffer.read(5), 0b11111)

            buffer.skip(1)
            self.assertEqual(buffer.read(3), 0b111)
            print(self.buffer.position)

            buffer.skip(2)
            self.assertEqual(buffer.read(3), 0b000)
