import dataclasses
import io
import unittest
from struct import unpack
from typing import List

import pytest
from ndpi_tiler.jpeg import JpegBuffer, JpegHeader, JpegScan, JpegSegment
from ndpi_tiler.jpeg_tags import MARER_MAPPINGS
from tifffile import TiffFile

from .create_jpeg_data import create_large_set, create_small_set, open_tif


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
        cls.large_scan._buffer.seek(0)
        cls.small_scan._buffer.seek(0)

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
            start=start,
            end=end,
            count=2,
            dc_offsets={'Y': 0, 'Cb': 0, 'Cr': 0},
            dc_sums={'Y': 508, 'Cb': 0, 'Cr': 0}
        )
        read_segment = self.small_scan.read_segment(
            2,
            {'Y': 0, 'Cb': 0, 'Cr': 0}
        )
        self.assertEqual(true_segment, read_segment)

    def test_large_scan_read_segments(self):
        # Need to check dc sum
        start = 8*0+0
        end = 8*1135+6
        true_segment = JpegSegment(
            start=start,
            end=end,
            count=512,
            dc_offsets={'Y': 0, 'Cb': 0, 'Cr': 0},
            dc_sums={'Y': 81, 'Cb': 2, 'Cr': 0}
        )

        read_segment = self.large_scan.read_segment(
            512,
            {'Y': 0, 'Cb': 0, 'Cr': 0}
        )

        self.assertEqual(true_segment, read_segment)

    def test_large_scan_read_mcus(self):
        # Header offset, as positions are readed from jpeg
        header_offset = 0x294
        Mcu = dataclasses.make_dataclass(
            'mcu',
            [('position', int), ('dc_sum', List[int])]
        )
        true_mcus = {
            0: Mcu(
                position=8*(0x294-header_offset) + 0,
                dc_sum={'Y': 80, 'Cb': 2, 'Cr': 0}
            ),
            1: Mcu(
                position=8*(0x297-header_offset) + 2,
                dc_sum={'Y': 1, 'Cb': 0, 'Cr': 0}
            ),
            150: Mcu(
                position=8*(0x3D4-header_offset) + 5,
                dc_sum={'Y': 0, 'Cb': 0, 'Cr': 0}
            ),
            151: Mcu(
                position=8*(0x3D6-header_offset) + 3,
                dc_sum={'Y': 0, 'Cb': 1, 'Cr': 0}
            ),
            510: Mcu(
                position=8*(0x700-header_offset) + 0,
                dc_sum={'Y': -1, 'Cb': 0, 'Cr': 0}
            ),
            511: Mcu(
                position=8*(0x702-header_offset) + 0,
                dc_sum={'Y': 0, 'Cb': 0, 'Cr': 0}
            )
        }
        self.large_scan._buffer.seek(0)
        read_mcus = {
            index: Mcu(
                position=self.large_scan._buffer.pos,
                dc_sum=self.large_scan._read_mcu({'Y': 0, 'Cb': 0, 'Cr': 0})
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
