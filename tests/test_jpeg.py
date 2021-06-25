import dataclasses
import io
import unittest
from struct import unpack
from typing import List
import pytest
from bitstring import BitArray
from ndpi_tiler.jpeg import JpegHeader, JpegScan, JpegSegment
from ndpi_tiler.jpeg_tags import MARER_MAPPINGS
from ndpi_tiler.stream import Stream
from tifffile import TiffFile

from .create_jpeg_data import (create_large_header, create_large_scan,
                               create_large_scan_data, create_small_header,
                               create_small_scan, create_small_scan_data,
                               get_page, open_tif, save_scan_as_jpeg)


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
        cls.tif = open_tif()
        cls.large_header = create_large_header(get_page(cls.tif))
        cls.large_scan_data = create_large_scan_data(cls.tif)
        cls.large_scan = create_large_scan(
            cls.large_header,
            cls.large_scan_data
        )
        cls.small_header = create_small_header()
        cls.small_scan = create_small_scan(cls.small_header)
        save_scan_as_jpeg(get_page(cls.tif).jpegheader, cls.large_scan_data)

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

    def test_small_scan_extract_segments(self):
        data = create_small_scan_data()
        actual_segment = JpegSegment(
            0,
            66,
            2,
            {'Y': 0, 'Cb': 0, 'Cr': 0},
            {'Y': 508, 'Cb': 0, 'Cr': 0}
        )
        self.small_scan._stream.seek(0)
        segment = self.small_scan._extract_segment(
            2,
            {'Y': 0, 'Cb': 0, 'Cr': 0}
        )
        print(segment)
        print(actual_segment)
        self.assertEqual(actual_segment, segment)

    def test_large_scan_extract_segments(self):
        self.large_scan._stream.seek(0)
        data = self.large_scan_data
        # Need to check dc sum
        actual_segment = JpegSegment(
            0,
            9086,
            length=512,
            dc_offset={'Y': 0, 'Cb': 0, 'Cr': 0},
            dc_sum={'Y': 81, 'Cb': 2, 'Cr': 0}
        )
        stream = Stream(data)
        segment = self.large_scan._extract_segment(
            512,
            {'Y': 0, 'Cb': 0, 'Cr': 0}
        )
        self.assertEqual(
            actual_segment,
            segment
        )

    def test_large_scan_read_mcus(self):
        header_offset = 8*0x294
        Mcu = dataclasses.make_dataclass(
            'mcu',
            [('position', int), ('dc_sum', List[int])]
        )
        actual_mcus = {
            0: Mcu(
                position=8*0x294+0-header_offset,
                dc_sum={'Y': 80, 'Cb': 2, 'Cr': 0}
            ),
            1: Mcu(
                position=8*0x297+2-header_offset,
                dc_sum={'Y': 1, 'Cb': 0, 'Cr': 0}
            ),
            150: Mcu(
                position=8*0x3D4+5-header_offset,
                dc_sum={'Y': 0, 'Cb': 0, 'Cr': 0}
            ),
            151: Mcu(
                position=8*0x3D6+3-header_offset,
                dc_sum={'Y': 0, 'Cb': 1, 'Cr': 0}
            ),
            510: Mcu(
                position=8*0x700+0-header_offset,
                dc_sum={'Y': -1, 'Cb': 0, 'Cr': 0}
            ),
            511: Mcu(
                position=8*0x702+0-header_offset,
                dc_sum={'Y': 0, 'Cb': 0, 'Cr': 0}
            )
        }
        self.large_scan._stream.seek(0)
        read_mcus = {
            index: Mcu(
                position=self.large_scan._stream.pos,
                dc_sum=self.large_scan._read_mcu({'Y': 0, 'Cb': 0, 'Cr': 0})
            )
            for index in range(self.large_scan._mcu_count)
        }

        for index, value in actual_mcus.items():
            self.assertEqual(value, read_mcus[index])

    def test_code_decode(self):
        for value in range(-2047, 2047):
            length, code = JpegScan._code_value(value)
            decode = JpegScan._decode_value(length, code)
            self.assertEqual(value, decode)
