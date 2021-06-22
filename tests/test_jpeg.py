import io
import unittest
from struct import unpack

import pytest
from ndpi_tiler.jpeg import JpegHeader, JpegScan, Mcu, McuBlock, SegmentStub
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
        actual_segment = SegmentStub(
            first_mcu=Mcu(
                [
                    McuBlock(0, -512),
                    McuBlock(29, 0),
                    McuBlock(33, 0)
                ]
            ),
            scan_start=37,
            scan_end=66,
            dc=0
        )
        data = create_small_scan_data()
        stream = Stream(data)
        segment = self.small_scan._extract_segment(stream, 2)
        print(segment)
        print(actual_segment)
        self.assertEqual(actual_segment, segment)

    # def test_large_scan_extract_segments(self):
    #     header_offset = 0x294

    #     actual_segment = SegmentStub (
    #         first_mcu=Mcu(
    #             [
    #                 McuBlock(0x294+0-header_offset, 140),
    #                 McuBlock(0x295+7-header_offset, 0),
    #                 McuBlock(0x298+1-header_offset, 0)
    #             ]
    #         ),
    #         scan_start=0x298+6-header_offset,
    #         scan_end=0x86E+7+2-header_offset,
    #         dc=0
    #     )
    #     data = create_large_scan_data(self.tif)
    #     stream = Stream(data)
    #     print(data.hex())
    #     segments = self.small_scan._extract_segment(stream, 512)

    #     self.assertEqual(
    #         actual_segment,
    #         segments
    #    )

    def test_large_scan_read_mcus(self):
        header_offset = 8*0x294

        actual_smus = {
            0: Mcu(
                [
                    McuBlock(8*0x294+0-header_offset, 80),
                    McuBlock(8*0x296+0-header_offset, 2),
                    McuBlock(8*0x296+6-header_offset, 0)
                ]
            ),
            1: Mcu(
                [
                    McuBlock(8*0x297+2-header_offset, 1),
                    McuBlock(8*0x298+2-header_offset, 0),
                    McuBlock(8*0x298+6-header_offset, 0)
                ]
            ),
            150: Mcu(
                [
                    McuBlock(8*0x3D4+5-header_offset, 0),
                    McuBlock(8*0x3D5+3-header_offset, 0),
                    McuBlock(8*0x3D5+7-header_offset, 0)
                ]
            ),
            151: Mcu(
                [
                    McuBlock(8*0x3D6+3-header_offset, 0),
                    McuBlock(8*0x3D7+1-header_offset, 1),
                    McuBlock(8*0x3D7+6-header_offset, 0)
                ]
            ),
            510: Mcu(
                [
                    McuBlock(8*0x700+0-header_offset, -1),
                    McuBlock(8*0x701+0-header_offset, 0),
                    McuBlock(8*0x701+4-header_offset, 0)
                ]
            ),
            511: Mcu(
                [
                    McuBlock(8*0x702+0-header_offset, 0),
                    McuBlock(8*0x702+6-header_offset, 0),
                    McuBlock(8*0x703+2-header_offset, 0)
                ]
            )
        }
        data = create_large_scan_data(self.tif)
        stream = Stream(data)
        mcus = {
            index: self.large_scan._read_mcu(stream)
            for index in range(self.large_scan._mcu_count)
        }

        for index in actual_smus.keys():
            print(index)
            print(actual_smus[index])
            print(mcus[index])
            self.assertEqual(
                actual_smus[index],
                mcus[index]
            )
