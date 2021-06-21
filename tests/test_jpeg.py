import io
import unittest
from struct import unpack

import pytest
from ndpi_tiler.jpeg import JpegHeader, JpegScan, Mcu
from ndpi_tiler.jpeg_tags import MARER_MAPPINGS
from tifffile import TiffFile

from .create_jpeg_data import (create_large_header, create_large_scan,
                               create_small_header, create_small_scan,
                               get_page, open_tif)


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
        cls.large_scan = create_large_scan(
            cls.large_header,
            get_page(cls.tif),
            cls.tif.filehandle)
        cls.small_header = create_small_header()
        cls.small_scan = create_small_scan(cls.small_header)

    @classmethod
    def tearDownClass(cls):
        cls.tif.close()

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
        for marker in MARER_MAPPINGS.keys():
            self.assertEqual(
                marker,
                JpegHeader.read_marker(io.BytesIO(
                    bytes([marker >> 8, marker & 0xFF])
                ))
            )

    def test_small_scan_mcus(self):
        actual_mcus = [
            Mcu(position=(0, 0), dc_amplitudes=[9841, 0, 0]),
            Mcu(position=(3, 5), dc_amplitudes=[29520, 0, 0])
        ]
        self.assertEqual(actual_mcus, self.small_scan.mcus)
