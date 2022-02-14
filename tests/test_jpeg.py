#    Copyright 2021 SECTRA AB
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import unittest
from hashlib import md5
from pathlib import Path

import pytest
from opentile.geometry import Size
from opentile.jpeg import Jpeg
from tifffile import TiffFile, TiffPage

ndpi_test_data_dir = os.environ.get(
    "NDPI_TESTDIR",
    "C:/temp/opentile/ndpi/"
)
sub_data_path = "ndpi2/input.ndpi"
ndpi_file_path = Path(ndpi_test_data_dir + '/' + sub_data_path)

svs_test_data_dir = os.environ.get(
    "OPEN_TILER_TESTDIR",
    "C:/temp/opentile/svs/"
)
sub_data_path = "svs1/input.svs"
svs_file_path = Path(svs_test_data_dir + '/' + sub_data_path)

turbojpeg_path = Path('C:/libjpeg-turbo64/bin/turbojpeg.dll')


@pytest.mark.unittest
class JpegTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.jpeg = Jpeg(turbojpeg_path)

    @classmethod
    def setUpClass(cls):
        cls.ndpi_tiff = TiffFile(ndpi_file_path)
        cls.ndpi_level = cls.ndpi_tiff.series[0].levels[0].pages[0]
        cls.svs_tiff = TiffFile(svs_file_path)
        cls.svs_overview = cls.svs_tiff.series[3].pages[0]

    @classmethod
    def tearDownClass(cls):
        cls.ndpi_tiff.close()
        cls.svs_tiff.close()

    @staticmethod
    def read_frame(tiff: TiffFile, level: TiffPage, index: int) -> bytes:
        offset = level.dataoffsets[index]
        length = level.databytecounts[index]
        tiff.filehandle.seek(offset)
        return tiff.filehandle.read(length)

    def test_tags(self):
        self.assertEqual(Jpeg.start_of_frame(), bytes([0xFF, 0xC0]))
        self.assertEqual(Jpeg.end_of_image(), bytes([0xFF, 0xD9]))
        self.assertEqual(Jpeg.restart_mark(0), bytes([0xD0]))
        self.assertEqual(Jpeg.restart_mark(7), bytes([0xD7]))
        self.assertEqual(Jpeg.restart_mark(9), bytes([0xD1]))

    def test_find_tag(self):
        header = self.ndpi_level.jpegheader
        index, length = Jpeg._find_tag(header, Jpeg.start_of_frame())
        self.assertEqual(621, index)
        self.assertEqual(17, length)

    def test_update_header(self):
        target_size = Size(512, 200)
        updated_header = Jpeg.manipulate_header(
            self.ndpi_level.jpegheader,
            target_size
        )
        (
            stripe_width,
            stripe_height,
            _, _
        ) = self.jpeg._turbo_jpeg.decode_header(updated_header)
        self.assertEqual(target_size, Size(stripe_width, stripe_height))

    def test_concatenate_fragments(self):
        frame = self.jpeg.concatenate_fragments(
            (
                self.read_frame(self.ndpi_tiff, self.ndpi_level, index)
                for index in range(10)
            ),
            self.ndpi_level.jpegheader
        )
        self.assertEqual(
            '3b13a2a65a8f0b026eb1822864b0af6a',
            md5(frame).hexdigest()
        )

    def test_concatenate_scans(self):
        frame = self.jpeg.concatenate_scans(
            (
                self.read_frame(self.svs_tiff, self.svs_overview, index)
                for index in range(len(self.svs_overview.databytecounts))
            ),
            self.svs_overview.jpegtables,
            True
        )
        self.assertEqual(
            '75bb45ee9c3135b0ade2427c6f673609',
            md5(frame).hexdigest()
        )

    def test_code_short(self):
        self.assertEqual(
            bytes([0x00, 0x06]),
            self.jpeg.code_short(6)
        )
