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

test_data_dir = os.environ.get('OPENTILE_TESTDIR', 'tests/testdata')
ndpi_file_path = Path(test_data_dir).joinpath('slides/ndpi/CMU-1/CMU-1.ndpi')
svs_file_path = Path(test_data_dir).joinpath('slides/svs/CMU-1/CMU-1.svs')


@pytest.mark.unittest
class JpegTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.jpeg = Jpeg()

    @classmethod
    def setUpClass(cls):
        try:
            cls.ndpi_tiff = TiffFile(ndpi_file_path)
            ndpi_level = cls.ndpi_tiff.series[0].levels[0].pages[0]
            assert isinstance(ndpi_level, TiffPage)
            cls.ndpi_level = ndpi_level
            header = cls.ndpi_level.jpegheader
            assert isinstance(header, bytes)
            cls.ndpi_header = header
            cls.svs_tiff = TiffFile(svs_file_path)
            svs_overview = cls.svs_tiff.series[3].pages[0]
            assert isinstance(svs_overview, TiffPage)
            cls.svs_overview = svs_overview
        except FileNotFoundError:
            raise unittest.SkipTest("Svs or ndpi test file not found, skipping")

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
        index, length = Jpeg._find_tag(self.ndpi_header, Jpeg.start_of_frame())
        self.assertEqual(621, index)
        self.assertEqual(17, length)

    def test_update_header(self):
        target_size = Size(512, 200)
        updated_header = Jpeg.manipulate_header(self.ndpi_header, target_size)
        (stripe_width, stripe_height, _, _) = self.jpeg._turbo_jpeg.decode_header(
            updated_header
        )
        self.assertEqual(target_size, Size(stripe_width, stripe_height))

    def test_concatenate_fragments(self):
        frame = self.jpeg.concatenate_fragments(
            (
                self.read_frame(self.ndpi_tiff, self.ndpi_level, index)
                for index in range(10)
            ),
            self.ndpi_header,
        )
        self.assertEqual("ea40e78b081c42a6aabf8da81f976f11", md5(frame).hexdigest())

    def test_concatenate_scans(self):
        frame = self.jpeg.concatenate_scans(
            (
                self.read_frame(self.svs_tiff, self.svs_overview, index)
                for index in range(len(self.svs_overview.databytecounts))
            ),
            self.svs_overview.jpegtables,
            True,
        )
        self.assertEqual("fdde19f6d10994c5b866b43027ff94ed", md5(frame).hexdigest())

    def test_code_short(self):
        self.assertEqual(bytes([0x00, 0x06]), self.jpeg.code_short(6))
