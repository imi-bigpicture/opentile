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

from datetime import datetime
import unittest
from hashlib import md5

import pytest
from opentile.philips_tiff_tiler import PhilipsTiffTiler

from .filepaths import philips_file_path


@pytest.mark.unittest
class PhilipsTiffTilerTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tiler: PhilipsTiffTiler

    @classmethod
    def setUpClass(cls):
        try:
            cls.tiler = PhilipsTiffTiler(philips_file_path)
        except FileNotFoundError:
            raise unittest.SkipTest(
                'Philips tiff test file not found, skipping'
            )
        cls.level = cls.tiler.get_level(0)

    @classmethod
    def tearDownClass(cls):
        cls.tiler.close()

    def test_get_tile(self):
        tile = self.level.get_tile((0, 0))
        self.assertEqual(
            '570d069f9de5d2716fb0d7167bc79195',
            md5(tile).hexdigest()
        )
        tile = self.level.get_tile((20, 20))
        self.assertEqual(
            'db28efb73a72ef7e2780fc72c624d7ae',
            md5(tile).hexdigest()
        )

    def test_photometric_interpretation(self):
        self.assertEqual(
            'YCBCR',
            self.tiler.get_level(0).photometric_interpretation
        )

    def test_subsampling(self):
        self.assertEqual(
            (2, 2),
            self.tiler.get_level(0).subsampling
        )

    def test_sumples_per_pixel(self):
        self.assertEqual(
            3,
            self.tiler.get_level(0).samples_per_pixel
        )

    def test_metadata_scanner_manufacturer(self):
        self.assertEqual(
            'PHILIPS',
            self.tiler.metadata.scanner_manufacturer
        )

    def test_metadata_scanner_software_versions(self):
        self.assertEqual(
            ['1.6.5505', '20111209_R44', '4.0.3'],
            self.tiler.metadata.scanner_software_versions
        )

    def test_metadata_scanner_serial_number(self):
        self.assertIsNotNone(
            self.tiler.metadata.scanner_serial_number
        )

    def test_metadata_aquisition_datetime(self):
        self.assertEqual(
            datetime(2013, 7, 1, 18, 59, 4),
            self.tiler.metadata.aquisition_datetime
        )
