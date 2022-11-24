#    Copyright 2022 SECTRA AB
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

import unittest
from hashlib import md5

import pytest
from tifffile.tifffile import PHOTOMETRIC

from opentile.histech_tiff_tiler import HistechTiffTiler

from .filepaths import histech_file_path


@pytest.mark.unittest
class HistechTiffTilerTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tiler: HistechTiffTiler

    @classmethod
    def setUpClass(cls):
        try:
            cls.tiler = HistechTiffTiler(histech_file_path)
        except FileNotFoundError:
            raise unittest.SkipTest(
                'Histech tiff test file not found, skipping'
            )
        cls.level = cls.tiler.get_level(0)

    @classmethod
    def tearDownClass(cls):
        cls.tiler.close()

    def test_get_tile(self):
        tile = self.level.get_tile((50, 50))
        self.assertEqual(
            '221f47792ebef7b9e394fc6c8ed7cb64',
            md5(tile).hexdigest()
        )

        tile = self.level.get_tile((100, 100))
        self.assertEqual(
            '0a2e459e94e9237bb866b3bc1ac10cb8',
            md5(tile).hexdigest()
        )

    def test_photometric_interpretation(self):
        self.assertEqual(
            PHOTOMETRIC.YCBCR,
            self.tiler.get_level(0).photometric_interpretation
        )

    def test_subsampling(self):
        self.assertEqual(
            None,
            self.tiler.get_level(0).subsampling
        )

    def test_sumples_per_pixel(self):
        self.assertEqual(
            3,
            self.tiler.get_level(0).samples_per_pixel
        )
