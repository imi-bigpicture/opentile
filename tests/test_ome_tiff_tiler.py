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
from opentile.ome_tiff import OmeTiffTiler

from .filepaths import ome_tiff_file_path


@pytest.mark.unittest
class OmeTiffTilerTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tiler: OmeTiffTiler

    @classmethod
    def setUpClass(cls):
        try:
            cls.tiler = OmeTiffTiler(ome_tiff_file_path)
        except FileNotFoundError:
            raise unittest.SkipTest(
                'Ome tiff test file not found, skipping'
            )
        cls.level = cls.tiler.get_level(0)

    @classmethod
    def tearDownClass(cls):
        cls.tiler.close()

    def test_get_tile(self):
        tile = self.level.get_tile((0, 0))
        self.assertEqual(
            md5(tile).hexdigest(),
            '646c70833b30aab095950424b59a0cf7',
        )

        tile = self.level.get_tile((20, 20))
        self.assertEqual(
            md5(tile).hexdigest(),
            '4c37c335b697aaf1550f77fd9e367f69',
        )

    def test_subsampling(self):
        self.assertIsNone(self.tiler.get_level(0).subsampling)

    def test_sumples_per_pixel(self):
        self.assertEqual(
            self.tiler.get_level(0).samples_per_pixel,
            3
        )
