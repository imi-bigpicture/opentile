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
from typing import cast

import pytest
from opentile.geometry import Point
from opentile.svs_tiler import SvsTiledPage, SvsTiler

svs_test_data_dir = os.environ.get(
    "OPENTILE_TESTDIR",
    "C:/temp/opentile/"
)
sub_data_path = "svs/CMU-1/CMU-1.svs"
svs_file_path = Path(svs_test_data_dir + '/' + sub_data_path)
turbojpeg_path = Path('C:/libjpeg-turbo64/bin/turbojpeg.dll')


@pytest.mark.unittest
class SvsTilerTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tiler: SvsTiler

    @classmethod
    def setUpClass(cls):
        try:
            cls.tiler = SvsTiler(svs_file_path, turbojpeg_path)
        except FileNotFoundError:
            raise unittest.SkipTest('Svs test file not found, skipping')
        cls.level = cast(SvsTiledPage, cls.tiler.get_level(0))

    @classmethod
    def tearDownClass(cls):
        cls.tiler.close()

    def test_get_tile(self):
        tile = self.level.get_tile((15, 25))
        self.assertEqual(
            'dd8402d5a7250ebfe9e33b3dfee3c1e1',
            md5(tile).hexdigest()
        )
        tile = self.level.get_tile((35, 30))
        self.assertEqual(
            '50f885cd7299bd3fff22743bfb4a4930',
            md5(tile).hexdigest()
        )

    def test_get_scaled_tile(self):
        level = cast(SvsTiledPage, self.tiler.get_level(1))
        tile = level._get_scaled_tile(Point(0, 0))
        self.assertEqual(
            'd9135db3b0bcc0d9e785754e760f80c4',
            md5(tile).hexdigest()
        )
        tile = level._get_scaled_tile(Point(50, 50))
        self.assertEqual(
            'bd0599aa1becf3511fa122582ecc7e3d',
            md5(tile).hexdigest()
        )

    def test_tile_is_at_edge(self):
        print(self.level.tiled_size)
        self.assertFalse(self.level._tile_is_at_right_edge(Point(178, 127)))
        self.assertFalse(self.level._tile_is_at_right_edge(Point(178, 128)))
        self.assertTrue(self.level._tile_is_at_right_edge(Point(179, 127)))
        self.assertTrue(self.level._tile_is_at_right_edge(Point(179, 128)))

        self.assertFalse(self.level._tile_is_at_bottom_edge(Point(178, 127)))
        self.assertFalse(self.level._tile_is_at_bottom_edge(Point(179, 127)))
        self.assertTrue(self.level._tile_is_at_bottom_edge(Point(178, 128)))
        self.assertTrue(self.level._tile_is_at_bottom_edge(Point(179, 128)))

    def test_detect_corrupt_edges(self):
        self.assertEqual(
            (False, False),
            self.level._detect_corrupt_edges()
        )
        self.assertEqual(
            (False, False),
            self.level._detect_corrupt_edges()
        )
