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
    "OPEN_TILER_TESTDIR",
    "C:/temp/opentile/svs/"
)
sub_data_path = "svs1/input.svs"
svs_file_path = Path(svs_test_data_dir + '/' + sub_data_path)
turbojpeg_path = Path('C:/libjpeg-turbo64/bin/turbojpeg.dll')


@pytest.mark.unittest
class SvsTilerTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tiler: SvsTiler

    @classmethod
    def setUpClass(cls):
        cls.tiler = SvsTiler(svs_file_path, turbojpeg_path)
        cls.level = cast(SvsTiledPage, cls.tiler.get_level(0))

    @classmethod
    def tearDownClass(cls):
        cls.tiler.close()

    def test_get_tile(self):
        tile = self.level.get_tile((15, 25))
        self.assertEqual(
            '042c56a5a8f989278750b0b89a9e3586',
            md5(tile).hexdigest()
        )
        tile = self.level.get_tile((35, 30))
        self.assertEqual(
            '0b716fecf5cb80a72454ebc8bb15a72c',
            md5(tile).hexdigest()
        )

    def test_get_scaled_tile(self):
        level = cast(SvsTiledPage, self.tiler.get_level(1))
        tile = level._get_scaled_tile(Point(0, 0))
        self.assertEqual(
            '87c887f735772a934f84674fa63a4a10',
            md5(tile).hexdigest()
        )
        tile = level._get_scaled_tile(Point(50, 50))
        self.assertEqual(
            '77e81998d3cb9d1e105cc27c396abd2a',
            md5(tile).hexdigest()
        )

    def test_tile_is_at_edge(self):
        self.assertFalse(self.level._tile_is_at_right_edge(Point(198, 145)))
        self.assertFalse(self.level._tile_is_at_right_edge(Point(198, 146)))
        self.assertTrue(self.level._tile_is_at_right_edge(Point(199, 145)))
        self.assertTrue(self.level._tile_is_at_right_edge(Point(199, 146)))

        self.assertFalse(self.level._tile_is_at_bottom_edge(Point(198, 145)))
        self.assertFalse(self.level._tile_is_at_bottom_edge(Point(199, 145)))
        self.assertTrue(self.level._tile_is_at_bottom_edge(Point(198, 146)))
        self.assertTrue(self.level._tile_is_at_bottom_edge(Point(199, 146)))

    def test_detect_corrupt_edges(self):
        self.assertEqual(
            (False, False),
            self.level._detect_corrupt_edges()
        )
        self.assertEqual(
            (False, False),
            self.level._detect_corrupt_edges()
        )
