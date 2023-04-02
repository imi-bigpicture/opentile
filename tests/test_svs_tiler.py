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

import unittest
from datetime import datetime
from hashlib import md5
from typing import Sequence, Tuple, cast

import pytest
from parameterized import parameterized
from tifffile.tifffile import PHOTOMETRIC

from opentile.formats import SvsTiler
from opentile.formats.svs.svs_page import SvsTiledPage
from opentile.geometry import Point

from .filepaths import svs_file_path


@pytest.mark.unittest
class SvsTilerTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tiler: SvsTiler

    @classmethod
    def setUpClass(cls):
        try:
            cls.tiler = SvsTiler(svs_file_path)
        except FileNotFoundError:
            raise unittest.SkipTest("Svs test file not found, skipping")
        cls.level = cast(SvsTiledPage, cls.tiler.get_level(0))

    @classmethod
    def tearDownClass(cls):
        cls.tiler.close()

    @parameterized.expand(
        [
            ((15, 25), "dd8402d5a7250ebfe9e33b3dfee3c1e1"),
            ((35, 30), "50f885cd7299bd3fff22743bfb4a4930"),
        ]
    )
    def test_get_tile(self, tile_point: Tuple[int, int], hash: str):
        tile = self.level.get_tile(tile_point)
        self.assertEqual(hash, md5(tile).hexdigest())

    @parameterized.expand(
        [
            (
                [(15, 25), (35, 30)],
                [
                    "dd8402d5a7250ebfe9e33b3dfee3c1e1",
                    "50f885cd7299bd3fff22743bfb4a4930",
                ],
            ),
        ]
    )
    def test_get_tiles(
        self, tile_points: Sequence[Tuple[int, int]], hashes: Sequence[str]
    ):
        tiles = self.level.get_tiles(tile_points)
        for tile, hash in zip(tiles, hashes):
            self.assertEqual(hash, md5(tile).hexdigest())

    @parameterized.expand(
        [
            ((0, 0), "d9135db3b0bcc0d9e785754e760f80c4"),
            ((50, 50), "bd0599aa1becf3511fa122582ecc7e3d"),
        ]
    )
    def test_get_scaled_tile(self, tile_point: Tuple[int, int], hash: str):
        level = cast(SvsTiledPage, self.tiler.get_level(1))
        tile = level._get_scaled_tile(Point.from_tuple(tile_point))
        self.assertEqual(hash, md5(tile).hexdigest())

    @parameterized.expand(
        [
            (Point(178, 127), False, False),
            (Point(178, 128), False, True),
            (Point(179, 127), True, False),
            (Point(179, 128), True, True),
        ]
    )
    def test_tile_is_at_edge(self, tile: Point, right: bool, bottom: bool):
        self.assertEqual(self.level._tile_is_at_right_edge(tile), right)
        self.assertEqual(self.level._tile_is_at_bottom_edge(tile), bottom)

    def test_detect_corrupt_edges(self):
        self.assertEqual((False, False), self.level._detect_corrupt_edges())

    def test_photometric_interpretation(self):
        self.assertEqual(
            PHOTOMETRIC.RGB, self.tiler.get_level(0).photometric_interpretation
        )

    def test_subsampling(self):
        self.assertEqual((2, 2), self.tiler.get_level(0).subsampling)

    def test_sumples_per_pixel(self):
        self.assertEqual(3, self.tiler.get_level(0).samples_per_pixel)

    def test_metadata_magnification(self):
        self.assertEqual(20.0, self.tiler.metadata.magnification)

    def test_metadata_aquisition_datetime(self):
        self.assertEqual(
            datetime(2009, 12, 29, 9, 59, 15), self.tiler.metadata.aquisition_datetime
        )
