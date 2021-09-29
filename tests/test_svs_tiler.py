import os
import unittest
from hashlib import md5

import pytest
from opentile.geometry import Point, Size
from opentile.svs_tiler import SvsTiler, SvsTiledPage
from tifffile import TiffFile

svs_test_data_dir = os.environ.get(
    "OPEN_TILER_TESTDIR",
    "C:/temp/opentile/svs/"
)
sub_data_path = "svs1/input.svs"
svs_file_path = svs_test_data_dir + '/' + sub_data_path


@pytest.mark.unittest
class SvsTilerTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tif: TiffFile
        self.tiler: SvsTiler
        self.level: SvsTiledPage

    @classmethod
    def setUpClass(cls):
        cls.tile_size = Size(1024, 1024)
        cls.tiler = SvsTiler(TiffFile(svs_file_path))
        cls.level: SvsTiledPage = cls.tiler.get_level(0)

    @classmethod
    def tearDownClass(cls):
        cls.tiler.close()

    def test_get_tile(self):
        tile = self.level.get_tile((0, 0))
        self.assertEqual(
            'bfc67c0c88684c96f605324649949c31',
            md5(tile).hexdigest()
        )
        tile = self.level.get_tile((20, 20))
        self.assertEqual(
            '7997893f529fc4f940751ef4bf2b6407',
            md5(tile).hexdigest()
        )

    def test_get_scaled_tile(self):
        level: SvsTiledPage = self.tiler.get_level(1)
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
            self.tiler.get_level(0)._detect_corrupt_edges()
        )
        self.assertEqual(
            (False, False),
            self.tiler.get_level(1)._detect_corrupt_edges()
        )