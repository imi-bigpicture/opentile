import os
import unittest
from hashlib import md5

import pytest
from wsidicom.interface import TiledLevel
from open_tiler import SvsTiler, __version__
from tifffile import TiffFile
from tifffile.tifffile import TiffFile
from wsidicom.geometry import Point, Size

svs_test_data_dir = os.environ.get(
    "OPEN_TILER_TESTDIR",
    "C:/temp/open_tiler/svs/"
)
sub_data_path = "input.svs"
svs_file_path = svs_test_data_dir + '/' + sub_data_path


@pytest.mark.unittest
class NdpiTilerTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tif: TiffFile
        self.tiler: SvsTiler
        self.level: TiledLevel

    @classmethod
    def setUpClass(cls):
        cls.tile_size = Size(1024, 1024)
        cls.tiler = SvsTiler(svs_file_path)
        cls.level: TiledLevel = cls.tiler.get_level(0)

    @classmethod
    def tearDownClass(cls):
        cls.tiler.close()

    def test_get_tile(self):
        tile = self.level.get_tile(Point(0, 0))
        self.assertEqual(
            'd233adca5123262394a45a2cc7d5f6cf',
            md5(tile).hexdigest()
        )
        tile = self.level.get_tile(Point(20, 20))
        self.assertEqual(
            'baab24a3fd1ef3e5b74bac00790c8480',
            md5(tile).hexdigest()
        )
