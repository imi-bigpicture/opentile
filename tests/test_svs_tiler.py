import os
import unittest
from hashlib import md5

import pytest
from opentile.geometry import Size
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
