import os
import unittest
from hashlib import md5

import pytest
from opentile import PhillipsTiffTiler, __version__
from opentile.geometry import Point
from tifffile import TiffFile
from tifffile.tifffile import TiffFile
from opentile.phillips_tiff_tiler import PhillipsTiffTiledPage

phillips_test_data_dir = os.environ.get(
    "OPEN_TILER_TESTDIR",
    "C:/temp/opentile/phillips_tiff/"
)
sub_data_path = "input.tif"
phillips_file_path = phillips_test_data_dir + '/' + sub_data_path


@pytest.mark.unittest
class PhillipsTiffTilerTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tif: TiffFile
        self.tiler: PhillipsTiffTiler
        self.level: PhillipsTiffTiledPage

    @classmethod
    def setUpClass(cls):
        cls.tiler = PhillipsTiffTiler(
            phillips_file_path,
            'C:/libjpeg-turbo64/bin/turbojpeg.dll'
        )
        cls.level: PhillipsTiffTiledPage = cls.tiler.get_level(0)

    @classmethod
    def tearDownClass(cls):
        cls.tiler.close()

    def test_get_tile(self):
        tile = self.level.get_tile(Point(0, 0))
        self.assertEqual(
            '570d069f9de5d2716fb0d7167bc79195',
            md5(tile).hexdigest()
        )
        tile = self.level.get_tile(Point(20, 20))
        self.assertEqual(
            'db28efb73a72ef7e2780fc72c624d7ae',
            md5(tile).hexdigest()
        )
