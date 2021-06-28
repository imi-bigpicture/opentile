import unittest

import pytest
import tifffile
from tifffile.tifffile import TiffFile

from ndpi_tiler import __version__, NdpiPageTiler
from .create_jpeg_data import open_tif


@pytest.mark.unittest
@pytest.mark.ndpitiler
class NdpiTilerTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tif: TiffFile
        self.tiler: NdpiPageTiler

    @classmethod
    def setUpClass(cls):
        cls.tif = open_tif()
        tiff_series = cls.tif.series[0]
        tiff_level = tiff_series.levels[0]
        page = tiff_level.pages[0]
        cls.tiler = NdpiPageTiler(cls.tif.filehandle, page, 1024, 1024)

    @classmethod
    def tearDownClass(cls):
        cls.tif.close()

    def test_stripe_range(self):
        self.assertEqual(
            range(0, 2),
            self.tiler.stripe_range(0, 1024, 2048)
        )
        self.assertEqual(
            range(2, 4),
            self.tiler.stripe_range(1, 1024, 2048)
        )

        self.assertEqual(
            range(0, 1),
            self.tiler.stripe_range(1, 1024, 512)
        )
