import unittest

import pytest
from tifffile.tifffile import TiffFile

from ndpi_tiler import __version__, NdpiPageTiler
from ndpi_tiler.interface import Tags, Size, Point
from .create_jpeg_data import open_tif


@pytest.mark.unittest
class NdpiTilerTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tif: TiffFile
        self.tiler: NdpiPageTiler

    @classmethod
    def setUpClass(cls):
        cls.tile_size = 512
        cls.tif = open_tif()
        tiff_series = cls.tif.series[0]
        tiff_level = tiff_series.levels[0]
        page = tiff_level.pages[0]
        cls.tiler = NdpiPageTiler(
            cls.tif.filehandle,
            page,
            (cls.tile_size, cls.tile_size),
            'C:/libjpeg-turbo64/bin/turbojpeg.dll'
        )

    @classmethod
    def tearDownClass(cls):
        cls.tif.close()

    def test_tags(self):
        self.assertEqual(Tags.start_of_frame(), bytes([0xFF, 0xC0]))
        self.assertEqual(Tags.end_of_image(), bytes([0xFF, 0xD9]))
        self.assertEqual(Tags.restart_mark(0), bytes([0xD0]))
        self.assertEqual(Tags.restart_mark(7), bytes([0xD7]))
        self.assertEqual(Tags.restart_mark(9), bytes([0xD1]))

    def test_find_tag(self):
        header = self.tiler._page.jpegheader
        index, length = self.tiler._find_tag(header, Tags.start_of_frame())
        self.assertEqual(621, index)
        self.assertEqual(17, length)

    def test__update_header(self):
        target_size = Size(512, 200)
        header = self.tiler._page.jpegheader
        updated_header = self.tiler._update_header(header, target_size)
        (
            stripe_width,
            stripe_height,
            _, _
        ) = self.tiler.jpeg.decode_header(updated_header)
        self.assertEqual(target_size, Size(stripe_width, stripe_height))

    def test_stripe_coordinate_to_index(self):
        self.assertEqual(
            50,
            self.tiler._stripe_coordinate_to_index(Point(50, 0))
        )
        self.assertEqual(
            800,
            self.tiler._stripe_coordinate_to_index(Point(20, 20))
        )

    def test_get_stripe(self):
        stripe = self.tiler._get_stripe(Point(50, 0))
        self.assertEqual(
            84212,
            sum(stripe)
        )

    def test_get_stitched_image(self):
        image = self.tiler._get_stitched_image(Point(10, 10))
        self.assertEqual(
            5153423,
            sum(image)
        )

    def test_map_tile_to_image(self):
        self.assertEqual(
            Point(5*self.tile_size, 5*self.tile_size),
            self.tiler._map_tile_to_image(Point(5, 5))
        )
