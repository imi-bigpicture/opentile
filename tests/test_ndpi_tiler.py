import unittest

import pytest
from ndpi_tiler import NdpiTiler, __version__
from ndpi_tiler.interface import (NdpiFileHandle, NdpiLevel, NdpiStripedLevel,
                                  Point, Size, Tags, NdpiTile)
from tifffile.tifffile import TiffFile

from .create_jpeg_data import open_tif


@pytest.mark.unittest
class NdpiTilerTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tif: TiffFile
        self.tiler: NdpiTiler
        self.level: NdpiLevel

    @classmethod
    def setUpClass(cls):
        cls.tile_size = Size(1024, 1024)
        cls.tif = open_tif()
        cls.tiler = NdpiTiler(
            cls.tif.series[0],
            NdpiFileHandle(cls.tif.filehandle),
            (cls.tile_size.width, cls.tile_size.height),
            'C:/libjpeg-turbo64/bin/turbojpeg.dll'
        )
        cls.level: NdpiStripedLevel = cls.tiler._create_level(0)

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
        header = self.level._page.jpegheader
        index, length = self.level._find_tag(header, Tags.start_of_frame())
        self.assertEqual(621, index)
        self.assertEqual(17, length)

    def test_update_header(self):
        target_size = Size(512, 200)
        header = self.level._page.jpegheader
        updated_header = self.level._update_header(header, target_size)
        (
            stripe_width,
            stripe_height,
            _, _
        ) = self.tiler.jpeg.decode_header(updated_header)
        self.assertEqual(target_size, Size(stripe_width, stripe_height))

    def test_stripe_position_to_index(self):
        self.assertEqual(
            50,
            self.level._stripe_position_to_index(Point(50, 0))
        )
        self.assertEqual(
            800,
            self.level._stripe_position_to_index(Point(20, 20))
        )

    def test_file_handle_read(self):
        offset = self.level._page.dataoffsets[50]
        length = self.level._page.databytecounts[50]
        data = self.level._fh.read(offset, length)
        self.assertEqual(
            84212,
            sum(data)
        )

    def test_level_read(self):
        data = self.level._read(50)
        self.assertEqual(
            84212,
            sum(data)
        )

    def test_get_stripe(self):
        stripe = self.level._get_stripe(Point(50, 0))
        self.assertEqual(
            84212,
            sum(stripe)
        )

    def test_get_frame(self):
        image = self.level._get_frame(Point(10, 10))
        self.assertEqual(
            10629431,
            sum(image)
        )

    def test_map_tile_to_frame(self):
        tile = NdpiTile(Point(5, 5), self.tile_size, self.level.frame_size)

        self.assertEqual(
            Point(1024, 0),
            Point(tile.left, tile.top)
        )

    def test_origin_tile(self):
        tile = NdpiTile(Point(3, 0), self.tile_size, self.level.frame_size)
        self.assertEqual(
            Point(0, 0),
            tile.origin
        )
        tile = NdpiTile(Point(7, 0), self.tile_size, self.level.frame_size)
        self.assertEqual(
            Point(4, 0),
            tile.origin
        )
        tile = NdpiTile(Point(5, 2), self.tile_size, self.level.frame_size)
        self.assertEqual(
            Point(4, 2),
            tile.origin
        )

    # def test_create_tile_jobs(self):
    #     requested_tiles = [
    #         Point(0, 0),
    #         Point(1, 0),
    #         Point(5, 0),
    #         Point(5, 1)
    #     ]
    #     expected_jobs = [
    #         [Point(0, 0), Point(1, 0)],
    #         [Point(5, 0)],
    #         [Point(5, 1)]
    #     ]
    #     jobs = self.level._create_tile_jobs(requested_tiles)
    #     self.assertEqual(expected_jobs, jobs)
