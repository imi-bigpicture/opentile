import os
import unittest
from hashlib import md5
from pathlib import Path
from typing import cast

import pytest
from opentile.geometry import Point, Size
from opentile.ndpi_tiler import (NdpiCache, NdpiFrameJob, NdpiStripedPage,
                                 NdpiTile, NdpiTiler)

ndpi_test_data_dir = os.environ.get(
    "NDPI_TESTDIR",
    "C:/temp/opentile/ndpi/"
)
sub_data_path = "ndpi2/input.ndpi"
ndpi_file_path = Path(ndpi_test_data_dir + '/' + sub_data_path)
turbojpeg_path = Path('C:/libjpeg-turbo64/bin/turbojpeg.dll')


@pytest.mark.unittest
class NdpiTilerTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tiler: NdpiTiler

    @classmethod
    def setUpClass(cls):
        cls.tile_size = Size(1024, 1024)
        cls.tiler = NdpiTiler(
            ndpi_file_path,
            cls.tile_size.width,
            turbojpeg_path
        )
        cls.level = cast(NdpiStripedPage, cls.tiler.get_level(0))

    @classmethod
    def tearDownClass(cls):
        cls.tiler.close()

    def test_get_stripe_position_to_index(self):
        self.assertEqual(
            50,
            self.level._get_stripe_position_to_index(Point(50, 0))
        )
        self.assertEqual(
            800,
            self.level._get_stripe_position_to_index(Point(20, 20))
        )

    def test_file_handle_read(self):
        offset = self.level._page.dataoffsets[50]
        length = self.level._page.databytecounts[50]
        data = self.level._fh.read(offset, length)
        self.assertEqual(
            'e2a7321a7d7032437f91df442b0182da',
            md5(data).hexdigest()
        )

    def test_level_read(self):
        data = self.level._read_frame(50)
        self.assertEqual(
            'e2a7321a7d7032437f91df442b0182da',
            md5(data).hexdigest()
        )

    def test_read_frame(self):
        index = self.level._get_stripe_position_to_index(Point(50, 0))

        stripe = self.level._read_frame(index)
        self.assertEqual(
            'e2a7321a7d7032437f91df442b0182da',
            md5(stripe).hexdigest()
        )

    def test_get_frame(self):
        image = self.level._read_extended_frame(
            Point(10, 10),
            self.level.frame_size
        )
        self.assertEqual(
            '25a908ef4b5340354e6d0d7771e18fcd',
            md5(image).hexdigest()
        )

    def test_get_tile(self):
        tile = self.level.get_tile((0, 0))
        self.assertEqual(
            '4d7d1eb65b8e86b32691aa4d9ab000e4',
            md5(tile).hexdigest()
        )
        tile = self.level.get_tile((20, 20))
        self.assertEqual(
            'eef2ff23353e54464a870d4fdcda6701',
            md5(tile).hexdigest()
        )

    def test_create_tiles(self):
        frame_job = NdpiFrameJob(
            [
                NdpiTile(
                    Point(x, 0),
                    self.tile_size,
                    self.level.frame_size
                )
                for x in range(4)
            ]
        )
        tiles_single = [self.level.get_tile((x, 0)) for x in range(4)]
        self.assertEqual(
            tiles_single,
            list(self.level._create_tiles(frame_job).values())
        )

    def test_crop_to_tiles(self):
        frame_job = NdpiFrameJob(
            [
                NdpiTile(
                    Point(x, 0),
                    self.tile_size,
                    self.level.frame_size
                )
                for x in range(4)
            ]
        )
        tiles_single = {
            Point(x, 0): self.level.get_tile((x, 0))
            for x in range(4)
        }
        frame_size = self.level._get_frame_size_for_tile(frame_job.position)
        frame = self.level._read_extended_frame(frame_job.position, frame_size)

        self.assertEqual(
            tiles_single,
            self.level._crop_to_tiles(frame_job, frame)
        )

    def test_map_tile_to_frame(self):
        tile = NdpiTile(Point(5, 5), self.tile_size, self.level.frame_size)

        self.assertEqual(
            (1024, 0, 1024, 1024),
            (tile.left, tile.top, tile.width, tile.height)
        )

    def test_frame_position_tile(self):
        tile = NdpiTile(Point(3, 0), self.tile_size, self.level.frame_size)
        self.assertEqual(
            Point(0, 0),
            tile.frame_position
        )
        tile = NdpiTile(Point(7, 0), self.tile_size, self.level.frame_size)
        self.assertEqual(
            Point(4, 0),
            tile.frame_position
        )
        tile = NdpiTile(Point(5, 2), self.tile_size, self.level.frame_size)
        self.assertEqual(
            Point(4, 2),
            tile.frame_position
        )

    def test_frame_jobs(self):
        tile0 = NdpiTile(Point(3, 0), self.tile_size, self.level.frame_size)
        frame_job = NdpiFrameJob([tile0])
        self.assertEqual(Point(0, 0), frame_job.position)

        tile1 = NdpiTile(Point(2, 0), self.tile_size, self.level.frame_size)
        frame_job.append(tile1)
        self.assertEqual([tile0, tile1], frame_job.tiles)

        tile2 = NdpiTile(Point(2, 1), self.tile_size, self.level.frame_size)
        with self.assertRaises(ValueError):
            frame_job.append(tile2)

    def test_sort_into_frame_jobs(self):
        self.assertEqual(
            [
                NdpiFrameJob(
                    [
                        NdpiTile(
                            Point(index_x_0+index_x_1, index_y),
                            self.tile_size,
                            self.level.frame_size
                        )
                        for index_x_1 in range(4)
                    ]

                )
                for index_x_0 in range(0, 8, 4)
                for index_y in range(2)
            ],
            self.level._sort_into_frame_jobs(
                [
                    (index_x, index_y)
                    for index_x in range(8)
                    for index_y in range(2)
                ]
            )
        )

    def test_cache(self):
        cache_size = 10
        cache = NdpiCache(cache_size)
        for index in range(10):
            point = Point(index, index)
            data = bytes([index])
            cache[point] = data
            self.assertEqual(data, cache[point])
        self.assertEqual(cache_size, len(cache))

        next = 10
        point = Point(next, next)
        data = bytes([next])
        cache[point] = data
        self.assertEqual(data, cache[point])
        self.assertEqual(cache_size, len(cache))
        with self.assertRaises(KeyError):
            cache[Point(0, 0)]

        update = {
            Point(index, index): bytes([index])
            for index in range(11, 20)
        }
        cache.update(update)
        self.assertEqual(cache_size, len(cache))
        for index in range(10, 20):
            point = Point(index, index)
            data = bytes([index])
            self.assertEqual(data, cache[point])

        for index in range(10):
            with self.assertRaises(KeyError):
                cache[Point(index, index)]

        self.assertEqual(
            [Point(index, index) for index in range(10, 20)],
            list(cache._content.keys())
        )

    def test_stripe_size(self):
        self.assertEqual(Size(4096, 8), self.level.stripe_size)

    def test_striped_size(self):
        self.assertEqual(Size(39, 12992), self.level.striped_size)

    def test_header(self):
        header = self.level._page.jpegheader
        self.assertEqual(
            '624428850c21156087d870b5a95ea8ac',
            md5(header).hexdigest()
        )

    def test_get_file_frame_size(self):
        self.assertEqual(Size(4096, 8), self.level._get_file_frame_size())

    def test_get_frame_size(self):
        self.assertEqual(
            Size(4096, 1024),
            self.level.frame_size
        )

    def test_get_frame_size_for_tile(self):
        self.assertEqual(
            Size(4096, 1024),
            self.level._get_frame_size_for_tile(Point(0, 0))
        )
        self.assertEqual(
            Size(4096, 512),
            self.level._get_frame_size_for_tile(Point(155, 101))
        )

    def test_tiled_size(self):
        self.assertEqual(Size(156, 102), self.level.tiled_size)

    def test_get_smallest_stripe_width(self):
        self.assertEqual(64, self.tiler._get_smallest_stripe_width())

    def test_adjust_tile_size(self):
        self.assertEqual(
            Size(512, 512),
            self.tiler._adjust_tile_size(
                512,
                self.tiler._get_smallest_stripe_width()
            )
        )

        self.assertEqual(
            Size(448, 448),
            self.tiler._adjust_tile_size(
                512,
                56
            )
        )

        self.assertEqual(
            Size(496, 496),
            self.tiler._adjust_tile_size(
                512,
                248
            )
        )
