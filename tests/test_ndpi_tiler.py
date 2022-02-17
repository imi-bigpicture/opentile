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
from opentile.geometry import Point, Size
from opentile.ndpi_tiler import (NdpiCache, NdpiFrameJob, NdpiStripedPage,
                                 NdpiTile, NdpiTiler)

test_data_dir = os.environ.get(
    "OPENTILE_TESTDIR",
    "C:/temp/opentile/"
)

ndpi_file_path = Path(test_data_dir).joinpath("ndpi/CMU-1/CMU-1.ndpi")
turbojpeg_path = Path('C:/libjpeg-turbo64/bin/turbojpeg.dll')


@pytest.mark.unittest
class NdpiTilerTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tiler: NdpiTiler

    @classmethod
    def setUpClass(cls):
        cls.tile_size = Size(512, 512)
        try:
            cls.tiler = NdpiTiler(
                ndpi_file_path,
                cls.tile_size.width,
                turbojpeg_path
            )
        except FileNotFoundError:
            raise unittest.SkipTest('ndpi test file not found, skipping')
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
            520,
            self.level._get_stripe_position_to_index(Point(20, 20))
        )

    def test_file_handle_read(self):
        offset = self.level._page.dataoffsets[50]
        length = self.level._page.databytecounts[50]
        data = self.level._fh.read(offset, length)
        self.assertEqual(
            '2a903c6e05bd10f10d856eecceb591f0',
            md5(data).hexdigest()
        )

    def test_level_read(self):
        data = self.level._read_frame(50)
        self.assertEqual(
            '2a903c6e05bd10f10d856eecceb591f0',
            md5(data).hexdigest()
        )

    def test_read_frame(self):
        index = self.level._get_stripe_position_to_index(Point(50, 0))

        stripe = self.level._read_frame(index)
        self.assertEqual(
            '2a903c6e05bd10f10d856eecceb591f0',
            md5(stripe).hexdigest()
        )

    def test_get_frame(self):
        image = self.level._read_extended_frame(
            Point(10, 10),
            self.level.frame_size
        )
        self.assertEqual(
            'aeffd12997ca6c232d0ef35aaa35f6b7',
            md5(image).hexdigest()
        )

    def test_get_tile(self):
        tile = self.level.get_tile((0, 0))
        self.assertEqual(
            '30c69cab610e5b3db4beac63806d6513',
            md5(tile).hexdigest()
        )
        tile = self.level.get_tile((20, 20))
        self.assertEqual(
            'fec8116d05485df513f4f41e13eaa994',
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
            (512, 0, 512, 512),
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
        self.assertEqual(Size(2048, 8), self.level.stripe_size)

    def test_striped_size(self):
        self.assertEqual(Size(25, 4768), self.level.striped_size)

    def test_header(self):
        header = self.level._page.jpegheader
        self.assertEqual(
            '579211c6b9fedca17d94b95840f4b985',
            md5(header).hexdigest()
        )

    def test_get_file_frame_size(self):
        self.assertEqual(Size(2048, 8), self.level._get_file_frame_size())

    def test_get_frame_size(self):
        self.assertEqual(
            Size(2048, 512),
            self.level.frame_size
        )

    def test_get_frame_size_for_tile(self):
        self.assertEqual(
            Size(2048, 512),
            self.level._get_frame_size_for_tile(Point(0, 0))
        )
        print(self.level.tiled_size)
        self.assertEqual(
            Size(2048, 256),
            self.level._get_frame_size_for_tile(Point(99, 74))
        )

    def test_tiled_size(self):
        self.assertEqual(Size(100, 75), self.level.tiled_size)

    def test_get_smallest_stripe_width(self):
        self.assertEqual(128, self.tiler._get_smallest_stripe_width())

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
