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

import pytest
from opentile.philips_tiff_tiler import PhilipsTiffTiler

philips_test_data_dir = os.environ.get(
    "OPEN_TILER_TESTDIR",
    "C:/temp/opentile/philips_tiff/"
)
sub_data_path = "philips1/input.tif"
philips_file_path = Path(philips_test_data_dir + '/' + sub_data_path)
turbojpeg_path = Path('C:/libjpeg-turbo64/bin/turbojpeg.dll')


@pytest.mark.unittest
class PhilipsTiffTilerTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tiler: PhilipsTiffTiler

    @classmethod
    def setUpClass(cls):
        cls.tiler = PhilipsTiffTiler(
            philips_file_path,
            turbojpeg_path
        )
        cls.level = cls.tiler.get_level(0)

    @classmethod
    def tearDownClass(cls):
        cls.tiler.close()

    def test_get_tile(self):
        tile = self.level.get_tile((0, 0))
        self.assertEqual(
            '570d069f9de5d2716fb0d7167bc79195',
            md5(tile).hexdigest()
        )
        tile = self.level.get_tile((20, 20))
        self.assertEqual(
            'db28efb73a72ef7e2780fc72c624d7ae',
            md5(tile).hexdigest()
        )
