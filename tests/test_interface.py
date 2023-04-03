#    Copyright 2022-2023 SECTRA AB
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

from pathlib import Path
import unittest

import pytest
from opentile import OpenTile

from .filepaths import (
    histech_file_path,
    ndpi_file_path,
    philips_file_path,
    svs_file_path,
    ome_tiff_file_path,
)


@pytest.mark.unittest
class InterfaceTest(unittest.TestCase):
    def test_open_svs(self):
        self._test_open(svs_file_path)

    def test_open_ndpi(self):
        self._test_open(ndpi_file_path)

    def test_open_philips(self):
        self._test_open(philips_file_path)

    def test_open_histech(self):
        self._test_open(histech_file_path)

    def test_open_ome_tiff(self):
        self._test_open(ome_tiff_file_path)

    def _test_open(self, file: Path):
        self._check_file_exists(file)
        OpenTile.open(file)

    def _check_file_exists(self, file: Path):
        if not file.exists():
            raise unittest.SkipTest(f"{file} test file not found, skipping")
