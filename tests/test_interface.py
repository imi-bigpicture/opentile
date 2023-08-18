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
class TestInterface:
    @pytest.mark.parametrize(
        "file_path",
        [
            svs_file_path,
            ndpi_file_path,
            philips_file_path,
            histech_file_path,
            ome_tiff_file_path,
        ],
    )
    def test_open(self, file_path: Path):
        # Arrange
        if not file_path.exists():
            pytest.skip(f"{file_path} test file not found, skipping")

        # Act & Assert
        with OpenTile.open(file_path):
            pass
