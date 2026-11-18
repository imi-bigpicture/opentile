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

import numpy as np
import pytest
import tifffile

from opentile import OpenTile
from opentile.exceptions import UnsupportedFileError

from tests.filepaths import (
    histech_file_path,
    ndpi_file_path,
    ome_tiff_file_path,
    philips_file_path,
    svs_file_path,
    trestle_file_path,
    ventana_file_path,
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
            trestle_file_path,
            ventana_file_path,
        ],
    )
    def test_open(self, file_path: Path):
        # Arrange
        if not file_path.exists():
            pytest.skip(f"{file_path} test file not found, skipping")

        # Act & Assert
        with OpenTile.open(file_path):
            pass

    def test_open_of_unsupported_tiff_raises_unsupported_file_error(
        self, tmp_path: Path
    ):
        # Arrange - a plain tiff that no tiler supports
        file_path = tmp_path / "plain.tiff"
        tifffile.imwrite(file_path, np.zeros((16, 16, 3), dtype=np.uint8))

        # Act & Assert
        with pytest.raises(UnsupportedFileError):
            OpenTile.open(file_path)
