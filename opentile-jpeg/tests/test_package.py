#    Copyright 2026 SECTRA AB
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

"""Tests of the package-level API: errors, warnings and the bundled
libjpeg-turbo version.
"""

import tomllib
import warnings
from pathlib import Path

import pytest

import opentile_jpeg
from conftest import Decode


class TestErrorsAndWarnings:
    @pytest.mark.parametrize(
        "operation", [opentile_jpeg.optimize, opentile_jpeg.header]
    )
    def test_bad_input_raises_libjpeg_turbo_message(self, operation) -> None:
        # Act & Assert
        # Exercises the C++ -> Python exception translation.
        with pytest.raises(opentile_jpeg.JpegError, match="Not a JPEG file"):
            operation(b"not a jpeg")

    def test_recoverable_corruption_warns_at_caller(
        self, frame_420: bytes, decode: Decode
    ) -> None:
        # Arrange
        truncated = frame_420[: len(frame_420) // 2]

        # Act
        with pytest.warns(opentile_jpeg.JpegWarning, match="Premature end") as record:
            out = opentile_jpeg.optimize(truncated)

        # Assert
        assert decode(out).shape == (512, 1024, 3)
        assert record[0].filename == __file__

    def test_recoverable_corruption_can_be_escalated(self, frame_420: bytes) -> None:
        # Arrange
        truncated = frame_420[: len(frame_420) // 2]

        # Act & Assert
        with warnings.catch_warnings():
            warnings.simplefilter("error", opentile_jpeg.JpegWarning)
            with pytest.raises(opentile_jpeg.JpegWarning):
                opentile_jpeg.optimize(truncated)


def test_libjpeg_turbo_version_is_the_pinned_one() -> None:
    # Arrange
    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    pin = tomllib.loads(pyproject.read_text(encoding="utf-8"))["tool"]["opentile-jpeg"]

    # Act
    version = opentile_jpeg.libjpeg_turbo_version

    # Assert
    assert version == pin["libjpeg-turbo"]["version"]
