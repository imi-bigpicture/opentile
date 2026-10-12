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

"""Tests of the package-level API: errors, warnings, the pixel limit and the
bundled libjpeg-turbo version.
"""

import inspect
import tomllib
import warnings
from collections.abc import Callable
from pathlib import Path

import pytest

import opentile_jpeg
from conftest import Decode, MakeJpeg


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


def claiming_size(jpeg: bytes, width: int, height: int) -> bytes:
    """``jpeg`` with its frame header claiming another size; the data is as is."""
    size = jpeg.index(b"\xff\xc0") + 5  # after the marker, length and precision
    claim = height.to_bytes(2, "big") + width.to_bytes(2, "big")
    return jpeg[:size] + claim + jpeg[size + 4 :]


Operation = Callable[..., object]

OPERATIONS_READING_IMAGE: dict[str, Operation] = {
    "crop": lambda jpeg, **limit: opentile_jpeg.crop(jpeg, 0, 0, 16, 16, **limit),
    "crop_multiple": lambda jpeg, **limit: list(
        opentile_jpeg.crop_multiple(jpeg, [(0, 0, 16, 16)], **limit)
    ),
    "compose": lambda jpeg, **limit: list(
        opentile_jpeg.compose([(jpeg, 0, 0)], (16, 16), (16, 16), [(0, 0)], **limit)
    ),
    "optimize": lambda jpeg, **limit: opentile_jpeg.optimize(jpeg, **limit),
    "fill": lambda jpeg, **limit: opentile_jpeg.fill(jpeg, 255, **limit),
    "decode_coefficients": lambda jpeg, **limit: opentile_jpeg.decode_coefficients(
        jpeg, **limit
    ),
}


@pytest.fixture(params=OPERATIONS_READING_IMAGE)
def operation(request: pytest.FixtureRequest) -> Operation:
    return OPERATIONS_READING_IMAGE[request.param]


class TestPixelLimit:
    def test_refuses_image_over_limit(
        self, operation: Operation, make_jpeg: MakeJpeg
    ) -> None:
        # Act & Assert
        with pytest.raises(opentile_jpeg.JpegTooLargeError, match="64x64"):
            operation(make_jpeg(64, 64), max_pixels=64 * 64 - 1)

    @pytest.mark.parametrize("max_pixels", [64 * 64, None])
    def test_reads_image_within_limit(
        self, operation: Operation, make_jpeg: MakeJpeg, max_pixels: int | None
    ) -> None:
        # Act & Assert
        operation(make_jpeg(64, 64), max_pixels=max_pixels)

    def test_default_refuses_huge_claim_before_allocating(
        self, operation: Operation, make_jpeg: MakeJpeg
    ) -> None:
        # Arrange
        # Reading this would allocate about 12 GB of coefficients.
        jpeg = claiming_size(make_jpeg(64, 64), 65_000, 65_000)

        # Act & Assert
        with pytest.raises(opentile_jpeg.JpegTooLargeError) as error:
            operation(jpeg)
        assert isinstance(error.value, opentile_jpeg.JpegError)

    @pytest.mark.parametrize("max_pixels", [0, -1])
    def test_rejects_non_positive_limit(
        self, operation: Operation, make_jpeg: MakeJpeg, max_pixels: int
    ) -> None:
        # Act & Assert
        with pytest.raises(ValueError, match="max_pixels"):
            operation(make_jpeg(64, 64), max_pixels=max_pixels)


@pytest.mark.parametrize("name", OPERATIONS_READING_IMAGE)
def test_max_pixels_is_keyword_only(name: str) -> None:
    # Act
    parameter = inspect.signature(getattr(opentile_jpeg, name)).parameters["max_pixels"]

    # Assert
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default == opentile_jpeg.DEFAULT_MAX_PIXELS


def test_libjpeg_turbo_version_is_the_pinned_one() -> None:
    # Arrange
    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    pin = tomllib.loads(pyproject.read_text(encoding="utf-8"))["tool"]["opentile-jpeg"]

    # Act
    version = opentile_jpeg.libjpeg_turbo_version

    # Assert
    assert version == pin["libjpeg-turbo"]["version"]
