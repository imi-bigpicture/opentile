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

"""Tests of :func:`opentile_jpeg.fill`."""

import io

import pytest
from PIL import Image

import opentile_jpeg
from conftest import RED, WHITE, AssertSolid, Decode, MakeJpeg


class TestFill:
    def test_gray_level_keeps_template_size(
        self, frame_420: bytes, decode: Decode, assert_solid: AssertSolid
    ) -> None:
        # Act
        out = opentile_jpeg.fill(frame_420, 255)

        # Assert
        pixels = decode(out)
        assert pixels.shape == (512, 1024, 3)
        assert_solid(pixels, WHITE, tolerance=5)

    @pytest.mark.parametrize("level", [0, 255])
    def test_extremes_are_exact_with_coarse_quantization(
        self, level: int, decode: Decode, assert_solid: AssertSolid
    ) -> None:
        # Arrange
        # Quality 5 quantizes the DC in steps of about 160, so the nearest DC for
        # white decodes to about 248; the fill must round it outwards instead.
        buffer = io.BytesIO()
        Image.new("RGB", (64, 64), (90, 120, 150)).save(buffer, "JPEG", quality=5)

        # Act
        out = opentile_jpeg.fill(buffer.getvalue(), level)

        # Assert
        assert_solid(decode(out), level, tolerance=0)

    def test_rgb_color(
        self, frame_420: bytes, decode: Decode, assert_solid: AssertSolid
    ) -> None:
        # Act
        out = opentile_jpeg.fill(frame_420, RED)

        # Assert
        assert_solid(decode(out), RED, tolerance=60)

    def test_grayscale_template_uses_luma(
        self, make_jpeg: MakeJpeg, decode: Decode, assert_solid: AssertSolid
    ) -> None:
        # Arrange
        gray = make_jpeg(64, 64, subsample="gray")

        # Act
        out = opentile_jpeg.fill(gray, 200)

        # Assert
        assert_solid(decode(out), 200, tolerance=4)

    def test_keeps_template_header(self, frame_420: bytes) -> None:
        # Act
        out = opentile_jpeg.fill(frame_420, 128)

        # Assert
        assert opentile_jpeg.header(out) == opentile_jpeg.header(frame_420)

    def test_accepts_any_rgb_sequence(self, frame_420: bytes) -> None:
        # Arrange
        as_list = [255, 0, 0]  # typed as a tuple, but any sequence works

        # Act
        out = opentile_jpeg.fill(frame_420, as_list)  # pyright: ignore[reportArgumentType]

        # Assert
        assert out == opentile_jpeg.fill(frame_420, RED)

    @pytest.mark.parametrize("color", [256, -1, (0, 300, 0)])
    def test_rejects_out_of_range_color(self, frame_420: bytes, color: object) -> None:
        # Act & Assert
        with pytest.raises(ValueError, match="0..255"):
            opentile_jpeg.fill(frame_420, color)  # pyright: ignore[reportArgumentType]

    @pytest.mark.parametrize("color", [(255, 0), (1, 2, 3, 4)])
    def test_wrong_length_error_names_the_parameter(
        self, frame_420: bytes, color: tuple[int, ...]
    ) -> None:
        # Act & Assert
        with pytest.raises(ValueError, match="color must be a gray level"):
            opentile_jpeg.fill(frame_420, color)  # pyright: ignore[reportArgumentType]
