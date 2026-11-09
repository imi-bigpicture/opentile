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

"""Tests of :func:`opentile_jpeg.crop`, :func:`opentile_jpeg.crop_multiple` and the
extension's Cropper behind them.
"""

import gc
import io

import numpy as np
import pytest
from PIL import Image

import opentile_jpeg
from conftest import RED, WHITE, AssertSolid, Decode
from opentile_jpeg import _binding


class TestCrop:
    def test_matches_source_region(self, frame_444: bytes, decode: Decode) -> None:
        # Act
        out = opentile_jpeg.crop(frame_444, 512, 0, 512, 512)

        # Assert
        assert np.array_equal(decode(out), decode(frame_444)[0:512, 512:1024])

    def test_420_interior_is_lossless(self, frame_420: bytes, decode: Decode) -> None:
        # Act
        out = opentile_jpeg.crop(frame_420, 512, 0, 512, 512)

        # Assert
        # 4:2:0 chroma upsampling differs only near the crop's outer edges, so
        # the interior (trimmed by one 16px MCU) is bit-exact.
        source = decode(frame_420)
        assert np.array_equal(decode(out)[16:496, 16:496], source[16:496, 528:1008])

    @pytest.mark.parametrize(
        "region",
        [
            (1024, 0, 16, 16),  # starts past the right edge
            (0, 512, 16, 16),  # starts past the bottom edge
            (1008, 0, 32, 16),  # runs past the right edge
            (-16, 0, 16, 16),  # starts before the left edge
            (0, 0, 0, 16),  # empty
        ],
    )
    def test_rejects_region_outside_image(
        self, frame_420: bytes, region: tuple[int, int, int, int]
    ) -> None:
        # Act & Assert
        with pytest.raises(ValueError, match="exceeds|between"):
            opentile_jpeg.crop(frame_420, *region)

    def test_rejects_unaligned_origin(self, frame_420: bytes) -> None:
        # Act & Assert
        with pytest.raises(ValueError, match="16x16 MCU grid"):
            opentile_jpeg.crop(frame_420, 8, 0, 16, 16)

    def test_rejects_size_beyond_jpeg_limit(self, frame_420: bytes) -> None:
        # Act & Assert
        with pytest.raises(ValueError, match="between 1 and 65500"):
            opentile_jpeg.crop(frame_420, 0, 0, 2**31 - 1, 16)

    def test_background_extends_past_edges(
        self, frame_444: bytes, decode: Decode, assert_solid: AssertSolid
    ) -> None:
        # Act
        out = opentile_jpeg.crop(frame_444, 512, 256, 1024, 512, background=255)

        # Assert
        pixels = decode(out)
        assert pixels.shape == (512, 1024, 3)
        assert np.array_equal(
            pixels[0:256, 0:512], decode(frame_444)[256:512, 512:1024]
        )
        assert_solid(pixels[256:, :], WHITE, tolerance=5)  # below the image
        assert_solid(pixels[:, 512:], WHITE, tolerance=5)  # right of the image

    def test_background_inside_image_matches_plain_crop(self, frame_420: bytes) -> None:
        # Act
        out = opentile_jpeg.crop(frame_420, 512, 0, 512, 512, background=255)

        # Assert
        assert out == opentile_jpeg.crop(frame_420, 512, 0, 512, 512)

    def test_background_fills_region_wholly_outside(
        self, frame_420: bytes, decode: Decode, assert_solid: AssertSolid
    ) -> None:
        # Act
        out = opentile_jpeg.crop(frame_420, 2048, 1024, 64, 64, background=RED)

        # Assert
        assert_solid(decode(out), RED, tolerance=60)

    def test_background_keeps_block_straddling_edge(
        self, decode: Decode, assert_solid: AssertSolid
    ) -> None:
        # Arrange
        # 1004px wide 4:4:4: block 125 (x 1000..1007) straddles the right edge
        # and is copied; blocks from x 1008 on lie wholly outside and are filled.
        rng = np.random.default_rng(3)
        image = Image.fromarray(rng.integers(0, 256, (16, 1004, 3), dtype=np.uint8))
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=90, subsampling="4:4:4")
        jpeg = buffer.getvalue()

        # Act
        out = opentile_jpeg.crop(jpeg, 992, 0, 32, 16, background=255)

        # Assert
        pixels = decode(out)
        assert np.array_equal(pixels[:, 0:12], decode(jpeg)[:, 992:1004])
        assert_solid(pixels[:, 16:], WHITE, tolerance=5)

    def test_background_still_rejects_negative_origin(self, frame_420: bytes) -> None:
        # Act & Assert
        with pytest.raises(ValueError, match="exceeds"):
            opentile_jpeg.crop(frame_420, -16, 0, 16, 16, background=255)

    @pytest.mark.parametrize("background", [256, -1, (0, 300, 0)])
    def test_rejects_out_of_range_background(
        self, frame_420: bytes, background: object
    ) -> None:
        # Act & Assert
        with pytest.raises(ValueError, match="0..255"):
            opentile_jpeg.crop(
                frame_420,
                0,
                0,
                16,
                16,
                background=background,  # pyright: ignore[reportArgumentType]
            )
        with pytest.raises(ValueError, match="0..255"):
            opentile_jpeg.crop_multiple(
                frame_420,
                [(0, 0, 16, 16)],
                background=background,  # pyright: ignore[reportArgumentType]
            )


class TestCropMultiple:
    def test_same_size_regions(self, frame_444: bytes, decode: Decode) -> None:
        # Act
        tiles = list(
            opentile_jpeg.crop_multiple(
                frame_444, [(0, 0, 512, 512), (512, 0, 512, 512)]
            )
        )

        # Assert
        source = decode(frame_444)
        assert np.array_equal(decode(tiles[0]), source[0:512, 0:512])
        assert np.array_equal(decode(tiles[1]), source[0:512, 512:1024])

    def test_different_size_regions(self, frame_444: bytes, decode: Decode) -> None:
        # Act
        # Differing sizes exercise the per-region (non-Cropper) path.
        tiles = list(
            opentile_jpeg.crop_multiple(
                frame_444, [(0, 0, 512, 512), (512, 0, 256, 512)]
            )
        )

        # Assert
        source = decode(frame_444)
        assert np.array_equal(decode(tiles[0]), source[0:512, 0:512])
        assert np.array_equal(decode(tiles[1]), source[0:512, 512:768])

    def test_no_regions_yields_nothing(self, frame_444: bytes) -> None:
        # Act
        tiles = list(opentile_jpeg.crop_multiple(frame_444, []))

        # Assert
        assert tiles == []

    @pytest.mark.parametrize(
        "regions",
        [
            [(0, 0, 64, 64), (1024, 0, 64, 64)],  # same sizes: Cropper path
            [(0, 0, 64, 64), (0, 4096, 32, 32)],  # different sizes: Source path
        ],
    )
    def test_rejects_region_outside_image_when_reached(
        self, frame_420: bytes, regions: list[tuple[int, int, int, int]]
    ) -> None:
        # Arrange
        tiles = opentile_jpeg.crop_multiple(frame_420, regions)
        next(tiles)

        # Act & Assert
        with pytest.raises(ValueError, match="exceeds"):
            next(tiles)

    def test_rejects_oversized_crops_before_iteration(self, frame_420: bytes) -> None:
        # Act & Assert
        with pytest.raises(ValueError, match="exceeds"):
            opentile_jpeg.crop_multiple(frame_420, [(0, 0, 2048, 512)])

    @pytest.mark.parametrize(
        "regions",
        [
            [(0, 0, 512, 512), (768, 256, 512, 512)],  # same sizes: Cropper path
            [(0, 0, 512, 512), (768, 256, 512, 256)],  # different sizes: Source path
        ],
    )
    def test_background_extends_past_edges(
        self,
        frame_444: bytes,
        regions: list[tuple[int, int, int, int]],
        decode: Decode,
        assert_solid: AssertSolid,
    ) -> None:
        # Act
        first, second = opentile_jpeg.crop_multiple(frame_444, regions, background=255)

        # Assert
        source = decode(frame_444)
        assert np.array_equal(decode(first), source[0:512, 0:512])
        pixels = decode(second)
        assert np.array_equal(pixels[0:256, 0:256], source[256:512, 768:1024])
        assert_solid(pixels[:, 256:], WHITE, tolerance=5)


class TestCropper:
    """The extension's Cropper, used directly."""

    def test_keeps_source_alive(self, frame_444: bytes, decode: Decode) -> None:
        # Arrange
        cropper = _binding.Cropper(_binding.Source(frame_444), 512, 512)
        gc.collect()

        # Act
        out = cropper.crop(512, 0)

        # Assert
        assert np.array_equal(decode(out), decode(frame_444)[0:512, 512:1024])

    def test_stays_usable_after_rejected_region(
        self, frame_444: bytes, decode: Decode
    ) -> None:
        # Arrange
        cropper = _binding.Cropper(_binding.Source(frame_444), 512, 512)
        with pytest.raises(ValueError):
            cropper.crop(4096, 0)

        # Act
        out = cropper.crop(512, 0)

        # Assert
        assert np.array_equal(decode(out), decode(frame_444)[0:512, 512:1024])
