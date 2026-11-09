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

"""Tests of :func:`opentile_jpeg.compose` and the extension's Composer behind it."""

import gc

import numpy as np
import pytest

import opentile_jpeg
from conftest import NEUTRAL_GRAY, RED, AssertSolid, Decode, MakeJpeg
from opentile_jpeg import _binding


class TestCompose:
    def test_stitches_sources_into_one_tile(
        self, side_by_side: tuple[bytes, bytes], decode: Decode
    ) -> None:
        # Arrange
        left, right = side_by_side

        # Act
        (out,) = opentile_jpeg.compose(
            [(left, 0, 0), (right, 1, 0)],
            content_size=(512, 512),
            tile_size=(1024, 512),
            positions=[(0, 0)],
        )

        # Assert
        stitched = decode(out)
        assert np.array_equal(stitched[:, 0:512], decode(left))
        assert np.array_equal(stitched[:, 512:1024], decode(right))

    def test_produces_one_tile_per_position(
        self, side_by_side: tuple[bytes, bytes], decode: Decode
    ) -> None:
        # Arrange
        left, right = side_by_side

        # Act
        tiles = list(
            opentile_jpeg.compose(
                [(left, 0, 0), (right, 1, 0)],
                content_size=(512, 512),
                tile_size=(512, 512),
                positions=[(0, 0), (512, 0)],
            )
        )

        # Assert
        assert np.array_equal(decode(tiles[0]), decode(left))
        assert np.array_equal(decode(tiles[1]), decode(right))

    def test_420_interior_is_lossless(
        self, make_jpeg: MakeJpeg, decode: Decode
    ) -> None:
        # Arrange
        left = make_jpeg(512, 512, seed=2, subsample="4:2:0")
        right = make_jpeg(512, 512, seed=3, subsample="4:2:0")

        # Act
        (out,) = opentile_jpeg.compose(
            [(left, 0, 0), (right, 1, 0)],
            content_size=(512, 512),
            tile_size=(1024, 512),
            positions=[(0, 0)],
        )

        # Assert
        # Chroma upsampling differs only at the outer edges and the shared seam,
        # so each source's interior (trimmed by one 16px MCU) is bit-exact.
        stitched = decode(out)
        interior = np.s_[16:496, 16:496]
        assert np.array_equal(stitched[interior], decode(left)[interior])
        assert np.array_equal(stitched[16:496, 528:1008], decode(right)[interior])

    def test_keeps_trailing_source_tail(
        self, side_by_side: tuple[bytes, bytes], decode: Decode
    ) -> None:
        # Arrange
        # The sources overlap by 128px (stride 384). The trailing source has no
        # successor, so `source_size` keeps its whole 512px on the plane.
        left, right = side_by_side

        # Act
        (out,) = opentile_jpeg.compose(
            [(left, 0, 0), (right, 1, 0)],
            content_size=(384, 512),
            tile_size=(896, 512),
            positions=[(0, 0)],
            source_size=(512, 512),
        )

        # Assert
        stitched = decode(out)
        assert np.array_equal(stitched[:, 0:384], decode(left)[:, 0:384])
        assert np.array_equal(stitched[:, 384:896], decode(right))

    def test_uncovered_tile_is_neutral_gray(
        self, make_jpeg: MakeJpeg, decode: Decode, assert_solid: AssertSolid
    ) -> None:
        # Arrange
        source = make_jpeg(512, 512, seed=2, subsample="4:4:4")

        # Act
        covered, uncovered = opentile_jpeg.compose(
            [(source, 0, 0)],
            content_size=(512, 512),
            tile_size=(512, 512),
            positions=[(0, 0), (5 * 512, 0)],
        )

        # Assert
        assert np.array_equal(decode(covered), decode(source))
        assert_solid(decode(uncovered), NEUTRAL_GRAY, tolerance=0)

    def test_source_size_beyond_source_is_neutral_gray(
        self, make_jpeg: MakeJpeg, decode: Decode, assert_solid: AssertSolid
    ) -> None:
        # Arrange
        # A source_size beyond the real source must not read past its arrays.
        source = make_jpeg(512, 512, seed=2, subsample="4:4:4")

        # Act
        (out,) = opentile_jpeg.compose(
            [(source, 0, 0)],
            content_size=(512, 512),
            tile_size=(1024, 512),
            positions=[(0, 0)],
            source_size=(1024, 512),
        )

        # Assert
        stitched = decode(out)
        assert np.array_equal(stitched[:, 0:512], decode(source))
        assert_solid(stitched[:, 512:], NEUTRAL_GRAY, tolerance=0)

    def test_background_fills_uncovered_tiles(
        self, make_jpeg: MakeJpeg, decode: Decode, assert_solid: AssertSolid
    ) -> None:
        # Arrange
        source = make_jpeg(512, 512, seed=2)

        # Act
        (_, uncovered) = opentile_jpeg.compose(
            [(source, 0, 0)],
            content_size=(512, 512),
            tile_size=(512, 512),
            positions=[(0, 0), (512, 0)],
            background=RED,
        )

        # Assert
        assert_solid(decode(uncovered), RED, tolerance=60)

    def test_background_replaces_source_padding(self, make_jpeg: MakeJpeg) -> None:
        # Arrange
        # 504px wide 4:2:0: the luma arrays are padded from 63 to 64 block columns.
        source = make_jpeg(504, 512, seed=2)

        # Act
        (out,) = opentile_jpeg.compose(
            [(source, 0, 0)],
            content_size=(512, 512),
            tile_size=(512, 512),
            positions=[(0, 0)],
            background=255,
        )

        # Assert
        composed, _ = opentile_jpeg.decode_coefficients(out)
        white, _ = opentile_jpeg.decode_coefficients(opentile_jpeg.fill(out, 255))
        original, _ = opentile_jpeg.decode_coefficients(source)
        assert np.array_equal(composed[0][:, 63], white[0][:, 63])
        assert np.array_equal(composed[0][:, :63], original[0][:, :63])

    def test_rejects_no_sources(self) -> None:
        # Act & Assert
        with pytest.raises(ValueError, match="at least one source"):
            opentile_jpeg.compose(
                [], content_size=(512, 512), tile_size=(512, 512), positions=[(0, 0)]
            )

    def test_rejects_mixed_sampling(self, make_jpeg: MakeJpeg) -> None:
        # Arrange
        a_420 = make_jpeg(512, 512, seed=2, subsample="4:2:0")
        a_444 = make_jpeg(512, 512, seed=3, subsample="4:4:4")

        # Act & Assert
        with pytest.raises(ValueError, match="sampling"):
            opentile_jpeg.compose(
                [(a_420, 0, 0), (a_444, 1, 0)],
                content_size=(512, 512),
                tile_size=(512, 512),
                positions=[(0, 0)],
            )

    @pytest.mark.parametrize(
        ("content_size", "position", "match"),
        [
            ((4, 4), (0, 0), "content size"),  # smaller than an MCU
            ((0, 512), (0, 0), "content size"),
            ((512, 512), (-64, 0), "tile position"),
            ((512, 512), (8, 0), "tile position"),  # off the MCU grid
        ],
    )
    def test_rejects_invalid_geometry(
        self,
        content_size: tuple[int, int],
        position: tuple[int, int],
        match: str,
        make_jpeg: MakeJpeg,
    ) -> None:
        # Arrange
        source = make_jpeg(512, 512, seed=2)

        # Act & Assert
        with pytest.raises(ValueError, match=match):
            list(
                opentile_jpeg.compose(
                    [(source, 0, 0)],
                    content_size=content_size,
                    tile_size=(64, 64),
                    positions=[position],
                )
            )

    def test_rejects_bad_geometry_before_iteration(self, make_jpeg: MakeJpeg) -> None:
        # Arrange
        source = make_jpeg(64, 64, seed=2)

        # Act & Assert
        with pytest.raises(ValueError, match="content size"):
            opentile_jpeg.compose(
                [(source, 0, 0)], content_size=(4, 4), tile_size=(16, 16), positions=[]
            )

    def test_rejects_out_of_range_background(self, make_jpeg: MakeJpeg) -> None:
        # Arrange
        source = make_jpeg(64, 64, seed=2)

        # Act & Assert
        with pytest.raises(ValueError, match="0..255"):
            opentile_jpeg.compose(
                [(source, 0, 0)],
                content_size=(64, 64),
                tile_size=(64, 64),
                positions=[],
                background=300,
            )


class TestComposer:
    """The extension's Composer, used directly."""

    def test_keeps_sources_alive(
        self, side_by_side: tuple[bytes, bytes], decode: Decode
    ) -> None:
        # Arrange
        # The Sources have no Python references left after construction;
        # keep_alive<1, 2> must keep them alive for the Composer.
        left, right = side_by_side
        composer = _binding.Composer(
            [_binding.Source(left), _binding.Source(right)],
            [(0, 0), (1, 0)],
            (512, 512),
            (512, 512),
            (512, 512),
        )
        gc.collect()

        # Act
        out = composer.tile(0, 0)

        # Assert
        assert np.array_equal(decode(out), decode(left))

    def test_rejects_none_source(self) -> None:
        # Act & Assert
        with pytest.raises(ValueError, match="None"):
            _binding.Composer(
                [None],  # pyright: ignore[reportArgumentType]
                [(0, 0)],
                (16, 16),
                (16, 16),
                (16, 16),
            )
