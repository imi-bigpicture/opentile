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

"""Tests of :func:`opentile_jpeg.decode_coefficients` and
:func:`opentile_jpeg.encode_coefficients`.
"""

import numpy as np
import pytest

import opentile_jpeg
from conftest import Decode, MakeJpeg


class TestCoefficients:
    def test_decode_has_expected_shapes(self, frame_420: bytes) -> None:
        # Act
        coeffs, info = opentile_jpeg.decode_coefficients(frame_420)

        # Assert
        # 4:2:0: luma 1024x512 -> 128x64 blocks; chroma halved -> 64x32 blocks.
        luma, cb, cr = coeffs
        assert info.size == (1024, 512)
        assert luma.dtype == np.int16
        assert luma.shape == (64, 128, 8, 8)
        assert cb.shape == cr.shape == (32, 64, 8, 8)
        assert info.sampling == ((2, 2), (1, 1), (1, 1))
        assert info.block_shapes == ((64, 128), (32, 64), (32, 64))
        assert hash(info)  # frozen and hashable

    def test_decode_single_component(self, make_jpeg: MakeJpeg) -> None:
        # Arrange
        gray = make_jpeg(64, 64, subsample="gray")

        # Act
        coeffs, info = opentile_jpeg.decode_coefficients(gray)

        # Assert
        assert len(coeffs) == 1
        assert coeffs[0].shape == (8, 8, 8, 8)  # 64/8 x 64/8 blocks
        assert info.sampling == ((1, 1),)
        assert opentile_jpeg.header(gray).sampling == (1, 1)

    def test_roundtrip_is_lossless(self, frame_420: bytes, decode: Decode) -> None:
        # Arrange
        coeffs, info = opentile_jpeg.decode_coefficients(frame_420)

        # Act
        out = opentile_jpeg.encode_coefficients(coeffs, info)

        # Assert
        assert np.array_equal(decode(out), decode(frame_420))

    def test_encode_with_optimize_is_lossless_and_smaller(
        self, frame_420: bytes, decode: Decode
    ) -> None:
        # Arrange
        coeffs, info = opentile_jpeg.decode_coefficients(frame_420)

        # Act
        plain = opentile_jpeg.encode_coefficients(coeffs, info)
        optimized = opentile_jpeg.encode_coefficients(coeffs, info, optimize=True)

        # Assert
        assert np.array_equal(decode(optimized), decode(frame_420))
        assert len(optimized) <= len(plain)

    def test_edited_coefficients_take_effect(
        self, make_jpeg: MakeJpeg, decode: Decode
    ) -> None:
        # Arrange
        # Keep only each block's DC coefficient, so every block becomes flat.
        source = make_jpeg(64, 64, seed=7, subsample="4:4:4")
        coeffs, info = opentile_jpeg.decode_coefficients(source)
        for blocks in coeffs:
            dc = blocks[:, :, 0, 0].copy()
            blocks[:] = 0
            blocks[:, :, 0, 0] = dc

        # Act
        out = opentile_jpeg.encode_coefficients(coeffs, info)

        # Assert
        first_block = decode(out)[0:8, 0:8]
        assert np.all(first_block == first_block[0, 0])

    def test_encode_rejects_mismatched_shape(self, frame_420: bytes) -> None:
        # Arrange
        coeffs, info = opentile_jpeg.decode_coefficients(frame_420)
        coeffs[0] = np.zeros((4, 4, 8, 8), dtype=np.int16)

        # Act & Assert
        with pytest.raises(ValueError, match="component 0"):
            opentile_jpeg.encode_coefficients(coeffs, info)

    def test_encode_rejects_wrong_component_count(self, frame_420: bytes) -> None:
        # Arrange
        coeffs, info = opentile_jpeg.decode_coefficients(frame_420)

        # Act & Assert
        with pytest.raises(ValueError, match="expected 3 coefficient arrays, got 2"):
            opentile_jpeg.encode_coefficients(coeffs[:2], info)
