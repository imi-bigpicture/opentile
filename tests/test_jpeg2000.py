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

from collections.abc import Sequence

import pytest

from opentile.jpeg2000 import Jpeg2000, Jpeg2000Info


def _siz(
    factors: Sequence[tuple[int, int]],
    bit_depth: int = 8,
    sizes: bytes = bytes(32),
    rsiz: int = 0,
) -> bytes:
    """Build a SIZ marker segment with one (XRsiz, YRsiz) per component.

    `sizes` is the 32-byte Xsiz..YTOsiz block, exposed so a test can embed a
    stray marker sequence inside the SIZ segment. `rsiz` is the capabilities
    field.
    """
    body = rsiz.to_bytes(2, "big")  # Rsiz
    body += sizes  # Xsiz, Ysiz, XOsiz, YOsiz, XTsiz, YTsiz, XTOsiz, YTOsiz
    body += len(factors).to_bytes(2, "big")  # Csiz
    for horizontal, vertical in factors:
        body += bytes([bit_depth - 1, horizontal, vertical])  # Ssiz, XRsiz, YRsiz
    return b"\xff\x51" + (len(body) + 2).to_bytes(2, "big") + body


def _cod(transformation: int, mct: int = 0) -> bytes:
    """Build a COD marker segment with the given transformation and MCT bytes."""
    # Scod, progression, layers(2), mct, levels, cbw, cbh, style, transformation
    payload = bytes(
        [0x00, 0x00, 0x00, 0x01, mct, 0x05, 0x04, 0x04, 0x00, transformation]
    )
    return b"\xff\x52" + (len(payload) + 2).to_bytes(2, "big") + payload


def _codestream(
    transformation: int = 0,
    mct: int = 0,
    factors: Sequence[tuple[int, int]] = ((1, 1), (2, 1), (2, 1)),
    bit_depth: int = 8,
    sizes: bytes = bytes(32),
    rsiz: int = 0,
) -> bytes:
    """Build a minimal JPEG 2000 codestream (SOC + SIZ + COD)."""
    return (
        b"\xff\x4f" + _siz(factors, bit_depth, sizes, rsiz) + _cod(transformation, mct)
    )


@pytest.mark.unittest
class TestJpeg2000Parse:
    def test_irreversible_subsampled_ybr(self):
        # Arrange — like Aperio 33003: lossy, no MCT, 4:2:2 chroma
        codestream = _codestream(
            transformation=0, mct=0, factors=[(1, 1), (2, 1), (2, 1)]
        )

        # Act
        info = Jpeg2000.parse(codestream)

        # Assert
        assert info == Jpeg2000Info(
            reversible=False,
            uses_mct=False,
            components=3,
            subsampling=(2, 1),
            bit_depth=8,
            extended=False,
        )

    def test_reversible_with_mct(self):
        # Arrange — reversible, MCT applied, no subsampling
        codestream = _codestream(
            transformation=1, mct=1, factors=[(1, 1), (1, 1), (1, 1)]
        )

        # Act
        info = Jpeg2000.parse(codestream)

        # Assert
        assert info == Jpeg2000Info(
            reversible=True,
            uses_mct=True,
            components=3,
            subsampling=(1, 1),
            bit_depth=8,
            extended=False,
        )

    def test_single_component_has_no_subsampling(self):
        # Arrange
        codestream = _codestream(factors=[(1, 1)], bit_depth=16)

        # Act
        info = Jpeg2000.parse(codestream)

        # Assert
        assert info is not None
        assert info.components == 1
        assert info.subsampling is None
        assert info.bit_depth == 16

    @pytest.mark.parametrize(
        ["rsiz", "expected"], [(0, False), (1, True), (0x8000, True)]
    )
    def test_extended_from_rsiz(self, rsiz: int, expected: bool):
        # Arrange
        codestream = _codestream(rsiz=rsiz)

        # Act
        info = Jpeg2000.parse(codestream)

        # Assert
        assert info is not None
        assert info.extended is expected

    def test_stray_cod_in_siz_is_skipped(self):
        # Arrange — a COD byte sequence inside the SIZ data must not be read as COD
        codestream = _codestream(transformation=1, sizes=b"\xff\x52" + bytes(30))

        # Act
        info = Jpeg2000.parse(codestream)

        # Assert — the real COD after the SIZ is used
        assert info is not None
        assert info.reversible is True

    def test_missing_cod_returns_none(self):
        # Arrange — SOC + SIZ only, no COD
        codestream = b"\xff\x4f" + _siz([(1, 1), (1, 1), (1, 1)])

        # Act & Assert
        assert Jpeg2000.parse(codestream) is None


@pytest.mark.unittest
class TestJpeg2000Reversible:
    def test_irreversible(self):
        assert Jpeg2000.reversible(_codestream(transformation=0)) is False

    def test_reversible(self):
        assert Jpeg2000.reversible(_codestream(transformation=1)) is True

    def test_missing_cod_returns_none(self):
        assert Jpeg2000.reversible(b"\xff\x4f" + _siz([(1, 1)])) is None
