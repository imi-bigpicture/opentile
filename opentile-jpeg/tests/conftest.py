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

"""Fixtures shared by the opentile_jpeg tests, and the Hypothesis profiles.

The tests check each operation's contract (losslessness, array shapes, color,
neutral fill) against a Pillow reference decode. The helpers are fixtures; test
modules import only the type aliases and colors below.

Hypothesis runs the ``default`` profile locally and ``ci`` in the wheel builds.
Encoding the generated JPEGs makes example durations vary with the machine, so
neither has a deadline. The wheel builds run the suite on every platform and
Python, so ``ci`` (``pytest --hypothesis-profile=ci``) runs fewer, derandomized
examples, the same on every run.
"""

import io
from collections.abc import Callable

import numpy as np
import pytest
from hypothesis import settings
from PIL import Image

settings.register_profile("default", max_examples=100, deadline=None)
settings.register_profile("ci", max_examples=50, deadline=None, derandomize=True)

MakeJpeg = Callable[..., bytes]
Decode = Callable[[bytes], np.ndarray]
AssertSolid = Callable[..., None]

WHITE = (255, 255, 255)
RED = (255, 0, 0)
NEUTRAL_GRAY = 128  # what an all-zero DCT block decodes to


def _make_jpeg(
    width: int, height: int, seed: int = 0, subsample: str = "4:2:0"
) -> bytes:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, size=(height // 8, width // 8, 3), dtype=np.uint8)
    image = Image.fromarray(np.repeat(np.repeat(base, 8, axis=0), 8, axis=1))
    out = io.BytesIO()
    if subsample == "gray":
        image.convert("L").save(out, "JPEG", quality=85)
    else:
        image.save(out, "JPEG", quality=85, subsampling=subsample)
    return out.getvalue()


@pytest.fixture
def make_jpeg() -> MakeJpeg:
    """Make a JPEG of smooth-ish 8x8-block noise (compresses without huge files).

    Call as ``make_jpeg(width, height, seed=0, subsample="4:2:0")``, with
    ``subsample`` one of ``"4:4:4"``, ``"4:2:0"`` or ``"gray"``.
    """
    return _make_jpeg


@pytest.fixture
def decode() -> Decode:
    """Decode a JPEG to pixels with Pillow, the reference decoder."""

    def decode(jpeg: bytes) -> np.ndarray:
        return np.asarray(Image.open(io.BytesIO(jpeg)))

    return decode


@pytest.fixture
def assert_solid() -> AssertSolid:
    """Assert every pixel is within ``tolerance`` of ``color`` (JPEG rounding).

    Call as ``assert_solid(pixels, color, tolerance)``.
    """

    def assert_solid(
        pixels: np.ndarray, color: int | tuple[int, int, int], tolerance: int
    ) -> None:
        assert np.abs(pixels.astype(int) - np.asarray(color)).max() <= tolerance

    return assert_solid


@pytest.fixture
def frame_420() -> bytes:
    """1024x512 4:2:0 frame (16px MCU, the WSI target subsampling)."""
    return _make_jpeg(1024, 512, seed=1)


@pytest.fixture
def frame_444() -> bytes:
    """1024x512 4:4:4 frame: no chroma subsampling, so a lossless block copy is
    pixel-exact against a slice of the source (no upsampling-at-the-seam noise)."""
    return _make_jpeg(1024, 512, seed=1, subsample="4:4:4")


@pytest.fixture
def side_by_side() -> tuple[bytes, bytes]:
    """Two different 512x512 4:4:4 sources, to place next to each other."""
    left = _make_jpeg(512, 512, seed=2, subsample="4:4:4")
    right = _make_jpeg(512, 512, seed=3, subsample="4:4:4")
    return left, right
