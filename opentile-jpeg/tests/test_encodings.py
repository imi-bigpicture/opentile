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

"""The lossless operations across source encodings the other tests don't use:
progressive, optimized tables, restart markers, grayscale and CMYK.

The source is sized off the MCU grid, so regions reach its partial edge blocks.
Without chroma subsampling each block decodes on its own, so a lossless result
must match a slice of Pillow's decode of the source exactly. With subsampling,
upsampling blends pixels across a crop's edges, so the DCT blocks are compared
instead.
"""

import io
from dataclasses import dataclass

import numpy as np
import pytest
from PIL import Image

import opentile_jpeg

SIZE = (203, 141)  # off the 16px MCU grid in both directions
REGIONS = [(0, 0, 64, 48), (48, 32, 96, 80), (128, 96, 75, 45), (0, 0, 203, 141)]


@dataclass(frozen=True)
class Encoding:
    name: str
    mode: str  # Pillow image mode
    options: dict

    @property
    def pixel_exact(self) -> bool:
        """Whether every block decodes on its own (no chroma subsampling)."""
        return self.mode != "RGB" or self.options.get("subsampling") == "4:4:4"

    def encode(self) -> bytes:
        rng = np.random.default_rng(5)
        coarse = rng.integers(0, 256, (SIZE[1] // 8 + 1, SIZE[0] // 8 + 1, 4))
        pixels = np.repeat(np.repeat(coarse, 8, 0), 8, 1)[: SIZE[1], : SIZE[0]]
        pixels = np.clip(pixels + rng.integers(-24, 25, pixels.shape), 0, 255)
        pixels = pixels.astype(np.uint8)
        if self.mode == "CMYK":
            image = Image.frombytes("CMYK", SIZE, pixels.tobytes())
        else:
            image = Image.fromarray(pixels[..., :3]).convert(self.mode)
        out = io.BytesIO()
        image.save(out, "JPEG", quality=85, **self.options)
        jpeg = out.getvalue()
        if "restart_marker_blocks" in self.options and b"\xff\xdd" not in jpeg:
            pytest.skip("this Pillow writes no restart markers")
        return jpeg


ENCODINGS = [
    Encoding("baseline 4:4:4", "RGB", {"subsampling": "4:4:4"}),
    Encoding("progressive 4:4:4", "RGB", {"subsampling": "4:4:4", "progressive": True}),
    Encoding("optimized 4:4:4", "RGB", {"subsampling": "4:4:4", "optimize": True}),
    Encoding(
        "restart markers 4:4:4",
        "RGB",
        {"subsampling": "4:4:4", "restart_marker_blocks": 5},
    ),
    Encoding("progressive 4:2:0", "RGB", {"subsampling": "4:2:0", "progressive": True}),
    Encoding(
        "restart markers 4:2:2",
        "RGB",
        {"subsampling": "4:2:2", "restart_marker_blocks": 3},
    ),
    Encoding("progressive gray", "L", {"progressive": True}),
    Encoding("CMYK", "CMYK", {}),
]


@pytest.fixture(params=ENCODINGS, ids=lambda encoding: encoding.name)
def encoding(request: pytest.FixtureRequest) -> Encoding:
    return request.param


@pytest.fixture
def source(encoding: Encoding) -> bytes:
    return encoding.encode()


def pixels(jpeg: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(jpeg)))


def assert_blocks_match_region(
    out: bytes, source: bytes, region: tuple[int, int, int, int]
) -> None:
    """Every visible DCT block of ``out`` is the source's block at that place."""
    x, y, width, height = region
    actual, _ = opentile_jpeg.decode_coefficients(out)
    expected, info = opentile_jpeg.decode_coefficients(source)
    max_h = max(h for h, _ in info.sampling)
    max_v = max(v for _, v in info.sampling)
    for component, (h, v) in enumerate(info.sampling):
        x_step, y_step = 8 * max_h // h, 8 * max_v // v
        rows, cols = -(-height // y_step), -(-width // x_step)
        row, col = y // y_step, x // x_step
        assert np.array_equal(
            actual[component][:rows, :cols],
            expected[component][row : row + rows, col : col + cols],
        ), f"component {component} differs"


def assert_lossless_region(
    encoding: Encoding, out: bytes, source: bytes, region: tuple[int, int, int, int]
) -> None:
    x, y, width, height = region
    header = opentile_jpeg.header(out)
    assert (header.width, header.height) == (width, height)
    if encoding.pixel_exact:
        assert np.array_equal(
            pixels(out), pixels(source)[y : y + height, x : x + width]
        )
    assert_blocks_match_region(out, source, region)


@pytest.mark.parametrize("region", REGIONS)
def test_crop_is_lossless(
    encoding: Encoding, source: bytes, region: tuple[int, int, int, int]
) -> None:
    # Act
    out = opentile_jpeg.crop(source, *region)

    # Assert
    assert_lossless_region(encoding, out, source, region)


def test_crop_multiple_matches_crop(source: bytes) -> None:
    # Arrange
    same_size = [(x, y, 48, 32) for x, y, _, _ in REGIONS if x + 48 <= SIZE[0]]

    # Act
    mixed = list(opentile_jpeg.crop_multiple(source, REGIONS))
    uniform = list(opentile_jpeg.crop_multiple(source, same_size))

    # Assert
    assert mixed == [opentile_jpeg.crop(source, *region) for region in REGIONS]
    assert uniform == [opentile_jpeg.crop(source, *region) for region in same_size]


def test_compose_rebuilds_source(encoding: Encoding, source: bytes) -> None:
    # Arrange
    # Cut the source into a grid of overlapping tiles, as a scanner with overlap
    # would store it, then compose tiles of the original back out of them.
    content, source_size = (64, 48), (80, 64)
    cells = [
        (gx, gy)
        for gy in range(-(-SIZE[1] // content[1]))
        for gx in range(-(-SIZE[0] // content[0]))
    ]
    sources = []
    for gx, gy in cells:
        x, y = gx * content[0], gy * content[1]
        width, height = (
            min(source_size[0], SIZE[0] - x),
            min(source_size[1], SIZE[1] - y),
        )
        sources.append((opentile_jpeg.crop(source, x, y, width, height), gx, gy))
    tile_size = (64, 64)
    positions = [(0, 0), (32, 48), (128, 64)]

    # Act
    tiles = list(
        opentile_jpeg.compose(sources, content, tile_size, positions, source_size)
    )

    # Assert
    for (x, y), tile in zip(positions, tiles, strict=True):
        assert_lossless_region(encoding, tile, source, (x, y, *tile_size))


@pytest.mark.parametrize("progressive", [False, True])
def test_optimize_is_lossless(source: bytes, progressive: bool) -> None:
    # Act
    out = opentile_jpeg.optimize(source, progressive=progressive)

    # Assert
    assert np.array_equal(pixels(out), pixels(source))


@pytest.mark.parametrize("optimize", [False, True])
def test_coefficient_round_trip_is_lossless(source: bytes, optimize: bool) -> None:
    # Arrange
    coefficients, info = opentile_jpeg.decode_coefficients(source)

    # Act
    out = opentile_jpeg.encode_coefficients(coefficients, info, optimize=optimize)

    # Assert
    assert np.array_equal(pixels(out), pixels(source))
