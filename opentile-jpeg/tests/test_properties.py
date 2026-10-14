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

"""Property tests: generated geometry checked against a block-by-block model.

Hypothesis draws JPEGs and geometry: sizes off the MCU grid, regions at, past
or off the image edges and the MCU grid, and sparse, overlapping source grids.
For every block of a result, a plain model of the operation says where that
block must come from (a source block, or the flat fill block), and the block
must equal it exactly, so a misplaced or garbage block fails as surely as a
crash. Invalid geometry must raise ``ValueError`` and nothing else.
"""

import io
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from PIL import Image

import opentile_jpeg

Size = tuple[int, int]
Background = tuple[int, int, int] | None
Blocks = list[np.ndarray]  # one coefficient array per component

FILL = "fill"  # the flat background block, or neutral gray without a background
UNCHECKED = "unchecked"  # copied from MCU padding, not visible to the test
Origin = tuple[int, int, int] | str  # (source index, row, col), FILL or UNCHECKED

MCU_SIZES = {"4:4:4": (8, 8), "4:2:2": (16, 8), "4:2:0": (16, 16), "gray": (8, 8)}

subsamplings = st.sampled_from(list(MCU_SIZES))
seeds = st.integers(0, 2**32 - 1)
backgrounds = st.none() | st.tuples(*[st.integers(0, 255)] * 3)


def multiples(step: int, most: int) -> st.SearchStrategy[int]:
    """0, step, 2 * step, ... up to ``most`` steps."""
    return st.integers(0, most).map(lambda count: count * step)


def near(value: int, below: int, above: int) -> st.SearchStrategy[int]:
    """From ``below`` under ``value`` to ``above`` over it, and at least 1."""
    return st.integers(max(1, value - below), value + above)


def encode(size: Size, subsampling: str, seed: int) -> bytes:
    """A JPEG of random pixels; same subsampling and quality, same tables."""
    width, height = size
    pixels = np.random.default_rng(seed).integers(0, 256, (height, width, 3))
    image = Image.fromarray(pixels.astype(np.uint8))
    out = io.BytesIO()
    if subsampling == "gray":
        image.convert("L").save(out, "JPEG", quality=80)
    else:
        image.save(out, "JPEG", quality=80, subsampling=subsampling)
    return out.getvalue()


def round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


@dataclass(frozen=True)
class BlockGrid:
    """How one component's 8x8 blocks lie on the image's pixels."""

    x_step: int  # pixels one block covers
    y_step: int
    h_samp: int
    v_samp: int

    @classmethod
    def of(cls, info: opentile_jpeg.CoefficientInfo, component: int) -> "BlockGrid":
        max_h = max(h for h, _ in info.sampling)
        max_v = max(v for _, v in info.sampling)
        h, v = info.sampling[component]
        return cls(8 * max_h // h, 8 * max_v // v, h, v)

    def blocks(self, size: Size) -> Size:
        """A pixel size in whole blocks, as (cols, rows)."""
        return size[0] // self.x_step, size[1] // self.y_step

    def in_padding(self, row: int, col: int, shape: Size) -> bool:
        """Whether (row, col) lies past ``shape`` but in its padding to whole MCUs."""
        rows, cols = shape
        return row < round_up(rows, self.v_samp) and col < round_up(cols, self.h_samp)


def grids_of(jpeg: bytes) -> list[BlockGrid]:
    _, info = opentile_jpeg.decode_coefficients(jpeg)
    return [BlockGrid.of(info, component) for component in range(len(info.sampling))]


def flat_blocks(template: bytes, background: Background) -> Blocks:
    """Per component, the block filling areas outside the sources."""
    if background is None:
        return [np.zeros((8, 8), np.int16) for _ in grids_of(template)]
    filled, _ = opentile_jpeg.decode_coefficients(
        opentile_jpeg.fill(template, background)
    )
    return [blocks[0, 0] for blocks in filled]


def assert_matches_model(
    actual: Blocks,
    sources: list[Blocks],
    flats: Blocks,
    origin: Callable[[int, int, int], Origin],
) -> None:
    """Every block of ``actual`` equals the block ``origin`` names for it."""
    for component, blocks in enumerate(actual):
        for row, col in np.ndindex(blocks.shape[:2]):
            where = origin(component, row, col)
            if where == UNCHECKED:
                continue
            if isinstance(where, tuple):
                source, source_row, source_col = where
                expected = sources[source][component][source_row, source_col]
            else:
                expected = flats[component]
            assert np.array_equal(blocks[row, col], expected), (
                f"component {component} block ({row}, {col}) should come from {where}"
            )


@dataclass(frozen=True)
class CropCase:
    subsampling: str
    image_size: Size
    region: tuple[int, int, int, int]
    background: Background

    @property
    def valid(self) -> bool:
        mcu_width, mcu_height = MCU_SIZES[self.subsampling]
        image_width, image_height = self.image_size
        x, y, width, height = self.region
        inside = x + width <= image_width and y + height <= image_height
        return (
            width > 0
            and height > 0
            and x >= 0
            and y >= 0
            and x % mcu_width == 0
            and y % mcu_height == 0
            and (inside or self.background is not None)
        )


@st.composite
def crop_cases(draw: st.DrawFn) -> CropCase:
    subsampling = draw(subsamplings)
    mcu_width, mcu_height = MCU_SIZES[subsampling]
    image_width, image_height = draw(st.integers(1, 96)), draw(st.integers(1, 96))

    if draw(st.booleans()):  # an aligned region inside the image
        x = draw(multiples(mcu_width, (image_width - 1) // mcu_width))
        y = draw(multiples(mcu_height, (image_height - 1) // mcu_height))
        width = draw(st.integers(1, image_width - x))
        height = draw(st.integers(1, image_height - y))
    else:  # anywhere around the image, mostly on the MCU grid

        def origin(mcu: int, extent: int) -> int:
            aligned = mcu * draw(st.integers(-1, extent // mcu + 2))
            return aligned + draw(st.sampled_from([0, 0, 0, 1, mcu // 2]))

        x, y = origin(mcu_width, image_width), origin(mcu_height, image_height)
        width = draw(st.integers(0, image_width + 2 * mcu_width))
        height = draw(st.integers(0, image_height + 2 * mcu_height))
    region = (x, y, width, height)
    return CropCase(subsampling, (image_width, image_height), region, draw(backgrounds))


def crop_origin(
    case: CropCase,
    grids: list[BlockGrid],
    source: Blocks,
    component: int,
    row: int,
    col: int,
) -> Origin:
    """Where block (row, col) of a crop comes from: the source, or the fill."""
    x, y, _, _ = case.region
    row += y // grids[component].y_step
    col += x // grids[component].x_step
    rows, cols = source[component].shape[:2]
    return (0, row, col) if row < rows and col < cols else FILL


@given(case=crop_cases().filter(lambda case: case.valid), seed=seeds)
def test_crop_matches_model(case: CropCase, seed: int) -> None:
    # Arrange
    jpeg = encode(case.image_size, case.subsampling, seed)
    source, _ = opentile_jpeg.decode_coefficients(jpeg)
    origin = partial(crop_origin, case, grids_of(jpeg), source)

    # Act
    out = opentile_jpeg.crop(jpeg, *case.region, background=case.background)
    batch = opentile_jpeg.crop_multiple(
        jpeg, [case.region] * 2, background=case.background
    )

    # Assert
    assert list(batch) == [out, out]  # the Cropper path agrees byte for byte
    actual, info = opentile_jpeg.decode_coefficients(out)
    assert info.size == case.region[2:]
    assert_matches_model(actual, [source], flat_blocks(jpeg, case.background), origin)


@given(case=crop_cases().filter(lambda case: not case.valid), seed=seeds)
def test_crop_rejects_invalid_geometry(case: CropCase, seed: int) -> None:
    # Arrange
    jpeg = encode(case.image_size, case.subsampling, seed)

    # Act & Assert
    with pytest.raises(ValueError):
        opentile_jpeg.crop(jpeg, *case.region, background=case.background)


@dataclass(frozen=True)
class ComposeCase:
    subsampling: str
    content_size: Size
    source_size: Size | None
    cells: list[Size]
    image_sizes: list[Size]
    tile_size: Size
    positions: list[Size]
    background: Background

    @property
    def kept_size(self) -> Size:
        return self.source_size or self.content_size

    @property
    def last_cell(self) -> Size:
        return max(x for x, _ in self.cells), max(y for _, y in self.cells)


@st.composite
def compose_cases(draw: st.DrawFn) -> ComposeCase:
    subsampling = draw(subsamplings)
    mcu_width, mcu_height = MCU_SIZES[subsampling]
    content = (
        mcu_width * draw(st.integers(1, 4)),
        mcu_height * draw(st.integers(1, 4)),
    )
    overlap = (draw(multiples(mcu_width, 2)), draw(multiples(mcu_height, 2)))
    full = (content[0] + overlap[0], content[1] + overlap[1])
    cells = draw(
        st.lists(
            st.tuples(st.integers(0, 2), st.integers(0, 2)),
            min_size=1,
            max_size=5,
            unique=True,
        )
    )
    # Sources a little smaller or larger than their nominal size, so their
    # arrays end on and off the MCU grid.
    image_sizes = [
        (draw(near(full[0], mcu_width + 3, 4)), draw(near(full[1], mcu_height + 3, 4)))
        for _ in cells
    ]
    tile = (draw(st.integers(1, 3 * content[0])), draw(st.integers(1, 3 * content[1])))
    positions = draw(
        st.lists(
            st.tuples(
                multiples(mcu_width, 4 * content[0] // mcu_width),
                multiples(mcu_height, 4 * content[1] // mcu_height),
            ),
            min_size=1,
            max_size=3,
        )
    )
    return ComposeCase(
        subsampling,
        content,
        draw(st.sampled_from([None, full])),
        cells,
        image_sizes,
        tile,
        positions,
        draw(backgrounds),
    )


def compose_origin(
    case: ComposeCase,
    grids: list[BlockGrid],
    sources: list[Blocks],
    position: Size,
    component: int,
    row: int,
    col: int,
) -> Origin:
    """Where block (row, col) of the tile at ``position`` comes from."""
    grid = grids[component]
    content_cols, content_rows = grid.blocks(case.content_size)
    kept_cols, kept_rows = grid.blocks(case.kept_size)
    last_x, last_y = case.last_cell

    # The block's place on the plane, and the grid cell covering it; past the
    # last cell, the last source keeps covering (its tail).
    plane_row = position[1] // grid.y_step + row
    plane_col = position[0] // grid.x_step + col
    cell = (
        min(plane_col // content_cols, last_x),
        min(plane_row // content_rows, last_y),
    )
    if cell not in case.cells:
        return FILL

    source = case.cells.index(cell)
    row = plane_row - cell[1] * content_rows
    col = plane_col - cell[0] * content_cols
    if row >= kept_rows or col >= kept_cols:
        return FILL
    shape = sources[source][component].shape[:2]
    if row < shape[0] and col < shape[1]:
        return source, row, col
    # Past the source's image: its padding is copied unless a background is set.
    if case.background is None and grid.in_padding(row, col, shape):
        return UNCHECKED
    return FILL


@given(case=compose_cases(), seed=seeds)
def test_compose_matches_model(case: ComposeCase, seed: int) -> None:
    # Arrange
    jpegs = [
        encode(size, case.subsampling, seed + index)
        for index, size in enumerate(case.image_sizes)
    ]
    sources = [opentile_jpeg.decode_coefficients(jpeg)[0] for jpeg in jpegs]
    grids = grids_of(jpegs[0])
    flats = flat_blocks(jpegs[0], case.background)

    # Act
    tiles = opentile_jpeg.compose(
        [(jpeg, gx, gy) for jpeg, (gx, gy) in zip(jpegs, case.cells, strict=True)],
        content_size=case.content_size,
        tile_size=case.tile_size,
        positions=case.positions,
        source_size=case.source_size,
        background=case.background,
    )

    # Assert
    for tile, position in zip(tiles, case.positions, strict=True):
        actual, info = opentile_jpeg.decode_coefficients(tile)
        assert info.size == case.tile_size
        origin = partial(compose_origin, case, grids, sources, position)
        assert_matches_model(actual, sources, flats, origin)


@given(
    subsampling=subsamplings,
    size=st.tuples(st.integers(1, 96), st.integers(1, 96)),
    seed=seeds,
    progressive=st.booleans(),
)
def test_reencoding_keeps_every_coefficient(
    subsampling: str, size: Size, seed: int, progressive: bool
) -> None:
    # Arrange
    jpeg = encode(size, subsampling, seed)
    original, info = opentile_jpeg.decode_coefficients(jpeg)

    # Act
    outputs = (
        opentile_jpeg.optimize(jpeg, progressive=progressive),
        opentile_jpeg.encode_coefficients(original, info, optimize=True),
    )

    # Assert
    for out in outputs:
        reencoded, _ = opentile_jpeg.decode_coefficients(out)
        for before, after in zip(original, reencoded, strict=True):
            np.testing.assert_array_equal(before, after)


@given(
    subsampling=subsamplings,
    size=st.tuples(st.integers(1, 96), st.integers(1, 96)),
    seed=seeds,
    color=st.tuples(*[st.integers(0, 255)] * 3),
)
def test_fill_writes_one_flat_block_per_component(
    subsampling: str, size: Size, seed: int, color: tuple[int, int, int]
) -> None:
    # Arrange
    jpeg = encode(size, subsampling, seed)

    # Act
    out = opentile_jpeg.fill(jpeg, color)

    # Assert
    filled, info = opentile_jpeg.decode_coefficients(out)
    assert info.size == size
    for blocks in filled:
        dc = blocks[..., 0, 0]
        ac = blocks.reshape(*blocks.shape[:2], 64)[..., 1:]
        assert (dc == dc[0, 0]).all()
        assert not ac.any()
