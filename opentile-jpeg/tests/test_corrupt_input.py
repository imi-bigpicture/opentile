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

"""Corrupt input: every operation must fail with JpegError or ValueError (or
MemoryError), never another exception or a crash.

Hypothesis corrupts valid JPEGs by overwriting, inserting and dropping bytes,
and corrupts decoded coefficient arrays before re-encoding them.
"""

import io
import warnings
from collections.abc import Iterator
from contextlib import contextmanager, suppress

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from PIL import Image

import opentile_jpeg

ALLOWED = (opentile_jpeg.JpegError, ValueError, MemoryError)
# A corrupted header can claim a huge image. A limit far below the default keeps
# the test from allocating hundreds of MB for one, and exercises the refusal.
MAX_PIXELS = 4_000_000


def _encode(subsampling: str, **options) -> bytes:
    pixels = np.random.default_rng(3).integers(0, 256, (40, 56, 3), np.uint8)
    image = Image.fromarray(pixels)
    out = io.BytesIO()
    if subsampling == "gray":
        image.convert("L").save(out, "JPEG", **options)
    else:
        image.save(out, "JPEG", subsampling=subsampling, **options)
    return out.getvalue()


SOURCES = {
    "baseline 4:2:0": _encode("4:2:0"),
    "progressive 4:4:4": _encode("4:4:4", progressive=True),
    "gray with restart markers": _encode("gray", restart_marker_blocks=2),
}
sources = st.sampled_from(list(SOURCES))  # by name, for readable failure reports

edits = st.lists(
    st.tuples(
        st.sampled_from(["overwrite", "insert", "delete"]),
        st.floats(0, 1),  # where, as a fraction of the length
        st.binary(min_size=1, max_size=4),
    ),
    min_size=1,
    max_size=6,
)


def corrupt(jpeg: bytes, changes: list[tuple[str, float, bytes]]) -> bytes:
    data = bytearray(jpeg)
    for kind, where, value in changes:
        index = int(where * len(data))
        if kind == "overwrite":
            data[index : index + len(value)] = value
        elif kind == "insert":
            data[index:index] = value
        else:
            del data[index : index + len(value)]
    return bytes(data)


@contextmanager
def tolerating_corrupt_data() -> Iterator[None]:
    """Swallow the errors corrupt data may raise, and ignore its warnings; any
    other exception propagates and fails the test."""
    with warnings.catch_warnings(), suppress(*ALLOWED):
        warnings.simplefilter("ignore", opentile_jpeg.JpegWarning)
        yield


@given(name=sources, changes=edits)
def test_corrupt_jpeg(name: str, changes: list[tuple[str, float, bytes]]) -> None:
    # Arrange
    source = SOURCES[name]
    jpeg = corrupt(source, changes)
    regions = [(0, 0, 16, 16), (16, 0, 32, 32)]
    grid = [(source, 0, 0), (jpeg, 1, 0)]

    # Act & Assert
    # The lazy operations are consumed in full, as they fail while producing.
    with tolerating_corrupt_data():
        opentile_jpeg.header(jpeg)
    with tolerating_corrupt_data():
        opentile_jpeg.decode_coefficients(jpeg, max_pixels=MAX_PIXELS)
    with tolerating_corrupt_data():
        opentile_jpeg.crop(jpeg, 16, 16, 24, 24, background=255, max_pixels=MAX_PIXELS)
    with tolerating_corrupt_data():
        list(
            opentile_jpeg.crop_multiple(
                jpeg, regions, background=0, max_pixels=MAX_PIXELS
            )
        )
    with tolerating_corrupt_data():
        opentile_jpeg.fill(jpeg, (10, 200, 30), max_pixels=MAX_PIXELS)
    with tolerating_corrupt_data():
        opentile_jpeg.optimize(jpeg, progressive=True, max_pixels=MAX_PIXELS)
    with tolerating_corrupt_data():
        list(
            opentile_jpeg.compose(
                grid, (16, 16), (32, 16), [(0, 0)], max_pixels=MAX_PIXELS
            )
        )


@given(
    name=sources,
    kind=st.sampled_from(["truncate", "extreme value", "drop component", "dtype"]),
    data=st.data(),
)
def test_corrupt_coefficients(name: str, kind: str, data: st.DataObject) -> None:
    # Arrange
    coefficients, info = opentile_jpeg.decode_coefficients(SOURCES[name])
    coefficients = [array.copy() for array in coefficients]
    component = data.draw(st.integers(0, len(coefficients) - 1))
    array = coefficients[component]
    if kind == "truncate":
        coefficients[component] = array[: data.draw(st.integers(0, array.shape[0] - 1))]
    elif kind == "extreme value":
        index = data.draw(st.integers(0, array.size - 1))
        array.flat[index] = data.draw(st.sampled_from([-32768, 32767, -2048, 2047]))
    elif kind == "drop component":
        del coefficients[component]
    else:
        coefficients[component] = array.astype(
            data.draw(st.sampled_from([np.int32, np.float64, np.uint8]))
        )

    optimize = data.draw(st.booleans())

    # Act & Assert
    with tolerating_corrupt_data():
        opentile_jpeg.encode_coefficients(coefficients, info, optimize=optimize)
