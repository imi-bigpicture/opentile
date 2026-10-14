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

"""The engine releases the GIL, so concurrent calls must give the serial results."""

from concurrent.futures import ThreadPoolExecutor

import pytest

import opentile_jpeg
from conftest import MakeJpeg

JOBS = 48


@pytest.fixture
def jpegs(make_jpeg: MakeJpeg) -> list[bytes]:
    return [
        make_jpeg(512, 384, seed=seed, subsample=subsample)
        for seed, subsample in enumerate(["4:2:0", "4:4:4", "gray"])
    ]


def work(jpegs: list[bytes], index: int) -> list[bytes]:
    """Every operation, with arguments varying by job."""
    jpeg = jpegs[index % len(jpegs)]
    shift = 16 * (index % 8)
    out = [
        opentile_jpeg.crop(jpeg, shift, 16, 256, 256, background=255),
        opentile_jpeg.fill(jpeg, (index % 256, 30, 200)),
        opentile_jpeg.optimize(jpeg, progressive=index % 2 == 0),
    ]
    out += opentile_jpeg.crop_multiple(jpeg, [(0, 0, 128, 128), (shift, 128, 128, 128)])
    out += opentile_jpeg.compose(
        [(jpeg, 0, 0), (jpeg, 1, 0)],
        (256, 384),
        (256, 256),
        [(shift, 0), (384, 128)],
        source_size=(512, 384),
    )
    coefficients, info = opentile_jpeg.decode_coefficients(jpeg)
    out.append(opentile_jpeg.encode_coefficients(coefficients, info, optimize=True))
    return out


@pytest.mark.parametrize("workers", [4, 16])
def test_concurrent_calls_match_serial(jpegs: list[bytes], workers: int) -> None:
    # Arrange
    serial = [work(jpegs, index) for index in range(JOBS)]

    # Act
    with ThreadPoolExecutor(workers) as pool:
        concurrent = list(pool.map(lambda index: work(jpegs, index), range(JOBS)))

    # Assert
    assert concurrent == serial
