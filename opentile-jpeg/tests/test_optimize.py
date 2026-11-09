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

"""Tests of :func:`opentile_jpeg.optimize`."""

import numpy as np

import opentile_jpeg
from conftest import Decode


class TestOptimize:
    def test_is_lossless_and_smaller(self, frame_420: bytes, decode: Decode) -> None:
        # Act
        out = opentile_jpeg.optimize(frame_420)

        # Assert
        assert np.array_equal(decode(out), decode(frame_420))
        assert len(out) <= len(frame_420)

    def test_progressive_is_lossless(self, frame_420: bytes, decode: Decode) -> None:
        # Act
        out = opentile_jpeg.optimize(frame_420, progressive=True)

        # Assert
        assert np.array_equal(decode(out), decode(frame_420))
