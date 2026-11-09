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

"""Tests of :func:`opentile_jpeg.header`."""

import opentile_jpeg


class TestHeader:
    def test_reads_dimensions_and_sampling(self, frame_420: bytes) -> None:
        # Act
        header = opentile_jpeg.header(frame_420)

        # Assert
        assert (header.width, header.height) == (1024, 512)
        assert header.sampling == (2, 2)  # 4:2:0 luma sampling factors

    def test_444_reports_no_subsampling(self, frame_444: bytes) -> None:
        # Act
        header = opentile_jpeg.header(frame_444)

        # Assert
        assert header.sampling == (1, 1)
