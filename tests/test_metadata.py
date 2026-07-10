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

from typing import Optional

import pytest
from decoy import Decoy
from tifffile import TiffPage

from opentile.formats.ndpi.ndpi_metadata import NdpiMetadata
from opentile.formats.svs.svs_metadata import SvsMetadata


class TestSvsMetadata:
    @pytest.mark.parametrize(
        ["description", "expected"],
        [
            (
                "Aperio Image|AppMag = 20|Title = univ missouri 07.15.09",
                "univ missouri 07.15.09",
            ),
            ("Aperio Image|AppMag = 20", None),
        ],
    )
    def test_label_text(
        self, decoy: Decoy, description: str, expected: Optional[str]
    ) -> None:
        # Arrange
        page = decoy.mock(cls=TiffPage)
        decoy.when(page.description).then_return(description)

        # Act
        metadata = SvsMetadata(page)

        # Assert
        assert metadata.label_text == expected


class TestNdpiMetadata:
    @pytest.mark.parametrize(
        ["ndpi_tags", "expected"],
        [
            ({"SlideLabel": "SR1274-908A"}, "SR1274-908A"),
            ({}, None),
        ],
    )
    def test_label_text(
        self, decoy: Decoy, ndpi_tags: dict[str, str], expected: Optional[str]
    ) -> None:
        # Arrange
        page = decoy.mock(cls=TiffPage)
        decoy.when(page.tags).then_return({})
        decoy.when(page.ndpi_tags).then_return(ndpi_tags)

        # Act
        metadata = NdpiMetadata(page)

        # Assert
        assert metadata.label_text == expected
