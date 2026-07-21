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
from opentile.formats.svs.svs_image import SvsTiledImage
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

    @pytest.mark.parametrize(
        ["description", "expected"],
        [
            ("Aperio Image|AppMag = 20|MPP = 0.25", 0.25),
            # Hamamatsu-saved-as-Aperio uses a ',' decimal separator
            ("Aperio Image|AppMag = 20|MPP = 0,4533", 0.4533),
            # Tecmed omit the MPP key; resolution is free text in the header
            (
                "Aperio Image Library v12.3.1 \r\n149949x66142 (256x256) J2K/KDU "
                "Q=50;Scan; Scan resolution 0.274 \xb5m/Pix; Scan position X=26 mm;",
                0.274,
            ),
        ],
    )
    def test_mpp(self, decoy: Decoy, description: str, expected: float) -> None:
        # Arrange
        page = decoy.mock(cls=TiffPage)
        decoy.when(page.description).then_return(description)

        # Act
        metadata = SvsMetadata(page)

        # Assert
        assert metadata.mpp == expected

    def test_mpp_missing_raises(self, decoy: Decoy) -> None:
        # Arrange
        page = decoy.mock(cls=TiffPage)
        decoy.when(page.description).then_return("Aperio Image|AppMag = 20")
        metadata = SvsMetadata(page)

        # Act, Assert
        with pytest.raises(ValueError):
            metadata.mpp

    @pytest.mark.parametrize(
        ["description", "expected"],
        [
            ("Aperio Image|AppMag = 20|OffsetZ = 1.8", 1.8),
            ("Aperio Image|AppMag = 20", 0.0),
            # Leica GT450 sub-level pages have an empty description (no
            # 'Aperio ' header) — must default rather than raise.
            ("", 0.0),
        ],
    )
    def test_get_focal_plane(
        self, decoy: Decoy, description: str, expected: float
    ) -> None:
        # Arrange
        page = decoy.mock(cls=TiffPage)
        decoy.when(page.description).then_return(description)
        image = SvsTiledImage.__new__(SvsTiledImage)  # bypass heavy __init__
        image._page = page

        # Act
        focal_plane = image._get_focal_plane()

        # Assert
        assert focal_plane == expected


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
        decoy.when(page.ndpi_tags).then_return(ndpi_tags)

        # Act
        metadata = NdpiMetadata(page)

        # Assert
        assert metadata.label_text == expected
