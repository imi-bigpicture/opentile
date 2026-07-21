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
from unittest.mock import Mock

import pytest
from decoy import Decoy
from tifffile import TiffPage

from opentile.formats.ndpi.ndpi_metadata import NdpiMetadata
from opentile.formats.philips.philips_tiff_metadata import PhilipsTiffMetadata
from opentile.formats.svs.svs_metadata import SvsMetadata
from opentile.formats.ventana.ventana_tiff_metadata import VentanaMetadata


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
            ("Aperio Image|AppMag = 20|Barcode = SR1274-908A", "SR1274-908A"),
            ("Aperio Image|AppMag = 20", None),
        ],
    )
    def test_barcode(
        self, decoy: Decoy, description: str, expected: Optional[str]
    ) -> None:
        # Arrange
        page = decoy.mock(cls=TiffPage)
        decoy.when(page.description).then_return(description)

        # Act
        metadata = SvsMetadata(page)

        # Assert
        assert metadata.barcode == expected


class TestPhilipsMetadata:
    @pytest.mark.parametrize(
        ["barcode_xml", "expected"],
        [
            # Philips stores the barcode Base64-encoded; "TEFCRUwwMDAx" -> "LABEL0001".
            ('<Attribute Name="PIM_DP_UFS_BARCODE">TEFCRUwwMDAx</Attribute>', "LABEL0001"),
            # Non-Base64 falls back to the raw value.
            ('<Attribute Name="PIM_DP_UFS_BARCODE">SR1274-908A</Attribute>', "SR1274-908A"),
            ("", None),
        ],
    )
    def test_barcode(self, barcode_xml: str, expected: Optional[str]) -> None:
        # Arrange
        tiff_file = Mock()
        tiff_file.philips_metadata = f"<DataObject>{barcode_xml}</DataObject>"

        # Act
        metadata = PhilipsTiffMetadata(tiff_file)

        # Assert
        assert metadata.barcode == expected


class TestVentanaMetadata:
    @pytest.mark.parametrize(
        ["iscan", "expected"],
        [
            ('<iScan Barcode1D="SR1274-908A" Barcode2D="2D-VAL" />', "SR1274-908A"),
            ('<iScan Barcode2D="2D-VAL" />', "2D-VAL"),
            ('<iScan Magnification="40" />', None),
        ],
    )
    def test_barcode(self, iscan: str, expected: Optional[str]) -> None:
        # Arrange
        page = Mock()
        page.tags = {"XMP": Mock(value=iscan)}

        # Act
        metadata = VentanaMetadata(page)

        # Assert
        assert metadata.barcode == expected


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
