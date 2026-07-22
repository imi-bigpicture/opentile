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

from datetime import datetime
from typing import Optional, cast
from unittest.mock import Mock

import pytest
from decoy import Decoy
from tifffile import TiffPage, TiffTags

from opentile.formats.huron.huron_tiff_metadata import HuronTiffMetadata
from opentile.formats.mikroscan.mikroscan_tiff_metadata import MikroscanTiffMetadata
from opentile.formats.motic.motic_tiff_metadata import MoticTiffMetadata
from opentile.formats.ndpi.ndpi_metadata import NdpiMetadata
from opentile.formats.philips.philips_tiff_metadata import PhilipsTiffMetadata
from opentile.formats.svs.svs_image import SvsTiledImage
from opentile.formats.svs.svs_metadata import SvsMetadata
from opentile.formats.ventana.ventana_tiff_metadata import VentanaMetadata


class TestSvsMetadata:
    @pytest.mark.parametrize(
        ["description", "manufacturer", "model", "software_versions"],
        [
            ("Aperio Image, Grundium Ocus|MPP = 0.5", "Grundium", "Ocus", None),
            ("Aperio Image, Grundium Ocus II|MPP = 0.5", "Grundium", "Ocus II", None),
            (
                "Aperio Leica Biosystems GT450 DX|ScanScope ID = 1",
                "Leica Biosystems",
                "GT450 DX",
                ["Aperio Leica Biosystems GT450 DX"],
            ),
            (
                "Aperio Image Library v12.4.7;Aperio Leica Biosystems GT450 v1.0.1|ScanScope ID = 2",
                "Leica Biosystems",
                "GT450",
                [
                    "Aperio Image Library v12.4.7",
                    "Aperio Leica Biosystems GT450 v1.0.1",
                ],
            ),
            (
                "Aperio Image Library v12.3.1",
                None,
                None,
                ["Aperio Image Library v12.3.1"],
            ),
        ],
    )
    def test_scanner_manufacturer_and_model_and_versions(
        self,
        decoy: Decoy,
        description: str,
        manufacturer: Optional[str],
        model: Optional[str],
        software_versions: Optional[list[str]],
    ) -> None:
        # Arrange
        page = decoy.mock(cls=TiffPage)
        decoy.when(page.description).then_return(description)

        # Act
        metadata = SvsMetadata(page)

        # Assert
        assert metadata.scanner_manufacturer == manufacturer
        assert metadata.scanner_model == model
        assert metadata.scanner_software_versions == software_versions

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
            _ = metadata.mpp

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
            (
                '<Attribute Name="PIM_DP_UFS_BARCODE">TEFCRUwwMDAx</Attribute>',
                "LABEL0001",
            ),
            # Non-Base64 falls back to the raw value.
            (
                '<Attribute Name="PIM_DP_UFS_BARCODE">SR1274-908A</Attribute>',
                "SR1274-908A",
            ),
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


class TestHuronMetadata:
    @pytest.mark.parametrize(
        ["description", "expected_barcode"],
        [
            # The barcode is Base64-encoded; "MTIzNDU2Ny0x" -> "1234567-1".
            ("Resolution = 0.5 um\nBarcode = MTIzNDU2Ny0x", "1234567-1"),
            # Non-Base64 falls back to the raw value.
            ("Resolution = 0.5 um\nBarcode = not base64!", "not base64!"),
            ("Resolution = 0.5 um", None),
        ],
    )
    def test_barcode(
        self, decoy: Decoy, description: str, expected_barcode: Optional[str]
    ) -> None:
        # Arrange
        page = decoy.mock(cls=TiffPage)
        decoy.when(page.description).then_return(description)
        decoy.when(page.tags).then_return(cast(TiffTags, []))

        # Act
        metadata = HuronTiffMetadata(page)

        # Assert
        assert metadata.barcode == expected_barcode

    def test_mpp_and_serial(self, decoy: Decoy) -> None:
        # Arrange
        page = decoy.mock(cls=TiffPage)
        decoy.when(page.description).then_return(
            "Resolution = 0.5 um\nDeviceID = LE176"
        )
        decoy.when(page.tags).then_return(cast(TiffTags, []))

        # Act
        metadata = HuronTiffMetadata(page)

        # Assert
        assert metadata.mpp == 0.5
        assert metadata.scanner_serial_number == "LE176"


class TestMikroscanMetadata:
    def test_fields(self, decoy: Decoy) -> None:
        # Arrange: the Aperio pipe-separated layout with a Mikroscan header (synthetic
        # values; the real test file is private).
        page = decoy.mock(cls=TiffPage)
        decoy.when(page.description).then_return(
            "Mikroscan Image Structure\n100x200 (256x256) JPEG / RGB Q = 30|"
            "AppMag = 40|SL5 SERIAL # = 42|Date = 01/02/03|Time = 04:05:06|"
            "MPP = 0.25"
        )

        # Act
        metadata = MikroscanTiffMetadata(page)

        # Assert
        assert metadata.mpp == 0.25
        assert metadata.magnification == 40.0
        assert metadata.scanner_manufacturer == "Mikroscan"
        assert metadata.scanner_model == "SL5"
        assert metadata.scanner_serial_number == "42"
        assert metadata.acquisition_datetime == datetime(2003, 1, 2, 4, 5, 6)

    def test_missing_serial_and_date(self, decoy: Decoy) -> None:
        # Arrange
        page = decoy.mock(cls=TiffPage)
        decoy.when(page.description).then_return("Mikroscan Image Structure|MPP = 0.5")

        # Act
        metadata = MikroscanTiffMetadata(page)

        # Assert
        assert metadata.scanner_model is None
        assert metadata.scanner_serial_number is None
        assert metadata.acquisition_datetime is None


class TestMoticMetadata:
    def test_fields(self, decoy: Decoy) -> None:
        # Arrange: the Aperio pipe-separated layout with a Motic header (synthetic
        # values; the real test file is private).
        page = decoy.mock(cls=TiffPage)
        decoy.when(page.description).then_return(
            "Motic V1.0.0\n100x200 [0,0 100x200] [512x512] JPEG/RGB Q = 75|"
            "AppMag = 40|MPP = 0.25|BackgroundColor = 16514557|Barcode = 12345"
        )

        # Act
        metadata = MoticTiffMetadata(page)

        # Assert
        assert metadata.mpp == 0.25
        assert metadata.magnification == 40.0
        assert metadata.scanner_manufacturer == "Motic"
        assert metadata.scanner_software_versions == ["V1.0.0"]
        assert metadata.barcode == "12345"

    def test_missing_barcode(self, decoy: Decoy) -> None:
        # Arrange
        page = decoy.mock(cls=TiffPage)
        decoy.when(page.description).then_return("Motic V1.0.0|MPP = 0.5")

        # Act
        metadata = MoticTiffMetadata(page)

        # Assert
        assert metadata.barcode is None
