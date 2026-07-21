#    Copyright 2021-2023 SECTRA AB
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

"""Metadata parser for Philips tiff files."""

import base64
import binascii
from datetime import datetime
from functools import cached_property
from typing import Any, Optional, TypeVar

from defusedxml import ElementTree
from tifffile import TiffFile

from opentile.metadata import Metadata

CastType = TypeVar("CastType", int, float, str)


class PhilipsTiffMetadata(Metadata):
    _scanner_manufacturer: Optional[str] = None
    _scanner_software_versions: Optional[list[str]] = None
    _scanner_serial_number: Optional[str] = None
    _pixel_spacing: Optional[tuple[float, float]] = None

    TAGS = [
        "DICOM_PIXEL_SPACING",
        "DICOM_ACQUISITION_DATETIME",
        "DICOM_MANUFACTURER",
        "DICOM_SOFTWARE_VERSIONS",
        "DICOM_DEVICE_SERIAL_NUMBER",
        "PIM_DP_UFS_BARCODE",
        "DICOM_LOSSY_IMAGE_COMPRESSION_METHOD",
        "DICOM_LOSSY_IMAGE_COMPRESSION_RATIO",
        "DICOM_BITS_ALLOCATED",
        "DICOM_BITS_STORED",
        "DICOM_HIGH_BIT",
        "DICOM_PIXEL_REPRESENTATION",
    ]

    def __init__(self, tiff_file: TiffFile):
        if tiff_file.philips_metadata is None:
            return

        metadata = ElementTree.fromstring(tiff_file.philips_metadata)
        self._tags: dict[str, Optional[str]] = {tag: None for tag in self.TAGS}
        for element in metadata.iter("Attribute"):
            if element.text is None:
                continue
            name = element.attrib["Name"]
            if name in self._tags and self._tags[name] is None:
                self._tags[name] = element.text

    @property
    def scanner_manufacturer(self) -> Optional[str]:
        return self._tags["DICOM_MANUFACTURER"]

    @property
    def scanner_software_versions(self) -> Optional[list[str]]:
        if self._tags["DICOM_SOFTWARE_VERSIONS"] is None:
            return None
        return self._split_and_cast_text(self._tags["DICOM_SOFTWARE_VERSIONS"], str)

    @property
    def scanner_serial_number(self) -> Optional[str]:
        return self._tags["DICOM_DEVICE_SERIAL_NUMBER"]

    @property
    def barcode(self) -> Optional[str]:
        # Philips stores the barcode Base64-encoded, unlike the plain-text DICOM
        # attributes.
        value = self._tags["PIM_DP_UFS_BARCODE"]
        if not value:
            return None
        try:
            decoded = base64.b64decode(value, validate=True).decode("utf-8")
        except (binascii.Error, ValueError, UnicodeDecodeError):
            return self._clean_string(value)
        return self._clean_string(decoded)

    @property
    def acquisition_datetime(self) -> Optional[datetime]:
        if self._tags["DICOM_ACQUISITION_DATETIME"] is None:
            return None
        try:
            return datetime.strptime(
                self._tags["DICOM_ACQUISITION_DATETIME"], r"%Y%m%d%H%M%S.%f"
            )
        except ValueError:
            return None

    @cached_property
    def pixel_spacing(self) -> Optional[tuple[float, float]]:
        if self._tags["DICOM_PIXEL_SPACING"] is None:
            return None
        values = self._split_and_cast_text(self._tags["DICOM_PIXEL_SPACING"], float)
        return values[1], values[0]

    @property
    def properties(self) -> dict[str, Any]:
        properties: dict[str, Any] = {}
        if self._tags["DICOM_LOSSY_IMAGE_COMPRESSION_METHOD"] is not None:
            properties["lossy_image_compression_method"] = self._split_and_cast_text(
                self._tags["DICOM_LOSSY_IMAGE_COMPRESSION_METHOD"], str
            )
        if self._tags["DICOM_LOSSY_IMAGE_COMPRESSION_RATIO"] is not None:
            properties["lossy_image_compression_ratio"] = self._split_and_cast_text(
                self._tags["DICOM_LOSSY_IMAGE_COMPRESSION_RATIO"], float
            )[0]
        if self._tags["DICOM_BITS_ALLOCATED"] is not None:
            properties["bits_allocated"] = int(self._tags["DICOM_BITS_ALLOCATED"])
        if self._tags["DICOM_BITS_STORED"] is not None:
            properties["bits_stored"] = int(self._tags["DICOM_BITS_STORED"])
        if self._tags["DICOM_HIGH_BIT"] is not None:
            properties["high_bit"] = int(self._tags["DICOM_HIGH_BIT"])

        if self._tags["DICOM_PIXEL_REPRESENTATION"] is not None:
            properties["pixel_representation"] = self._tags[
                "DICOM_PIXEL_REPRESENTATION"
            ]
        return properties

    @staticmethod
    def _split_and_cast_text(string: str, cast_type: type[CastType]) -> list[CastType]:
        return [cast_type(element) for element in string.replace('"', "").split()]
