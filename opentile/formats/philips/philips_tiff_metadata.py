#    Copyright 2021 SECTRA AB
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

from datetime import datetime
from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

from defusedxml import ElementTree
from tifffile.tifffile import TiffFile

from opentile.metadata import Metadata

CastType = TypeVar("CastType", int, float, str)


class PhilipsTiffMetadata(Metadata):
    _scanner_manufacturer: Optional[str] = None
    _scanner_software_versions: Optional[List[str]] = None
    _scanner_serial_number: Optional[str] = None
    _pixel_spacing: Optional[Tuple[float, float]] = None

    TAGS = [
        "DICOM_PIXEL_SPACING",
        "DICOM_ACQUISITION_DATETIME",
        "DICOM_MANUFACTURER",
        "DICOM_SOFTWARE_VERSIONS",
        "DICOM_DEVICE_SERIAL_NUMBER",
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
        self._tags: Dict[str, Optional[str]] = {tag: None for tag in self.TAGS}
        for element in metadata.iter():
            if element.tag == "Attribute" and element.text is not None:
                name = element.attrib["Name"]
                if name in self._tags and self._tags[name] is None:
                    self._tags[name] = element.text

    @property
    def scanner_manufacturer(self) -> Optional[str]:
        return self._tags["DICOM_MANUFACTURER"]

    @cached_property
    def scanner_software_versions(self) -> Optional[List[str]]:
        print(self._tags["DICOM_SOFTWARE_VERSIONS"])
        if self._tags["DICOM_SOFTWARE_VERSIONS"] is None:
            return None
        return self._split_and_cast_text(self._tags["DICOM_SOFTWARE_VERSIONS"], str)

    @property
    def scanner_serial_number(self) -> Optional[str]:
        return self._tags["DICOM_DEVICE_SERIAL_NUMBER"]

    @cached_property
    def aquisition_datetime(self) -> Optional[datetime]:
        if self._tags["DICOM_ACQUISITION_DATETIME"] is None:
            return None
        try:
            return datetime.strptime(
                self._tags["DICOM_ACQUISITION_DATETIME"], r"%Y%m%d%H%M%S.%f"
            )
        except ValueError:
            return None

    @cached_property
    def pixel_spacing(self) -> Optional[Tuple[float, float]]:
        if self._tags["DICOM_PIXEL_SPACING"] is None:
            return None
        return tuple(
            self._split_and_cast_text(self._tags["DICOM_PIXEL_SPACING"], float)[0:2]
        )

    @cached_property
    def properties(self) -> Dict[str, Any]:
        properties: Dict[str, Any] = {}
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
    def _split_and_cast_text(string: str, cast_type: Type[CastType]) -> List[CastType]:
        return [cast_type(element) for element in string.replace('"', "").split()]
