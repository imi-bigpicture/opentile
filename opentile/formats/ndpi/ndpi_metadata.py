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

"""Metadata parser for ndpi files."""

from datetime import datetime
from functools import cached_property
from typing import Any, Dict, List, Optional

from tifffile.tifffile import TiffPage

from opentile.metadata import Metadata


def get_value_from_ndpi_comments(
    comments: str, value_name: str, value_type: Any
) -> Any:
    """Read value from ndpi comment string."""
    for line in comments.split("\n"):
        if value_name in line:
            value_string = line.split("=")[1]
            return value_type(value_string)


class NdpiMetadata(Metadata):
    def __init__(self, page: TiffPage):
        self._tags = page.tags
        if page.ndpi_tags is not None:
            self._ndpi_tags = page.ndpi_tags
        else:
            self._ndpi_tags = {}

    @cached_property
    def magnification(self) -> Optional[float]:
        try:
            return float(self._ndpi_tags["Magnification"])
        except (AttributeError, ValueError):
            return None

    @property
    def scanner_manufacturer(self) -> Optional[str]:
        return self._ndpi_tags.get("Make")

    @property
    def scanner_model(self) -> Optional[str]:
        return self._ndpi_tags.get("Model")

    @property
    def scanner_software_versions(self) -> Optional[List[str]]:
        software_version = self._ndpi_tags.get("Software")
        if software_version is not None:
            return [software_version]
        return None

    @property
    def scanner_serial_number(self) -> Optional[str]:
        return self._ndpi_tags.get("ScannerSerialNumber")

    @cached_property
    def aquisition_datetime(self) -> Optional[datetime]:
        datetime_tag = self._tags.get("DateTime")
        if datetime_tag is None:
            return None
        try:
            return datetime.strptime(datetime_tag.value, "%Y:%m:%d %H:%M:%S")
        except ValueError:
            return None

    @cached_property
    def properties(self) -> Dict[str, Any]:
        x_offset_from_slide_center = self._ndpi_tags.get("XOffsetFromSlideCenter")
        y_offset_from_slide_center = self._ndpi_tags.get("YOffsetFromSlideCenter")
        z_offset_from_slide_center = self._ndpi_tags.get("ZXOffsetFromSlideCenter")
        return {
            "x_offset_from_slide_center": x_offset_from_slide_center,
            "y_offset_from_slide_center": y_offset_from_slide_center,
            "z_offset_from_slide_center": z_offset_from_slide_center,
        }
