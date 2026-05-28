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
from typing import Any, Optional, Union

from tifffile import TiffPage

from opentile.metadata import Metadata


class NdpiMetadata(Metadata):
    def __init__(self, page: TiffPage):
        self._tags = page.tags
        if page.ndpi_tags is not None:
            self._ndpi_tags = page.ndpi_tags
        else:
            self._ndpi_tags = {}

    @property
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
    def scanner_software_versions(self) -> Optional[list[str]]:
        software_version = self._ndpi_tags.get("Software")
        if software_version is not None:
            return [software_version]
        return None

    @property
    def scanner_serial_number(self) -> Optional[str]:
        return self._ndpi_tags.get("ScannerSerialNumber")

    @property
    def aquisition_datetime(self) -> Optional[datetime]:
        datetime_tag = self._tags.get("DateTime")
        if datetime_tag is None:
            return None
        try:
            return datetime.strptime(datetime_tag.value, "%Y:%m:%d %H:%M:%S")
        except ValueError:
            return None

    @property
    def properties(self) -> dict[str, Any]:
        properties: dict[str, Any] = {
            "x_offset_from_slide_center": self._ndpi_tags.get("XOffsetFromSlideCenter"),
            "y_offset_from_slide_center": self._ndpi_tags.get("YOffsetFromSlideCenter"),
            "z_offset_from_slide_center": self._ndpi_tags.get("ZOffsetFromSlideCenter"),
        }
        comments = self._ndpi_tags.get("Comments")
        if isinstance(comments, str):
            properties.update(self._parse_comments(comments))
        return properties

    @staticmethod
    def _parse_comments(comments: str) -> dict[str, Union[str, dict[str, str]]]:
        """Parse the NDPI ``Comments`` stream (TIFF tag 65449).

        The stream is a Hamamatsu-specific text block of ``Key=Value`` records
        terminated by ``\\r\\n`` or bare ``\\r``. Records prefixed with ``;``
        belong to a named section: a record with ``;`` and no ``=`` opens the
        section, and subsequent ``;Key=Value`` records are its entries.
        Unprefixed records are global and become top-level string values;
        sections become nested ``dict[str, str]`` values keyed by section name.
        """
        result: dict[str, Union[str, dict[str, str]]] = {}
        current: Optional[dict[str, str]] = None

        for raw_line in comments.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(";"):
                line = line[1:]
            if "=" in line:
                key, _, value = line.partition("=")
                if current is None:
                    result[key.strip()] = value.strip()
                else:
                    current[key.strip()] = value.strip()
            else:
                current = {}
                result[line.strip()] = current
        return result
