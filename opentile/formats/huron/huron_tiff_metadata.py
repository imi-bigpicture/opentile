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

"""Metadata parser for Huron tiff files."""

import base64
import binascii
from datetime import datetime
from typing import Any, Optional

from tifffile import TiffPage

from opentile.metadata import Metadata


class HuronTiffMetadata(Metadata):
    def __init__(self, page: TiffPage):
        """Metadata read from a Huron tiff base page. The description is a set of
        newline-separated ``Key = Value`` records, e.g.::

            Scan Size = 8.07x7.86 mm
            Image Dimensions = 20170x19657 Pixels
            Resolution = 0.40 um
            Scan Started = 2017:09:29 14:45:38

        Parameters
        ----------
        page: TiffPage
            The base level TiffPage to read metadata from.
        """
        self._fields = self._parse_description(page.description)
        self._manufacturer = self._get_value_from_tiff_tags(page.tags, "Make")
        self._model = self._get_value_from_tiff_tags(page.tags, "Model")
        self._software = self._get_value_from_tiff_tags(page.tags, "Software")

    @staticmethod
    def _parse_description(description: str) -> dict[str, str]:
        """Parse the newline-separated ``Key = Value`` Huron description."""
        fields: dict[str, str] = {}
        for line in description.splitlines():
            key, separator, value = line.partition(" = ")
            if separator:
                fields[key.strip()] = value.strip()
        return fields

    @property
    def mpp(self) -> float:
        """The pixel spacing (um/pixel) from the ``Resolution`` field (isotropic)."""
        # e.g. "Resolution = 0.40 um"
        return float(self._fields["Resolution"].split()[0])

    @property
    def scanner_manufacturer(self) -> Optional[str]:
        return self._clean_string(self._manufacturer)

    @property
    def scanner_model(self) -> Optional[str]:
        return self._clean_string(self._model)

    @property
    def scanner_serial_number(self) -> Optional[str]:
        return self._clean_string(self._fields.get("DeviceID"))

    @property
    def scanner_software_versions(self) -> Optional[list[str]]:
        software = self._clean_string(self._software)
        return [software] if software is not None else None

    @property
    def barcode(self) -> Optional[str]:
        # The barcode is stored Base64-encoded, as for Philips.
        value = self._fields.get("Barcode")
        if not value:
            return None
        try:
            decoded = base64.b64decode(value, validate=True).decode("utf-8")
        except (binascii.Error, ValueError, UnicodeDecodeError):
            return self._clean_string(value)
        return self._clean_string(decoded)

    @property
    def acquisition_datetime(self) -> Optional[datetime]:
        # e.g. "Scan Started = 2017:09:29 14:45:38"
        scan_started = self._fields.get("Scan Started")
        if scan_started is None:
            return None
        try:
            return datetime.strptime(scan_started, "%Y:%m:%d %H:%M:%S")
        except ValueError:
            return None

    @property
    def properties(self) -> dict[str, Any]:
        return dict(self._fields)
