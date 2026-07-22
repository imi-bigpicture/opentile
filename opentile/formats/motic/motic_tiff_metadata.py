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

"""Metadata parser for Motic tiff files."""

from typing import Any, Optional

from tifffile import TiffPage

from opentile.metadata import Metadata


class MoticTiffMetadata(Metadata):
    def __init__(self, page: TiffPage):
        """Metadata read from a Motic tiff base page. The description uses the Aperio
        pipe-separated ``Header|Key = Value`` layout, but with a ``Motic <version>``
        header instead of the ``Aperio `` prefix, e.g.::

            Motic V1.0.0
            53046x51735 [0,0 53046x51735] [512x512] JPEG/RGB Q = 75|AppMag = 40|
            MPP = 0.260417|BackgroundColor = 16514557|Barcode = 900444

        Parameters
        ----------
        page: TiffPage
            The base level TiffPage to read metadata from.
        """
        items = (page.description or "").split("|")
        # The header is the first pipe-item; its first line is "Motic <version>".
        self._header = items[0].splitlines()[0].strip() if items else ""
        self._fields = self._parse_fields(items[1:])

    @staticmethod
    def _parse_fields(items: list[str]) -> dict[str, str]:
        """Parse the ``Key = Value`` pipe-items into a dict (items without ``=``, such
        as the trailing padding, are ignored)."""
        fields: dict[str, str] = {}
        for item in items:
            key, separator, value = item.partition(" = ")
            if separator:
                fields[key.strip()] = value.strip()
        return fields

    @property
    def magnification(self) -> Optional[float]:
        try:
            return float(self._fields["AppMag"])
        except (KeyError, ValueError):
            return None

    @property
    def mpp(self) -> float:
        """The pixel spacing (um/pixel) from the ``MPP`` field (isotropic)."""
        return float(self._fields["MPP"])

    @property
    def scanner_manufacturer(self) -> Optional[str]:
        return "Motic"

    @property
    def scanner_software_versions(self) -> Optional[list[str]]:
        # The header is "Motic <version>", e.g. "Motic V1.0.0".
        version = self._header.removeprefix("Motic").strip()
        return [version] if version else None

    @property
    def barcode(self) -> Optional[str]:
        return self._clean_string(self._fields.get("Barcode"))

    @property
    def properties(self) -> dict[str, Any]:
        return dict(self._fields)
