#    Copyright 2022-2023 SECTRA AB
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

"""Base metadata class."""

from datetime import datetime
from typing import Any, Optional

from tifffile import TiffPage, TiffTags


class Metadata:
    """Class for retrieving metadata from tiff-file. Should be sub-classed
    to read metadata from specific file formats."""

    @property
    def magnification(self) -> Optional[float]:
        """Return the objective magnification if present in file."""
        return None

    @property
    def scanner_manufacturer(self) -> Optional[str]:
        """Return the scanner manufacturer if present in file."""
        return None

    @property
    def scanner_model(self) -> Optional[str]:
        """Return the scanner model if present in file."""
        return None

    @property
    def scanner_software_versions(self) -> Optional[list[str]]:
        """Return the scanner software versions if present in file."""
        return None

    @property
    def scanner_serial_number(self) -> Optional[str]:
        """Return the scanner serial number if present in file."""
        return None

    @property
    def acquisition_datetime(self) -> Optional[datetime]:
        """Return the acquisition datetime if present in file."""
        return None

    @property
    def label_text(self) -> Optional[str]:
        """Return the human-readable slide label text if present in file.

        Corresponds to DICOM Label Text (2200,0002)."""
        return None

    @property
    def barcode(self) -> Optional[str]:
        """Return the slide barcode value if present in file.

        Corresponds to DICOM Barcode Value (2200,0005)."""
        return None

    @property
    def properties(self) -> dict[str, Any]:
        """Return a dictionary of other metadata present in file."""
        return {}

    @staticmethod
    def _clean_string(value: Optional[str]) -> Optional[str]:
        """Normalize a text field: strip surrounding whitespace and null padding,
        returning ``None`` for an empty or absent value."""
        if value is None:
            return None
        stripped = value.strip(" \t\r\n\v\f\x00")
        return stripped or None

    @staticmethod
    def _get_value_from_tiff_tags(
        tiff_tags: TiffTags, value_name: str
    ) -> Optional[str]:
        for tag in tiff_tags:
            if tag.name == value_name:
                return str(tag.value)
        return None


class SvsLikeMetadata(Metadata):
    """Base metadata for Aperio-like formats (e.g. Mikroscan, Motic) that reuse the
    pipe-separated ``Header|Key = Value|...`` description with their own header instead
    of the ``Aperio `` prefix (so tifffile's ``svs_description_metadata``, which needs
    that prefix, cannot be used). Subclasses read the vendor-specific fields; the common
    ``AppMag`` and ``MPP`` fields are parsed here."""

    def __init__(self, page: TiffPage):
        self._header, self._fields = self._parse_description(page.description)

    @staticmethod
    def _parse_description(description: str) -> tuple[str, dict[str, str]]:
        """Parse the description into the header's first line and the ``Key = Value``
        fields (pipe-items without a ``=`` separator, such as trailing padding, are
        ignored)."""
        items = (description or "").split("|")
        header = items[0].splitlines()[0].strip() if items else ""
        fields: dict[str, str] = {}
        for item in items[1:]:
            key, separator, value = item.partition(" = ")
            if separator:
                fields[key.strip()] = value.strip()
        return header, fields

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
    def properties(self) -> dict[str, Any]:
        return dict(self._fields)
