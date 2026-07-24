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

"""Metadata parser for Argos avs files.

Metadata lives in the ``Argos.Scan.Metadata`` XML stored in private TIFF tag 65000.
Microns-per-pixel is derived from the standard resolution tags (openslide does the
same), which are authoritative and simpler than the ``ScanArea`` bounds in the XML.
"""

from datetime import datetime
from functools import cached_property
from typing import Any, Optional

from defusedxml import ElementTree
from tifffile import RESUNIT, TiffPage

from opentile.metadata import Metadata

ARGOS_METADATA_TAG = 65000

# XML elements that carry a single scalar value; everything else (ScanArea, FocusPoints,
# FocusMap) is structured and skipped for the flat property view.
_SCALAR_ELEMENTS = [
    "ObjectiveMagnification",
    "ObjectiveName",
    "AcquisitionDate",
    "AcquisitionTime",
    "ZRange",
    "MinZ",
    "MaxZ",
    "Barcode",
    "Scanner",
]


class ArgosMetadata(Metadata):
    def __init__(self, page: TiffPage):
        self._page = page
        self._values: dict[str, str] = {}
        tag = page.tags.get(ARGOS_METADATA_TAG)
        if tag is not None and isinstance(tag.value, str):
            root = ElementTree.fromstring(tag.value)
            for name in _SCALAR_ELEMENTS:
                element = root.find(name)
                if element is not None and element.text is not None:
                    value = element.text.strip()
                    if value:
                        self._values[name] = value
            # ScanArea is a nested element holding the scanned bounding box in cm;
            # flatten its X1/Y1/X2/Y2 children into dotted keys (e.g. "ScanArea.X1").
            scan_area = root.find("ScanArea")
            if scan_area is not None:
                for corner in scan_area:
                    if corner.text is not None and corner.text.strip():
                        self._values[f"ScanArea.{corner.tag}"] = corner.text.strip()

    @property
    def magnification(self) -> Optional[float]:
        value = self._values.get("ObjectiveMagnification")
        return float(value) if value is not None else None

    @property
    def scanner_model(self) -> Optional[str]:
        return self._clean_string(self._values.get("Scanner"))

    @property
    def barcode(self) -> Optional[str]:
        return self._clean_string(self._values.get("Barcode"))

    @property
    def acquisition_datetime(self) -> Optional[datetime]:
        date = self._values.get("AcquisitionDate")
        time = self._values.get("AcquisitionTime")
        if date is None:
            return None
        try:
            if time is not None:
                return datetime.strptime(f"{date} {time}", "%m.%d.%Y %H:%M:%S")
            return datetime.strptime(date, "%m.%d.%Y")
        except ValueError:
            return None

    @cached_property
    def pixel_spacing(self) -> Optional[tuple[float, float]]:
        """Return (x, y) pixel spacing in mm from the resolution tags, or None if the
        resolution is not expressed in centimeters."""
        if self._page.resolutionunit != RESUNIT.CENTIMETER:
            return None
        x = self._pixels_per_cm("XResolution")
        y = self._pixels_per_cm("YResolution")
        if x is None or y is None:
            return None
        # mm per pixel = 10 mm/cm / (pixels per cm).
        return 10.0 / x, 10.0 / y

    def focal_plane(self, page_index: int) -> float:
        """Focal plane offset (um) from slide center for a z-plane page.

        Stacked avs files store the planes as the Z axis of the base series in ascending
        order, so page ``page_index`` is plane ``MinZ + page_index`` and its offset is
        that times ``ZRange`` (um between planes). Single-plane files have ZRange 0, so
        this is 0.
        """
        return (self._min_z + page_index) * self._z_range

    @property
    def _min_z(self) -> int:
        try:
            return int(self._values.get("MinZ", 0))
        except ValueError:
            return 0

    @property
    def _z_range(self) -> float:
        try:
            return float(self._values.get("ZRange", 0))
        except ValueError:
            return 0.0

    @property
    def properties(self) -> dict[str, Any]:
        # Keys are the raw Argos XML element names (nested ScanArea corners as
        # "ScanArea.X1" etc.), matching how Svs exposes its native metadata keys.
        return dict(self._values)

    def _pixels_per_cm(self, tag_name: str) -> Optional[float]:
        tag = self._page.tags.get(tag_name)
        if tag is None:
            return None
        value = tag.value
        if isinstance(value, tuple):
            numerator, denominator = value
            return numerator / denominator if denominator else None
        return float(value) if value else None
