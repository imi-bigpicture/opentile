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

"""Metadata parser for PerkinElmer/Akoya qptiff files.

Every IFD carries a ``PerkinElmer-QPI-ImageDescription`` XML document in its
ImageDescription tag, describing that image rather than the file as a whole. The
``ImageType`` element in it classifies the IFD (see `image_type`), and the baseline IFD
additionally carries the file-level fields read here.

Microns-per-pixel comes from the standard resolution tags, which the format defines as
pixels/cm when the true resolution is known (and as 96 pixels/inch when it is not).

Format described in "PerkinElmer image format" (Peter Miller, 2015), distributed under
CC BY 4.0 alongside the sample files at
https://downloads.openmicroscopy.org/images/Vectra-QPTIFF/perkinelmer/.
"""

import re
from datetime import datetime
from typing import Any, Optional

from defusedxml import ElementTree
from tifffile import RESUNIT, TiffPage

from opentile.geometry import SizeMm
from opentile.metadata import Metadata

_IMAGE_TYPE_PATTERN = re.compile(r"<ImageType>(.*?)</ImageType>")
_NAME_PATTERN = re.compile(r"<Name>(.*?)</Name>")

# Description elements holding a single scalar value. ScanProfile is skipped: it is a
# large nested document that the format spec declares "opaque to most readers".
_SCALAR_ELEMENTS = (
    "DescriptionVersion",
    "AcquisitionSoftware",
    "ImageType",
    "Identifier",
    "SlideID",
    "Barcode",
    "ComputerName",
    "IsUnmixedComponent",
    "ExposureTime",
    "SignalUnits",
    "Name",
    "Color",
    "Objective",
    "CameraName",
)


def image_type(page: TiffPage) -> Optional[str]:
    """Return the ImageType of a page, or None if it has no ImageType element.

    One of ``FullResolution``, ``ReducedResolution``, ``Thumbnail``, ``Overview``, or
    ``Label``. Matched with a regex rather than by parsing, as the baseline page's
    description embeds the large ScanProfile document.
    """
    match = _IMAGE_TYPE_PATTERN.search(page.description)
    if match is None:
        return None
    return match.group(1).strip() or None


def band_name(page: TiffPage) -> Optional[str]:
    """Return the band (component) name of a page, e.g. "DAPI".

    Only present for fluorescence and unmixed multispectral images, where each band is
    stored as its own directory. Not present for RGB (brightfield) images.
    """
    match = _NAME_PATTERN.search(page.description)
    if match is None:
        return None
    return match.group(1).strip() or None


class QptiffMetadata(Metadata):
    def __init__(self, page: TiffPage):
        self._values: dict[str, str] = {}
        root = ElementTree.fromstring(page.description)
        for name in _SCALAR_ELEMENTS:
            element = root.find(name)
            if element is not None and element.text is not None:
                value = element.text.strip()
                if value:
                    self._values[name] = value
        self._datetime = self._get_value_from_tiff_tags(page.tags, "DateTime")
        self._mpp = self._get_mpp(page)

    @property
    def magnification(self) -> Optional[float]:
        """Objective magnification, parsed from the objective name (e.g. "20x")."""
        objective = self._values.get("Objective")
        if objective is None:
            return None
        try:
            return float(objective.lower().rstrip("x"))
        except ValueError:
            return None

    @property
    def scanner_manufacturer(self) -> Optional[str]:
        # Files are identified by a Software tag starting with "PerkinElmer-QPI".
        return "PerkinElmer"

    @property
    def scanner_model(self) -> Optional[str]:
        return self._acquisition_software[0]

    @property
    def scanner_software_versions(self) -> Optional[list[str]]:
        version = self._acquisition_software[1]
        return [version] if version is not None else None

    @property
    def barcode(self) -> Optional[str]:
        return self._clean_string(self._values.get("Barcode"))

    @property
    def acquisition_datetime(self) -> Optional[datetime]:
        if self._datetime is None:
            return None
        try:
            return datetime.strptime(self._datetime, "%Y:%m:%d %H:%M:%S")
        except ValueError:
            return None

    @property
    def mpp(self) -> Optional[SizeMm]:
        """Base level mpp (um/pixel), or None if the file records no true resolution."""
        return self._mpp

    @property
    def properties(self) -> dict[str, Any]:
        # Keys are the raw ImageDescription element names.
        return dict(self._values)

    @property
    def _acquisition_software(self) -> tuple[Optional[str], Optional[str]]:
        """The AcquisitionSoftware value split into (name, version), e.g.
        "VectraPolaris 1.0" into ("VectraPolaris", "1.0"). The name is the scanner
        model. If there is no trailing version the whole value is the name."""
        software = self._clean_string(self._values.get("AcquisitionSoftware"))
        if software is None:
            return None, None
        name, _, version = software.rpartition(" ")
        if not name or not version[:1].isdigit():
            return software, None
        return name, version

    @staticmethod
    def _get_mpp(page: TiffPage) -> Optional[SizeMm]:
        """Return (x, y) mpp (um/pixel) from the resolution tags, or None if the
        resolution is not expressed in centimeters (i.e. is not known)."""
        if page.resolutionunit != RESUNIT.CENTIMETER:
            return None
        resolutions = []
        for tag_name in ("XResolution", "YResolution"):
            tag = page.tags.get(tag_name)
            if tag is None:
                return None
            numerator, denominator = tag.value
            if not numerator or not denominator:
                return None
            resolutions.append(numerator / denominator)
        # um per pixel = 10 000 um/cm / (pixels per cm).
        return SizeMm(1e4 / resolutions[0], 1e4 / resolutions[1])
