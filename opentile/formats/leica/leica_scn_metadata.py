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

"""Metadata parser for Leica SCN files.

The slide description is XML in the first page's ``ImageDescription`` tag (parsed by
tifffile as ``TiffFile.scn_metadata``). A ``collection`` holds one macro ``image``
(view offset 0,0 spanning the whole collection) and one or more main ``image``s; each
main image is a pyramid whose ``view`` gives its physical size in nanometres, from which
mpp is derived. The barcode is Base64-encoded in the 2010/10/01 namespace.
"""

import base64
import binascii
from datetime import datetime
from typing import Any, Optional
from xml.etree.ElementTree import Element

from defusedxml import ElementTree

from opentile.geometry import Size, SizeMm
from opentile.metadata import Metadata


class LeicaImage:
    """One ``image`` element: its name, a pyramid (view size in nm, base pixel size),
    and whether it is the collection-spanning macro image."""

    def __init__(self, element: Element, collection_size: Size):
        self.name = element.get("name", "Unknown")
        self.element = element
        view = element.find("{*}view")
        pixels = element.find("{*}pixels")
        if view is None or pixels is None:
            raise ValueError(f"SCN image {self.name} missing view or pixels element.")
        self.view_size = Size(int(view.get("sizeX", 0)), int(view.get("sizeY", 0)))
        offset = Size(int(view.get("offsetX", 0)), int(view.get("offsetY", 0)))
        self.pixel_size = Size(int(pixels.get("sizeX", 0)), int(pixels.get("sizeY", 0)))
        # openslide's macro test: offset 0,0 and view matching the collection bounds.
        self.is_macro = offset == Size(0, 0) and self.view_size == collection_size

    @property
    def mpp(self) -> SizeMm:
        """Base level mpp (um/pixel): view size (nm) over pixel size, nm to um."""
        return SizeMm(
            self.view_size.width / self.pixel_size.width / 1000,
            self.view_size.height / self.pixel_size.height / 1000,
        )


class LeicaScnMetadata(Metadata):
    def __init__(self, scn_xml: str):
        root = ElementTree.fromstring(scn_xml)
        collection = root.find("{*}collection")
        if collection is None:
            raise ValueError("SCN XML has no collection element.")
        self._collection = collection
        collection_size = Size(
            int(collection.get("sizeX", 0)), int(collection.get("sizeY", 0))
        )
        self._images = [
            LeicaImage(image, collection_size)
            for image in collection.findall("{*}image")
        ]

    @property
    def images(self) -> list[LeicaImage]:
        return self._images

    @property
    def main_image(self) -> LeicaImage:
        """The largest non-macro image; the pyramid served as levels."""
        mains = [image for image in self._images if not image.is_macro]
        candidates = mains or self._images
        return max(candidates, key=lambda image: image.pixel_size.width)

    @property
    def barcode(self) -> Optional[str]:
        # Base64-encoded in the 2010/10/01 namespace element text.
        value = self._collection.findtext("{*}barcode")
        if not value:
            return None
        try:
            decoded = base64.b64decode(value, validate=True).decode("utf-8")
        except (binascii.Error, ValueError, UnicodeDecodeError):
            return self._clean_string(value)
        return self._clean_string(decoded)

    @property
    def magnification(self) -> Optional[float]:
        objective = self.main_image.element.findtext(".//{*}objective")
        try:
            return float(objective) if objective is not None else None
        except ValueError:
            return None

    @property
    def scanner_manufacturer(self) -> Optional[str]:
        return "Leica"

    @property
    def scanner_model(self) -> Optional[str]:
        device = self.main_image.element.find("{*}device")
        if device is None:
            return None
        # The model attribute joins scanner names with ';'; take the first.
        return self._clean_string((device.get("model") or "").split(";")[0])

    @property
    def scanner_software_versions(self) -> Optional[list[str]]:
        device = self.main_image.element.find("{*}device")
        if device is None or device.get("version") is None:
            return None
        versions = [v.strip() for v in device.get("version", "").split(";")]
        return [v for v in versions if v] or None

    @property
    def acquisition_datetime(self) -> Optional[datetime]:
        creation = self.main_image.element.findtext("{*}creationDate")
        if creation is None:
            return None
        try:
            return datetime.fromisoformat(creation)
        except ValueError:
            return None

    @property
    def properties(self) -> dict[str, Any]:
        properties: dict[str, Any] = {}
        if self.magnification is not None:
            properties["objective"] = self.magnification
        aperture = self.main_image.element.findtext(".//{*}numericalAperture")
        if aperture is not None:
            properties["numerical_aperture"] = aperture
        illumination = self.main_image.element.findtext(".//{*}illuminationSource")
        if illumination is not None:
            properties["illumination_source"] = illumination
        return properties
