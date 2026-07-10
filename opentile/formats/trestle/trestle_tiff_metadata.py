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

"""Metadata parser for Trestle tiff files."""

from functools import cached_property
from typing import Any, Optional

from tifffile import TiffPage

from opentile.geometry import SizeMm
from opentile.metadata import Metadata


class TrestleMetadata(Metadata):
    def __init__(self, page: TiffPage):
        self._properties = self._parse_description(page.description)
        self._mpp = SizeMm(
            self._rational(page, "XResolution"),
            self._rational(page, "YResolution"),
        )

    @property
    def magnification(self) -> Optional[float]:
        try:
            return float(self._properties["Objective Power"])
        except (KeyError, ValueError):
            return None

    @property
    def mpp(self) -> SizeMm:
        """Base level mpp (um/pixel). Trestle stores um/pixel directly in the
        X/YResolution tags (non-standard, no ResolutionUnit)."""
        return self._mpp

    @property
    def properties(self) -> dict[str, Any]:
        return self._properties

    def level_overlap(self, level: int) -> tuple[int, int]:
        """The (x, y) tile overlap for a level from the OverlapsXY field; (0, 0) for
        levels beyond those listed."""
        overlaps = self._overlaps
        return (
            overlaps[2 * level] if 2 * level < len(overlaps) else 0,
            overlaps[2 * level + 1] if 2 * level + 1 < len(overlaps) else 0,
        )

    @cached_property
    def _overlaps(self) -> list[int]:
        """The raw OverlapsXY values (x0, y0, x1, y1, ...)."""
        overlaps_string = self._properties.get("OverlapsXY", "")
        return [int(value) for value in overlaps_string.split()]

    @staticmethod
    def _rational(page: TiffPage, tag_name: str) -> float:
        numerator, denominator = page.tags[tag_name].value
        return numerator / denominator

    @staticmethod
    def _parse_description(description: str) -> dict[str, str]:
        """Parse a Trestle ImageDescription of semicolon-separated key=value pairs."""
        items: dict[str, str] = {}
        for part in description.split(";"):
            key, sep, value = part.partition("=")
            if sep:
                items[key.strip()] = value.strip()
        return items
