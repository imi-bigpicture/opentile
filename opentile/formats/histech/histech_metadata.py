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

"""Metadata parser for 3DHistech/TIFF files."""

from datetime import datetime
from typing import Any, Optional

from tifffile import TiffPage

from opentile.metadata import Metadata


class HistechMetadata(Metadata):
    def __init__(self, page: TiffPage):
        self._histech_metadata = self._histech_description_metadata(page.description)

    @property
    def acquisition_datetime(self) -> Optional[datetime]:
        try:
            date = datetime.strptime(self._histech_metadata["Date"], r"%d/%m/%Y")
            time = datetime.strptime(self._histech_metadata["Time"], r"%H:%M:%S")
        except (KeyError, ValueError):
            return None
        return datetime.combine(date, time.time())

    @property
    def mpp(self) -> float:
        return float(self._histech_metadata["MPP"])

    @property
    def properties(self) -> dict[str, Any]:
        return self._histech_metadata

    @staticmethod
    def _histech_description_metadata(description: str, /) -> dict[str, Any]:
        """Return metadata from 3DHistech/TIFF image description."""
        if not description.startswith("\r\n"):
            msg = "invalid 3DHistech image description"
            raise ValueError(msg)
        result: dict[str, Any] = {}
        items = description.split("|")
        result["Header"] = items[0].strip()
        for item in items[1:]:
            try:
                key, value = item.strip().split("=", maxsplit=1)
            except ValueError:
                # skip empty items or those missing '='
                continue
            result[key.strip()] = value.strip()
        return result
