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

"""Metadata parser for Mikroscan tiff files.

The description uses the Aperio pipe-separated ``Header|Key = Value|...`` layout (parsed
by `SvsLikeMetadata`), e.g.::

    Mikroscan Image Structure
    26880x42240 [0, 0 26880x42240] (256x256) JPEG / RGB Q = 30|AppMag = 20|
    SL5 SERIAL # = 1053|Date = 09/26/19|Time = 09:57:06|MPP = 0.453023|...
"""

from datetime import datetime
from typing import Optional

from opentile.metadata import SvsLikeMetadata


class MikroscanTiffMetadata(SvsLikeMetadata):
    @property
    def _serial_field(self) -> tuple[Optional[str], Optional[str]]:
        """The ``<model> SERIAL # = <serial>`` field as (model, serial), or
        (None, None) if absent."""
        suffix = "SERIAL #"
        for key, value in self._fields.items():
            if key.endswith(suffix):
                model = key[: -len(suffix)].strip()
                return (model or None, value)
        return (None, None)

    @property
    def scanner_manufacturer(self) -> Optional[str]:
        return "Mikroscan"

    @property
    def scanner_model(self) -> Optional[str]:
        # e.g. the "SL5" in "SL5 SERIAL # = 1053".
        return self._serial_field[0]

    @property
    def scanner_serial_number(self) -> Optional[str]:
        return self._clean_string(self._serial_field[1])

    @property
    def acquisition_datetime(self) -> Optional[datetime]:
        # e.g. "Date = 09/26/19", "Time = 09:57:06".
        date = self._fields.get("Date")
        time = self._fields.get("Time")
        if date is None or time is None:
            return None
        try:
            return datetime.strptime(f"{date} {time}", "%m/%d/%y %H:%M:%S")
        except ValueError:
            return None
