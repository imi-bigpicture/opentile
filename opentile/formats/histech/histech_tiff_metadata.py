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

"""Metadata parser for 3Dhistech tiff files.

The description uses the Aperio pipe-separated ``Header|Key = Value|...`` layout (parsed
by `SvsLikeMetadata`), but with no vendor prefix on the header line, e.g.::

    68608x95232 (256x256) JPEG/RGB Q=80|Date = 29/12/2009|Time = 12:43:52|
    MPP = 0.2325|3dh_PixelSizeX = 0.2325|3dh_PixelSizeY = 0.2325|
    3dh_Filter = Default|3dh_Profile = Current Profile
"""

from datetime import datetime
from typing import Optional

from opentile.metadata import SvsLikeMetadata


class HistechTiffMetadata(SvsLikeMetadata):
    @property
    def acquisition_datetime(self) -> Optional[datetime]:
        # e.g. "Date = 29/12/2009", "Time = 12:43:52". Unlike the Aperio-like formats
        # the date is written day first.
        date = self._fields.get("Date")
        time = self._fields.get("Time")
        if date is None or time is None:
            return None
        try:
            return datetime.strptime(f"{date} {time}", "%d/%m/%Y %H:%M:%S")
        except ValueError:
            return None
