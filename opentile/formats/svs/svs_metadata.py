#    Copyright 2021-2023 SECTRA AB
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

"""Metadata parser for svs files."""

from datetime import datetime, timezone, timedelta
from typing import Any, Optional

from tifffile import TiffPage
from tifffile.tifffile import svs_description_metadata

from opentile.metadata import Metadata


class SvsMetadata(Metadata):
    def __init__(self, page: TiffPage):
        self._svs_metadata = svs_description_metadata(page.description)

    @property
    def magnification(self) -> Optional[float]:
        try:
            return float(self._svs_metadata["AppMag"])
        except (KeyError, ValueError):
            return None

    def _get_timezone(self) -> Optional[timezone]:
        if "Time Zone" in self._svs_metadata:
          tz_str = self._svs_metadata["Time Zone"]
          assert tz_str.startswith("GMT")
          # Example tz_str: "GMT-07:00"
          if tz_str.startswith("GMT"):
            offset_str = tz_str[3:]  # "-07:00"
            sign = 1 if offset_str[0] == "+" else -1
            hours, minutes = map(int, offset_str[1:].split(":"))
            offset = timedelta(hours=hours, minutes=minutes) * sign
            return timezone(offset)
        return None

    @property
    def aquisition_datetime(self) -> Optional[datetime]:
        try:
            date = datetime.strptime(self._svs_metadata["Date"], r"%m/%d/%y")
            time = datetime.strptime(self._svs_metadata["Time"], r"%H:%M:%S")
            timezone = self._get_timezone()
        except (KeyError, ValueError):
            return None
        return datetime.combine(date, time.time(), tzinfo=timezone)

    @property
    def mpp(self) -> float:
        return float(self._svs_metadata["MPP"])

    @property
    def properties(self) -> dict[str, Any]:
        return self._svs_metadata

    # @property
    # def image_offset(self) -> Optional[Tuple[float, float]]:
    #     return (float(self._svs_metadata["Left"]), float(self._svs_metadata["Top"]))
