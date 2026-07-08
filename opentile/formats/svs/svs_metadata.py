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

from datetime import datetime, timedelta, timezone
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
            # as of today, we have assert tz_str.startswith("GMT") == True
            # Example tz_str: "GMT-07:00" or "GMT-0400"
            if not (tz_str.startswith("GMT") and tz_str[3] in ("+", "-")):
                return None
            sign = -1 if tz_str[3] == "-" else 1
            digits = tz_str[4:].replace(":", "")  # "06:00" or "0600" -> "0600"
            hours, minutes = map(int, (digits[:2], digits[2:]))
            return timezone(sign * timedelta(hours=hours, minutes=minutes))

    @staticmethod
    def _extract_date(date_string):
        """
        Extract date from either 'MM/DD/YYYY' or 'MM/DD/YY' format.

        Args:
            date_string: Date string in format 'MM/DD/YYYY' or 'MM/DD/YY'

        Returns:
            datetime object

        Raises:
            ValueError: If date string doesn't match expected formats
        """
        # Try 4-digit year first (MM/DD/YYYY)
        try:
            return datetime.strptime(date_string, "%m/%d/%Y")
        except ValueError:
            pass

        # Try 2-digit year (MM/DD/YY)
        try:
            return datetime.strptime(date_string, "%m/%d/%y")
        except ValueError:
            raise ValueError(  # noqa: B904
                f"Date '{date_string}' doesn't match expected formats (MM/DD/YYYY or MM/DD/YY)"
            )

    @property
    def aquisition_datetime(self) -> Optional[datetime]:
        try:
            date = SvsMetadata._extract_date(self._svs_metadata["Date"])
            time = datetime.strptime(self._svs_metadata["Time"], r"%H:%M:%S")
            tz_info = self._get_timezone()
        except (KeyError, ValueError):
            return None
        return datetime.combine(date, time.time(), tzinfo=tz_info)

    @property
    def mpp(self) -> float:
        return float(self._svs_metadata["MPP"])

    @property
    def properties(self) -> dict[str, Any]:
        return self._svs_metadata

    # @property
    # def image_offset(self) -> Optional[Tuple[float, float]]:
    #     return (float(self._svs_metadata["Left"]), float(self._svs_metadata["Top"]))
