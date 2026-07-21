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

import re
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

    @property
    def scanner_manufacturer(self) -> Optional[str]:
        if self.scanner_serial_number is None:
            return None
        return "Leica Biosystems"

    @property
    def scanner_model(self) -> Optional[str]:
        scanner_type = self._svs_metadata.get("ScannerType")
        if scanner_type:
            return scanner_type
        if self.scanner_serial_number is None:
            return None
        header = self._header.splitlines()[0] if self._header else ""
        if "GT450 DX" in header:
            return "GT450 DX"
        # must be after 'GT450 DX':
        if "GT450" in header:
            return "GT450"
        return "Aperio"

    @property
    def scanner_software_versions(self) -> Optional[list[str]]:
        return [
            segment.splitlines()[0].strip()
            for segment in self._header.split(";")
            if segment.strip()
        ]

    @property
    def scanner_serial_number(self) -> Optional[str]:
        return self._svs_metadata.get("ScanScope ID")

    @property
    def acquisition_datetime(self) -> Optional[datetime]:
        try:
            date = SvsMetadata._extract_date(self._svs_metadata["Date"])
            time = datetime.strptime(self._svs_metadata["Time"], r"%H:%M:%S")
            tz_info = self._get_timezone()
        except (KeyError, ValueError):
            return None
        return datetime.combine(date, time.time(), tzinfo=tz_info)

    @property
    def label_text(self) -> Optional[str]:
        return self._clean_string(self._svs_metadata.get("Title"))

    @property
    def barcode(self) -> Optional[str]:
        return self._clean_string(self._svs_metadata.get("Barcode"))

    @property
    def mpp(self) -> float:
        value = self._svs_metadata.get("MPP")
        if value is not None:
            return float(str(value).replace(",", "."))
        match = re.search(r"Scan resolution\s+([0-9.,]+)", self._header)
        if match is not None:
            return float(match.group(1).replace(",", "."))
        raise ValueError("No MPP or scan resolution found in SVS image description")

    @property
    def properties(self) -> dict[str, Any]:
        return self._svs_metadata

    @property
    def _header(self) -> str:
        return self._svs_metadata.get("Header", "")

    def _get_timezone(self) -> Optional[timezone]:
        tz_str = self._svs_metadata.get("Time Zone")
        if tz_str is not None:
            if not (tz_str.startswith("GMT") and tz_str[3] in ("+", "-")):
                return None
            sign = -1 if tz_str[3] == "-" else 1
            digits = tz_str[4:].replace(":", "")  # "06:00" or "0600" -> "0600"
            hours, minutes = map(int, (digits[:2], digits[2:]))
            return timezone(sign * timedelta(hours=hours, minutes=minutes))
        return None

    @staticmethod
    def _extract_date(date_string: str) -> datetime:
        """Extract date from either 'MM/DD/YYYY' or 'MM/DD/YY' format.

        Parameters
        ----------
        date_string : str
            Date string in format 'MM/DD/YYYY' or 'MM/DD/YY'

        Returns
        -------
        datetime
            datetime object

        Raises
        ------
        ValueError
            If date string doesn't match expected formats
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
            raise ValueError(
                f"Date '{date_string}' doesn't match expected formats "
                "(MM/DD/YYYY or MM/DD/YY)"
            ) from None
