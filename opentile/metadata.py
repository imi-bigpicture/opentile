#    Copyright 2022-2023 SECTRA AB
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

"""Base metadata class."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from tifffile import TiffTags


class Metadata:
    """Class for retrieving metadata from tiff-file. Should be sub-classed
    to read metadata from specific file formats."""

    @property
    def magnification(self) -> Optional[float]:
        """Return the objective magnification if present in file."""
        return None

    @property
    def scanner_manufacturer(self) -> Optional[str]:
        """Return the scanner manufacturer if present in file."""
        return None

    @property
    def scanner_model(self) -> Optional[str]:
        """Return the scanner model if present in file."""
        return None

    @property
    def scanner_software_versions(self) -> Optional[List[str]]:
        """Return the scanner software versions if present in file."""
        return None

    @property
    def scanner_serial_number(self) -> Optional[str]:
        """Return the scanner serial number if present in file."""
        return None

    @property
    def aquisition_datetime(self) -> Optional[datetime]:
        """Return the acquisition datetime if present in file."""
        return None

    @property
    def properties(self) -> Dict[str, Any]:
        """Return a dictionary of other metadata present in file."""
        return {}

    @staticmethod
    def _get_value_from_tiff_tags(
        tiff_tags: TiffTags, value_name: str
    ) -> Optional[str]:
        for tag in tiff_tags:
            if tag.name == value_name:
                return str(tag.value)
        return None
