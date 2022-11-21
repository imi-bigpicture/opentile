from datetime import datetime
from typing import Any, Dict, List, Optional

from tifffile.tifffile import TiffTags


class Metadata:
    @property
    def magnification(self) -> Optional[float]:
        """Returns the objective magnification if present in file."""
        return None

    @property
    def scanner_manufacturer(self) -> Optional[str]:
        """Returns the scanner manufacturer if present in file."""
        return None

    @property
    def scanner_model(self) -> Optional[str]:
        """Returns the scanner model if present in file."""
        return None

    @property
    def scanner_software_versions(self) -> Optional[List[str]]:
        """Returns the scanner software versions if present in file."""
        return None

    @property
    def scanner_serial_number(self) -> Optional[str]:
        """Returns the scanner serial number if present in file."""
        return None

    @property
    def aquisition_datetime(self) -> Optional[datetime]:
        """Returns the aquisition datetime if present in file."""
        return None

    @property
    def properties(self) -> Dict[str, Any]:
        """Returns a dictionary of other metadata present in file."""
        return {}

    @staticmethod
    def _get_value_from_tiff_tags(
        tiff_tags: TiffTags,
        value_name: str
    ) -> Optional[str]:
        for tag in tiff_tags:
            if tag.name == value_name:
                return str(tag.value)
        return None
