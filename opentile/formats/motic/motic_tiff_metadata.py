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

"""Metadata parser for Motic tiff files.

The description uses the Aperio pipe-separated ``Header|Key = Value|...`` layout (parsed
by `SvsLikeMetadata`), but with a ``Motic <version>`` header instead of the ``Aperio ``
prefix, e.g.::

    Motic V1.0.0
    53046x51735 [0,0 53046x51735] [512x512] JPEG/RGB Q = 75|AppMag = 40|
    MPP = 0.260417|BackgroundColor = 16514557|Barcode = 900444
"""

from typing import Optional

from opentile.metadata import SvsLikeMetadata


class MoticTiffMetadata(SvsLikeMetadata):
    @property
    def scanner_manufacturer(self) -> Optional[str]:
        return "Motic"

    @property
    def scanner_software_versions(self) -> Optional[list[str]]:
        # The header is "Motic <version>", e.g. "Motic V1.0.0".
        version = self._header.removeprefix("Motic").strip()
        return [version] if version else None

    @property
    def barcode(self) -> Optional[str]:
        return self._clean_string(self._fields.get("Barcode"))
