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

"""Lossless JPEG operations in the DCT-coefficient domain.

All operations copy or set DCT coefficients and re-encode the entropy stream
only, so pixels are never decoded and re-quantized: the result is lossless.

Regions and tile geometry must be aligned to the JPEG's MCU grid, which is
``8 * Hmax`` pixels wide by ``8 * Vmax`` tall, where Hmax/Vmax are the maximum
component sampling factors (the luma factors returned by :func:`header`). Align
x to the MCU width and y to the MCU height:

    subsampling   luma HxV   MCU (WxH)
    4:4:4         1x1        8 x 8
    4:2:2         2x1        16 x 8
    4:4:0         1x2        8 x 16
    4:2:0         2x2        16 x 16
    4:1:1         4x1        32 x 8

The implementation lives in :mod:`opentile_jpeg.core` over the nanobind extension
:mod:`opentile_jpeg._binding`; this module only re-exports the public API.
"""

from importlib.metadata import version

from opentile_jpeg._binding import LIBJPEG_TURBO_VERSION
from opentile_jpeg.core import (
    compose,
    crop,
    crop_multiple,
    decode_coefficients,
    encode_coefficients,
    fill,
    header,
    optimize,
)
from opentile_jpeg.errors import JpegError, JpegWarning
from opentile_jpeg.models import CoefficientInfo, Color, JpegHeader, Metadata

__version__ = version("opentile-jpeg")

libjpeg_turbo_version: str = LIBJPEG_TURBO_VERSION
"""The libjpeg-turbo version statically linked into this build, e.g. ``"3.2.0"``.

Dependency scanners cannot see a statically linked library, so check this one
against libjpeg-turbo security advisories.
"""

__all__ = [
    "CoefficientInfo",
    "Color",
    "JpegError",
    "JpegWarning",
    "JpegHeader",
    "Metadata",
    "compose",
    "crop",
    "crop_multiple",
    "decode_coefficients",
    "encode_coefficients",
    "fill",
    "header",
    "libjpeg_turbo_version",
    "optimize",
]
