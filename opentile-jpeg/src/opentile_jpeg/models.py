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

"""Public data structures used by opentile_jpeg."""

import enum
from dataclasses import dataclass

import opentile_jpeg._binding as lib

Color = int | tuple[int, int, int]
"""A 0..255 gray level or an ``(r, g, b)`` triple of 0..255 values."""


class Metadata(enum.Flag):
    """Metadata an operation can strip from its output; combine with ``|``.

    By default every output keeps all of the source's metadata markers. They are
    copied verbatim, so after a crop or compose the EXIF dimensions and thumbnail
    still describe the source; strip ``EXIF`` where that matters. JFIF and Adobe
    markers are always written by libjpeg-turbo to match the output.
    """

    NONE = 0
    ICC = lib.STRIP_ICC  # ICC color profile (APP2)
    EXIF = lib.STRIP_EXIF  # EXIF, including orientation and thumbnail (APP1)
    XMP = lib.STRIP_XMP  # XMP, standard and extended (APP1)
    COMMENTS = lib.STRIP_COMMENTS  # COM segments
    OTHER = lib.STRIP_OTHER  # every other APPn segment, e.g. IPTC (APP13)
    ALL = ICC | EXIF | XMP | COMMENTS | OTHER


@dataclass(frozen=True)
class JpegHeader:
    """Dimensions and luma sampling factors read from a JPEG's frame header.

    Attributes
    ----------
    width: int
        Image width in pixels.
    height: int
        Image height in pixels.
    sampling: tuple[int, int]
        The luma component's ``(h_samp_factor, v_samp_factor)``, which give the
        MCU grid (``8 * h`` by ``8 * v`` pixels) used for MCU alignment.
    """

    width: int
    height: int
    sampling: tuple[int, int]


@dataclass(frozen=True)
class CoefficientInfo:
    """Structure of a JPEG's DCT coefficients, and what's needed to re-encode.

    Attributes
    ----------
    size: tuple[int, int]
        ``(width, height)`` of the image in pixels.
    sampling: tuple[tuple[int, int], ...]
        Per component, its ``(h_samp_factor, v_samp_factor)``.
    block_shapes: tuple[tuple[int, int], ...]
        Per component, its coefficient array shape ``(height_in_blocks,
        width_in_blocks)`` (the leading two dimensions of the arrays from
        :func:`decode_coefficients`).
    template: bytes
        The original codestream. :func:`encode_coefficients` reuses it to
        reproduce the exact quantization tables, sampling and dimensions, so only
        the coefficient values you changed take effect.
    """

    size: tuple[int, int]
    sampling: tuple[tuple[int, int], ...]
    block_shapes: tuple[tuple[int, int], ...]
    template: bytes
