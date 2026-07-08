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

"""Helpers for inspecting JPEG 2000 codestreams."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Jpeg2000Info:
    """Properties parsed from a JPEG 2000 codestream."""

    reversible: bool
    """True if the reversible 5/3 wavelet (lossless) is used, False for the
    irreversible 9/7 wavelet (lossy)."""
    uses_mct: bool
    """True if the multiple component transform is applied."""
    components: int
    """Number of components."""
    subsampling: Optional[tuple[int, int]]
    """Maximum (horizontal, vertical) component subsampling factors, or None if
    there is only a single component. (1, 1) means no subsampling."""
    bit_depth: int
    """Bit depth of the first component."""
    extended: bool
    """True if the SIZ Rsiz capabilities field is non-zero, i.e. the codestream
    is not the unrestricted ISO/IEC 15444-1 (Part 1) profile (Rsiz 0). Both
    restricted Part-1 profiles and Part-2 extensions are flagged."""


class Jpeg2000:
    """Parsing of JPEG 2000 codestream markers."""

    SIZ = b"\xff\x51"
    """Image and tile size marker. Holds the image/tile dimensions and the
    per-component bit depth and subsampling."""
    COD = b"\xff\x52"
    """Coding style default marker. Holds the default coding parameters,
    including the wavelet transformation (reversible 5/3 or irreversible 9/7)
    and whether the multiple component transform is used."""
    _SEGMENT_LENGTH_BYTES = 2
    """Width of a marker segment length field (Lxxx), always 2 bytes."""
    _SIZ_RSIZ_OFFSET = 4
    """Offset of the capabilities field (Rsiz) from the start of the SIZ marker:
    marker[2] Lsiz[2] = 4."""
    _SIZ_CSIZ_OFFSET = 38
    """Offset of the component count (Csiz) from the start of the SIZ marker:
    marker[2] Lsiz[2] Rsiz[2] Xsiz[4] Ysiz[4] XOsiz[4] YOsiz[4] XTsiz[4]
    YTsiz[4] XTOsiz[4] YTOsiz[4] = 38."""
    _SIZ_COMPONENT_OFFSET = 40
    """Offset of the first per-component triplet (Ssiz, XRsiz, YRsiz) from the
    start of the SIZ marker, i.e. just after Csiz."""
    _COD_MCT_OFFSET = 8
    """Offset of the multiple-component-transform byte from the start of the COD
    marker: marker[2] Lcod[2] Scod[1] progression[1] layers[2] = 8."""
    _COD_TRANSFORMATION_OFFSET = 13
    """Offset of the transformation (wavelet) byte from the start of the COD
    marker: the MCT byte at 8, then levels[1] code-block-width[1]
    code-block-height[1] style[1] = 13."""

    @classmethod
    def parse(cls, codestream: bytes) -> Optional[Jpeg2000Info]:
        """Parse a JPEG 2000 codestream into a `Jpeg2000Info`, or None if the
        SIZ or COD marker segments can not be located or are truncated."""
        siz_offset = codestream.find(cls.SIZ)
        cod_offset = cls._find_cod(codestream, siz_offset)
        if siz_offset == -1 or cod_offset == -1:
            return None

        siz = cls._parse_siz(codestream, siz_offset)
        cod = cls._parse_cod(codestream, cod_offset)
        if siz is None or cod is None:
            return None
        components, subsampling, bit_depth, extended = siz
        reversible, uses_mct = cod
        return Jpeg2000Info(
            reversible=reversible,
            uses_mct=uses_mct,
            components=components,
            subsampling=subsampling,
            bit_depth=bit_depth,
            extended=extended,
        )

    @classmethod
    def reversible(cls, codestream: bytes) -> Optional[bool]:
        """Return whether a JPEG 2000 codestream uses the reversible wavelet
        (lossless), or None if it can not be determined. Convenience wrapper
        around `parse`."""
        info = cls.parse(codestream)
        return None if info is None else info.reversible

    @classmethod
    def _parse_siz(
        cls, codestream: bytes, siz_offset: int
    ) -> Optional[tuple[int, Optional[tuple[int, int]], int, bool]]:
        """Return (components, subsampling, bit_depth, extended) from SIZ."""
        csiz_offset = siz_offset + cls._SIZ_CSIZ_OFFSET
        if len(codestream) < csiz_offset + cls._SEGMENT_LENGTH_BYTES:
            return None
        rsiz_offset = siz_offset + cls._SIZ_RSIZ_OFFSET
        extended = (
            int.from_bytes(
                codestream[rsiz_offset : rsiz_offset + cls._SEGMENT_LENGTH_BYTES], "big"
            )
            != 0
        )
        components = int.from_bytes(
            codestream[csiz_offset : csiz_offset + cls._SEGMENT_LENGTH_BYTES], "big"
        )
        component_offset = siz_offset + cls._SIZ_COMPONENT_OFFSET
        # Each component is a (Ssiz, XRsiz, YRsiz) triplet.
        if len(codestream) < component_offset + components * 3:
            return None
        # Ssiz low 7 bits are the bit depth minus one.
        bit_depth = (codestream[component_offset] & 0x7F) + 1
        factors = [
            (
                codestream[component_offset + i * 3 + 1],
                codestream[component_offset + i * 3 + 2],
            )
            for i in range(components)
        ]
        subsampling = None
        if components > 1:
            subsampling = (
                max(horizontal for horizontal, _ in factors),
                max(vertical for _, vertical in factors),
            )
        return components, subsampling, bit_depth, extended

    @classmethod
    def _parse_cod(
        cls, codestream: bytes, cod_offset: int
    ) -> Optional[tuple[bool, bool]]:
        """Return (reversible, uses_mct) from the COD segment."""
        if len(codestream) <= cod_offset + cls._COD_TRANSFORMATION_OFFSET:
            return None
        uses_mct = codestream[cod_offset + cls._COD_MCT_OFFSET] != 0
        reversible = codestream[cod_offset + cls._COD_TRANSFORMATION_OFFSET] == 1
        return reversible, uses_mct

    @classmethod
    def _find_cod(cls, codestream: bytes, siz_offset: int) -> int:
        """Return the offset of the COD marker, or -1 if not found. Searched
        after the SIZ segment so that a stray COD-like byte sequence inside the
        SIZ data is not mistaken for it."""
        start = 0
        if siz_offset != -1:
            # Lsiz, the length field after the marker, is the segment length. It
            # counts itself and the parameters but not the marker, so the next
            # marker is at marker offset + marker length + Lsiz.
            length_start = siz_offset + len(cls.SIZ)
            length_end = length_start + cls._SEGMENT_LENGTH_BYTES
            siz_length = int.from_bytes(codestream[length_start:length_end], "big")
            start = length_start + siz_length
        return codestream.find(cls.COD, start)
