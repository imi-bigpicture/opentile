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

"""Lossless jpeg handling."""

import os
import platform
from collections.abc import Iterator, Sequence
from ctypes.util import find_library
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from struct import pack, unpack
from typing import Optional, Union

from turbojpeg import DEFAULT_LIB_PATHS, TurboJPEG, tjMCUHeight, tjMCUWidth

from opentile.geometry import Size
from opentile.jpeg.jpeg_filler import JpegFiller


class JpegProcess(Enum):
    """JPEG coding process, identified by the start-of-frame (SOF) marker."""

    BASELINE = "baseline"  # SOF0
    EXTENDED = "extended"  # SOF1
    PROGRESSIVE = "progressive"  # SOF2
    LOSSLESS = "lossless"  # SOF3
    OTHER = "other"  # differential / arithmetic / unsupported


@dataclass(frozen=True)
class JpegInfo:
    """Properties read from a JPEG frame header (no pixel decoding)."""

    process: JpegProcess
    """The coding process from the SOF marker."""
    bit_depth: int
    """Sample precision in bits (8 or 12)."""
    components: int
    """Number of components."""
    subsampling: Optional[tuple[int, int]]
    """(horizontal, vertical) maximum component sampling factors, or None for a
    single (grayscale) component. (1, 1) means no subsampling."""
    rgb_signalled: bool
    """True if the frame signals RGB (Adobe APP14 transform 0, or R/G/B component
    ids). A False value does not prove the samples are not RGB: RGB samples can
    be stored without any RGB signal, in which case this is False."""
    lossless_predictor: Optional[int]
    """For a lossless (`SOF3`) frame, the predictor selection value from the SOS
    header; None otherwise. Selection value 1 corresponds to JPEG Lossless,
    First-Order Prediction."""


def find_turbojpeg_path() -> Path:
    """Find the turbojpeg library path.

    Searches using find_library, then DEFAULT_LIB_PATHS from PyTurboJPEG,
    then falls back to the TURBOJPEG environment variable.

    Raises FileNotFoundError if the library cannot be found.
    """
    lib_names = ("libturbojpeg", "turbojpeg")
    lib_extensions: dict[str, tuple[str, ...]] = {
        "nt": (".dll",),
        "posix": (".so", ".so.0", ".dylib"),
    }
    for name in lib_names:
        lib_path = find_library(name)
        if lib_path is not None:
            return Path(lib_path)
    for lib_path_str in DEFAULT_LIB_PATHS.get(platform.system(), []):
        lib_path = Path(lib_path_str)
        if lib_path.exists():
            return lib_path
    turbojpeg_lib_dir = os.environ.get("TURBOJPEG")
    if turbojpeg_lib_dir is not None:
        turbojpeg_path = Path(turbojpeg_lib_dir)
        if turbojpeg_path.is_file():
            return turbojpeg_path
        for name in lib_names:
            for ext in lib_extensions.get(os.name, ()):
                lib_path = turbojpeg_path / (name + ext)
                if lib_path.exists():
                    return lib_path
    raise FileNotFoundError(
        "Could not find turbojpeg library. Install libjpeg-turbo or set the "
        "`TURBOJPEG` environment variable to the library path."
    )


class JpegTagNotFound(Exception):
    """Raised when expected Jpeg tag was not found."""

    pass


class JpegCropError(Exception):
    """Raised when crop operation fails."""

    pass


class Jpeg:
    TAGS = {
        "tag marker": 0xFF,
        "start of image": 0xD8,
        "application default header": 0xE0,
        "quantization table": 0xDB,
        "start of frame": 0xC0,
        "huffman table": 0xC4,
        "start of scan": 0xDA,
        "end of image": 0xD9,
        "restart interval": 0xDD,
        "restart mark": 0xD0,
    }

    def __init__(self, turbo_path: Optional[Union[str, Path]] = None) -> None:
        if turbo_path is None:
            turbo_path = find_turbojpeg_path()
        turbo_path = str(turbo_path)
        self._turbo_jpeg = TurboJPEG(turbo_path)
        self._jpeg_filler = JpegFiller(turbo_path)

    _SOF_MARKERS = frozenset(set(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC})
    """Start-of-frame markers (0xC0-0xCF excluding DHT, JPG, and DAC)."""
    _SOF_PROCESSES = {
        0xC0: JpegProcess.BASELINE,
        0xC1: JpegProcess.EXTENDED,
        0xC2: JpegProcess.PROGRESSIVE,
        0xC3: JpegProcess.LOSSLESS,
    }
    _APP14 = 0xEE
    _SOS = 0xDA
    _RGB_COMPONENT_IDS = (0x52, 0x47, 0x42)  # ascii "R", "G", "B"

    @classmethod
    def info(cls, frame: Union[bytes, bytearray]) -> JpegInfo:
        """Read JPEG properties from a frame header by parsing its markers.

        Only the header is read (SOF, APP14, and, for lossless, the SOS
        predictor); no pixel decoding is performed, so this works for any process
        and bit depth. `rgb_signalled` is True when the frame carries an RGB
        signal: R/G/B component ids in the SOF, or an Adobe APP14 transform of 0.

        Parameters
        ----------
        frame: bytes
            JPEG frame with header.

        Returns
        ----------
        JpegInfo:
            Properties read from the frame header.

        Raises
        ----------
        ValueError:
            If no start-of-frame marker is found.
        """
        sof: Optional[Union[bytes, bytearray]] = None
        sof_marker: Optional[int] = None
        app14_transform: Optional[int] = None
        predictor: Optional[int] = None
        for marker, segment in cls._iter_segments(frame):
            if marker in cls._SOF_MARKERS and sof is None:
                sof_marker, sof = marker, segment
            elif (
                marker == cls._APP14 and len(segment) >= 12 and segment[:5] == b"Adobe"
            ):
                app14_transform = segment[11]
            elif marker == cls._SOS:
                # SOS: Ns(1), then Ns (component, table) pairs, then Ss(1). For a
                # lossless frame the Ss byte is the predictor selection value.
                number_of_components = segment[0]
                spectral_start = 1 + number_of_components * 2
                if spectral_start < len(segment):
                    predictor = segment[spectral_start]
                break
        if sof is None or sof_marker is None:
            raise ValueError("No start-of-frame marker found in JPEG frame")

        process = cls._SOF_PROCESSES.get(sof_marker, JpegProcess.OTHER)
        # SOF: precision(1), height(2), width(2), components(1), then per
        # component id(1), sampling(1, H<<4|V), quantization table(1).
        bit_depth = sof[0]
        components = sof[5]
        ids = [sof[6 + index * 3] for index in range(components)]
        horizontal = [sof[6 + index * 3 + 1] >> 4 for index in range(components)]
        vertical = [sof[6 + index * 3 + 1] & 0x0F for index in range(components)]
        rgb_signalled = tuple(ids) == cls._RGB_COMPONENT_IDS or app14_transform == 0
        subsampling = None if components <= 1 else (max(horizontal), max(vertical))
        return JpegInfo(
            process=process,
            bit_depth=bit_depth,
            components=components,
            subsampling=subsampling,
            rgb_signalled=rgb_signalled,
            lossless_predictor=predictor if process == JpegProcess.LOSSLESS else None,
        )

    @classmethod
    def _iter_segments(
        cls, frame: Union[bytes, bytearray]
    ) -> Iterator[tuple[int, Union[bytes, bytearray]]]:
        """Yield (marker, segment) for each marker segment in a JPEG frame up to
        and including the start-of-scan, then stop. Standalone markers (SOI, EOI,
        RST, TEM) and padding carry no segment and are skipped."""
        index = 0
        length = len(frame)
        while index + 1 < length:
            if frame[index] != 0xFF:
                index += 1
                continue
            marker = frame[index + 1]
            if marker == 0xFF:
                # Fill byte; re-examine from the next byte.
                index += 1
                continue
            index += 2
            if marker in (0x00, 0x01, 0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
                # Padding or standalone markers (SOI/EOI/RST/TEM): no length.
                continue
            if index + 2 > length:
                return
            segment_length = int.from_bytes(frame[index : index + 2], "big")
            yield marker, frame[index + 2 : index + segment_length]
            if marker == cls._SOS:
                return
            index += segment_length

    def get_mcu(self, frame: bytes) -> Size:
        """Return MCU size read from frame header.

        Parameters
        ----------
        frame: bytes
            Frame with header.

        Returns
        ----------
        bytes:
            MCU size in header.
        """
        _, _, subsampling, _ = self._turbo_jpeg.decode_header(frame)
        try:
            return Size(tjMCUWidth[subsampling], tjMCUHeight[subsampling])
        except IndexError:
            raise ValueError(f"Unknown subsampling {subsampling}.") from None

    def concatenate_fragments(self, fragments: Iterator[bytes], header: bytes) -> bytes:
        """Return frame created by vertically concatenating fragments.

        Parameters
        ----------
        fragments: Iterator[bytes]
            Iterator providing fragments to concatenate.
        header: bytes
            Heaeder for the frame.

        Returns
        ----------
        bytes:
            Concatenated frame in bytes.
        """
        frame = header
        for fragment_index, fragment in enumerate(fragments):
            if not (fragment[-2] == Jpeg.TAGS["tag marker"] and fragment[-1] != b"0"):
                raise JpegTagNotFound(
                    "Tag for end of scan or restart marker not found in scan"
                )
            frame += fragment[:-1]  # Do not include restart mark index
            frame += self.restart_mark(fragment_index)
        frame += self.end_of_image()
        return frame

    def concatenate_scans(
        self,
        scans: Iterator[Union[bytes, bytearray]],
        jpeg_tables: Optional[bytes],
        rgb_colorspace_fix: bool = False,
    ) -> bytes:
        """Return frame created by horizontal concatenating scans. Scans must
        have the same header content, and only the last scan is allowed to have
        a different height.

        Parameters
        ----------
        scans: Iterator[bytes]
            Iterator providing scans to concatenate.
        jpeg_tables: Optional[bytes]
            Optional jpeg tables to insert into frame.
        rgb_colorspace_fix: bool = False
            If to apply rgb color space fix (for svs files).

        Returns
        ----------
        bytes:
            Concatenated frame in bytes.
        """
        frame = bytearray()
        image_width: Optional[int] = None
        image_height = 0
        scan_size: Optional[Size] = None
        subsample: Optional[int] = None
        for scan_index, scan in enumerate(scans):
            width, height, _subsample, _ = self._turbo_jpeg.decode_header(scan)
            if image_width is None:
                image_width = width
                image_height = height
                scan_size = Size(width, height)
                scan_start = 0
                subsample = _subsample
            else:
                image_height += height
                start_of_scan, length = self._find_tag(scan, self.start_of_scan())
                if start_of_scan is None or length is None:
                    raise JpegTagNotFound("Start of scan not found in header")
                scan_start = start_of_scan + length + 2

            frame += scan[scan_start:-2]
            frame += b"\xff" + self.restart_mark(scan_index)

        frame[-2:] = self.end_of_image()

        assert (
            image_width is not None and scan_size is not None and subsample is not None
        )
        restart_interval = scan_size.ceil_div(self.subsample_to_mcu(subsample)).area
        image_size = Size(image_width, image_height)
        frame = self._manipulate_header(frame, image_size, restart_interval)

        if jpeg_tables is not None:
            prefix, scan_offset = self.calculate_prefix_and_scan_offset(
                frame, jpeg_tables, rgb_colorspace_fix
            )
            return self.add_jpeg_prefix(prefix, scan_offset, frame)

        return bytes(frame)

    def fill_frame(self, frame: bytes, luminance: float) -> bytes:
        """Return frame filled with color from luminance.

        Parameters
        ----------
        frame: bytes
            Frame to fill.
        luminance: float
            Luminance to fill (0: black - 1: white).

        Returns
        ----------
        bytes:
            Frame with constant color from luminance.
        """
        return self._jpeg_filler.fill_image(frame, luminance)

    def crop_multiple(
        self, frame: bytes, crop_parameters: Sequence[tuple[int, int, int, int]]
    ) -> list[bytes]:
        """Crop multiple frames out of frame.

        Parameters
        ----------
        frame: bytes
            Frame to crop from.
        crop_parameters: Sequence[Tuple[int, int, int, int]]
            Parameters for each crop, specified as left position, top position,
            width, height.

        Returns
        ----------
        List[bytes]:
            Croped frames.
        """
        try:
            return self._turbo_jpeg.crop_multiple(frame, crop_parameters)
        except OSError as exception:
            raise JpegCropError(
                f"Crop of frame failed with parameters {crop_parameters}"
            ) from exception

    @classmethod
    def add_jpeg_prefix(
        cls,
        prefix: bytes,
        scan_offset: int,
        frame: Union[bytes, bytearray],
    ) -> bytes:
        """Build an interchange frame by replacing an abbreviated frame's header
        with a `prefix` produced by `calculate_prefix_and_scan_offset`.

        `prefix` and `scan_offset` are identical for every tile of a tiff page,
        so they can be computed once and reused for all tiles.
        """
        return prefix + bytes(frame[scan_offset:])

    # Adobe APP14 marker with transform flag 0 (image is encoded as RGB, not
    # YCbCr).
    _RGB_COLOR_SPACE_FIX = (
        b"\xff\xee\x00\x0e\x41\x64\x6f\x62\x65\x00\x64\x80\x00\x00\x00\x00"
    )

    @classmethod
    def calculate_prefix_and_scan_offset(
        cls,
        frame: Union[bytes, bytearray],
        jpeg_tables: bytes,
        apply_rgb_colorspace_fix: bool = False,
    ) -> tuple[bytes, int]:
        """Return the header prefix and scan data offset for an abbreviated
        frame, for use with `add_jpeg_prefix`.

        ``add_jpeg_prefix(prefix, scan_offset, frame)`` builds the interchange
        frame. The prefix and offset only depend on the frame header, which is
        identical for every tile of a tiff page, so they can be computed once and
        reused for all tiles instead of re-parsing the header for each tile.
        """
        start_of_scan_index, length = cls._find_tag(frame, cls.start_of_scan())
        if start_of_scan_index is None or length is None:
            raise JpegTagNotFound("Start of scan tag not found in header")
        scan_offset = start_of_scan_index + length + 2
        header = bytearray(frame[:scan_offset])
        if apply_rgb_colorspace_fix:
            cls._set_rgb_component_ids(header)
            tables = cls._RGB_COLOR_SPACE_FIX + jpeg_tables[2:-2]
        else:
            tables = jpeg_tables[2:-2]
        # SOI, then the tables (and APP14), then the rest of the header (start of
        # frame ... start of scan); the caller appends the scan data.
        prefix = bytes(header[:2]) + tables + bytes(header[2:])
        return prefix, scan_offset

    @classmethod
    def manipulate_header(
        cls,
        frame: bytes,
        image_size: Optional[Size] = None,
        restart_interval: Optional[int] = None,
    ) -> bytes:
        """Return frame with changed header to reflect changed image size
        or restart interval.

        Parameters
        ----------
        frame: bytes
            Frame with header to update.
        image_size: Optional[Size] = None
            Image size to update header with.
        restart_interval: Optional[int] = None
            Restart interval to update header with.

        Returns
        ----------
        bytes:
            Frame with updated header.

        """
        return bytes(
            cls._manipulate_header(bytearray(frame), image_size, restart_interval)
        )

    @classmethod
    def start_of_frame(cls) -> bytes:
        """Return bytes representing a start of frame tag."""
        return bytes([cls.TAGS["tag marker"], cls.TAGS["start of frame"]])

    @classmethod
    def start_of_scan(cls) -> bytes:
        """Return bytes representing a start of scan tag."""
        return bytes([cls.TAGS["tag marker"], cls.TAGS["start of scan"]])

    @classmethod
    def end_of_image(cls) -> bytes:
        """Return bytes representing a end of image tag."""
        return bytes([cls.TAGS["tag marker"], cls.TAGS["end of image"]])

    @classmethod
    def restart_mark(cls, index: int) -> bytes:
        """Return bytes representing a restart marker of index (0-7), without
        the prefixing tag (0xFF)."""
        return bytes([cls.TAGS["restart mark"] + index % 8])

    @classmethod
    def restart_interval(cls) -> bytes:
        return bytes([cls.TAGS["tag marker"], cls.TAGS["restart interval"]])

    @staticmethod
    def code_short(value: int) -> bytes:
        return pack(">H", value)

    @staticmethod
    def subsample_to_mcu(subsample: int) -> Size:
        """Return the MCU size (in pixels) for a turbojpeg subsample value."""
        return Size(tjMCUWidth[subsample], tjMCUHeight[subsample])

    @staticmethod
    def _find_tag(
        frame: Union[bytes, bytearray], tag: bytes
    ) -> tuple[Optional[int], Optional[int]]:
        """Return first index and length of payload of tag in header.

        Parameters
        ----------
        frame: bytes
            Frame with header to search.
        tag: bytes
            Tag to search for.

        Returns
        ----------
        Tuple[Optional[int], Optional[int]]:
            Position of tag in header and length of payload.
        """
        index = frame.find(tag)
        if index != -1:
            (length,) = unpack(">H", frame[index + 2 : index + 4])
            return index, length

        return None, None

    @classmethod
    def _manipulate_header(
        cls,
        frame: bytearray,
        size: Optional[Size] = None,
        restart_interval: Optional[int] = None,
    ) -> bytearray:
        """Return manipulated header with changed pixel size (width, height)
        and/or restart interval.

        Parameters
        ----------
        frame: bytearray
            Frame to manipulate header of.
        size: Optional[Size] = None
            Pixel size to insert into header.
        restart_interval: Optional[int] = None
            Restart interval to code into header.

        Returns
        ----------
        bytearray:
            Manipulated header.
        """
        if size is not None:
            start_of_frame_index, _ = cls._find_tag(frame, cls.start_of_frame())
            if start_of_frame_index is None:
                raise JpegTagNotFound("Start of frame tag not found in header")
            size_index = start_of_frame_index + 5
            frame[size_index : size_index + 2] = cls.code_short(size.height)
            frame[size_index + 2 : size_index + 4] = cls.code_short(size.width)

        if restart_interval is not None:
            restart_payload = cls.code_short(restart_interval)
            restart_index, _ = cls._find_tag(frame, cls.restart_interval())
            if restart_index is not None:
                # Modify existing restart tag
                payload_index = restart_index + 4
                frame[payload_index : payload_index + 2] = restart_payload
            else:
                # Make and insert new restart tag
                start_of_scan_index, _ = cls._find_tag(frame, cls.start_of_scan())
                if start_of_scan_index is None:
                    raise JpegTagNotFound("Start of scan tag not found in header")
                frame[start_of_scan_index:start_of_scan_index] = (
                    cls.restart_interval() + cls.code_short(4) + restart_payload
                )
        return frame

    # ASCII 'R', 'G', 'B'. libjpeg (and thus most decoders) treat a three
    # component frame with these component ids as RGB when no Adobe or JFIF
    # marker is present.
    RGB_COMPONENT_IDS = (0x52, 0x47, 0x42)

    @classmethod
    def _set_rgb_component_ids(cls, frame: bytearray) -> bytearray:
        """Rename the three frame components to ASCII 'R', 'G', 'B'.

        Only applied to three component (RGB) frames; other frames are returned
        unchanged. The scan component selectors reference the frame component
        ids by value and are updated to match.
        """
        start_of_frame_index, _ = cls._find_tag(frame, cls.start_of_frame())
        if start_of_frame_index is None:
            raise JpegTagNotFound("Start of frame tag not found in header")
        component_count = frame[start_of_frame_index + 9]
        if component_count != 3:
            return frame
        id_map: dict[int, int] = {}
        for component, new_id in enumerate(cls.RGB_COMPONENT_IDS):
            id_index = start_of_frame_index + 10 + component * 3
            id_map[frame[id_index]] = new_id
            frame[id_index] = new_id

        start_of_scan_index, _ = cls._find_tag(frame, cls.start_of_scan())
        if start_of_scan_index is None:
            raise JpegTagNotFound("Start of scan tag not found in header")
        scan_component_count = frame[start_of_scan_index + 4]
        for component in range(scan_component_count):
            selector_index = start_of_scan_index + 5 + component * 2
            frame[selector_index] = id_map.get(
                frame[selector_index], frame[selector_index]
            )
        return frame
