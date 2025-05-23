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

from pathlib import Path
from struct import pack, unpack
from typing import Iterator, List, Optional, Sequence, Tuple, Union

from turbojpeg import tjMCUHeight, tjMCUWidth

from opentile.geometry import Size
from opentile.jpeg.turbojpeg_patch import TurboJPEG_patch as TurboJPEG
from opentile.jpeg.turbojpeg_patch import find_turbojpeg_path


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
        self._turbo_jpeg = TurboJPEG(turbo_path)

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
            raise ValueError(f"Unknown subsampling {subsampling}.")

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
        scans: Iterator[bytes],
        jpeg_tables: Optional[bytes],
        rgb_colorspace_fix: bool = False,
    ) -> bytes:
        """Return frame created by horisontal concatenating scans. Scans must
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
        image_size: Optional[Size] = None
        scan_size: Optional[Size] = None
        subsample: Optional[int] = None
        for scan_index, scan in enumerate(scans):
            width, height, _subsample, _ = self._turbo_jpeg.decode_header(scan)
            if image_size is None:
                image_size = Size(width, height)
                scan_size = Size(width, height)
                scan_start = 0
                subsample = _subsample
            else:
                image_size.height += height
                start_of_scan, length = self._find_tag(scan, self.start_of_scan())
                if start_of_scan is None or length is None:
                    raise JpegTagNotFound("Start of scan not found in header")
                scan_start = start_of_scan + length + 2

            frame += scan[scan_start:-2]
            frame += b"\xff" + self.restart_mark(scan_index)

        frame[-2:] = self.end_of_image()

        if jpeg_tables is not None:
            if rgb_colorspace_fix:
                frame = self._add_jpeg_tables_and_rgb_color_space_fix(
                    frame, jpeg_tables
                )
            else:
                frame = self._add_jpeg_tables(frame, jpeg_tables)

        assert (
            image_size is not None and scan_size is not None and subsample is not None
        )
        frame = self._manipulate_header(
            frame, image_size, scan_size.area // self.subsample_to_mcu_size(subsample)
        )
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
        return self._turbo_jpeg.fill_image(frame, luminance)

    def crop_multiple(
        self, frame: bytes, crop_parameters: Sequence[Tuple[int, int, int, int]]
    ) -> List[bytes]:
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
                f"Crop of frame failed " f"with parameters {crop_parameters}"
            ) from exception

    @classmethod
    def add_jpeg_tables(
        cls,
        frame: bytes,
        jpeg_tables: bytes,
        apply_rgb_colorspace_fix: Optional[bool] = False,
    ) -> bytes:
        """Add jpeg tables to frame. Tables are inserted before 'start of
        scan'-tag, and leading 'start of image' and ending 'end of image' tags
        are removed from the header prior to insertion.

        Parameters
        ----------
        frame: bytes
            'Abbreviated' jpeg frame lacking jpeg tables.
        jpeg_tables: bytes
            Jpeg tables to add
        apply_rgb_colorspace_fix: Optional[bool] = False
            If to also apply

        Returns
        ----------
        bytes:
            'Interchange' jpeg frame containing jpeg tables.

        """

        if apply_rgb_colorspace_fix:
            frame = cls._add_jpeg_tables_and_rgb_color_space_fix(
                bytearray(frame), jpeg_tables
            )
        else:
            frame = cls._add_jpeg_tables(bytearray(frame), jpeg_tables)
        return bytes(frame)

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
    def subsample_to_mcu_size(subsample: int) -> int:
        return tjMCUWidth[subsample] * tjMCUHeight[subsample]

    @staticmethod
    def _find_tag(
        frame: Union[bytes, bytearray], tag: bytes
    ) -> Tuple[Optional[int], Optional[int]]:
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

    @classmethod
    def _add_jpeg_tables(
        cls,
        frame: bytearray,
        jpegtables: bytes,
    ) -> bytearray:
        """Add jpeg tables to frame."""
        start_of_scan = frame.find(cls.start_of_scan())
        frame[start_of_scan:start_of_scan] = jpegtables[2:-2]
        return frame

    @classmethod
    def _add_jpeg_tables_and_rgb_color_space_fix(
        cls,
        frame: bytearray,
        jpegtables: bytes,
    ) -> bytearray:
        """Add jpeg tables to frame and add Adobe APP14 marker with transform
        flag 0 indicating image is encoded as RGB (not YCbCr)."""
        start_of_scan = frame.find(cls.start_of_scan())
        frame[start_of_scan:start_of_scan] = (
            jpegtables[2:-2]
            + b"\xff\xee\x00\x0e\x41\x64\x6f\x62"
            + b"\x65\x00\x64\x80\x00\x00\x00\x00"
        )
        return frame
