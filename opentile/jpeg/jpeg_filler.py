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

"""Lossless filling of a JPEG image with a constant color using the same tables."""

from ctypes import (
    POINTER,
    Structure,
    _Pointer,
    byref,
    c_char_p,
    c_int,
    c_short,
    c_ubyte,
    c_ulong,
    c_void_p,
    cast,
    cdll,
    create_string_buffer,
    memmove,
    pointer,
)
from pathlib import Path
from struct import calcsize, unpack
from typing import Optional, Union

import numpy as np
from turbojpeg import (
    CUSTOMFILTER,
    TJXOP_NONE,
    TJXOPT_PERFECT,
    CroppingRegion,
    tjMCUHeight,
    tjMCUWidth,
)

# Uses the libjpeg-turbo 2.x C API (tjInitTransform, tjTransform, etc.)
# which is also available in libjpeg-turbo 3.x for backwards compatibility.
TJFLAG_ACCURATEDCT = 4096


class BlankStruct(Structure):
    """Struct to send data to blank_image callback function.

    Parameters
    ----------
    subsample: c_int
        Subsample value of image.
    lum: c_int
        Luminance value to use as background when extending the image.
    """

    _fields_ = [
        ("subsample", c_int),
        ("lum", c_int),
    ]


class BlankTransformStruct(Structure):
    _fields_ = [
        ("r", CroppingRegion),
        ("op", c_int),
        ("options", c_int),
        ("data", POINTER(BlankStruct)),
        ("customFilter", CUSTOMFILTER),
    ]


class BlankImage:
    _operation = TJXOP_NONE
    _options = TJXOPT_PERFECT

    @staticmethod
    def get_transform_data(transform_ptr: _Pointer) -> BlankStruct:
        # Cast the content of the transform pointer into a transform structure
        transform = cast(transform_ptr, POINTER(BlankTransformStruct)).contents
        # Cast the content of the callback data pointer in the transform
        # structure to a background structure
        return cast(transform.data, POINTER(BlankStruct)).contents

    @staticmethod
    def get_np_coeffs(coeffs_ptr: _Pointer, array_region: CroppingRegion) -> np.ndarray:
        coeff_array_size = array_region.w * array_region.h
        # Read the coefficients in the pointer as a np array (no copy)
        array_type = c_short * coeff_array_size
        array_pointer = cast(coeffs_ptr, POINTER(array_type))
        coeffs = np.frombuffer(array_pointer.contents, dtype=np.int16)
        coeffs.shape = (array_region.h // 8, array_region.w // 8, 64)
        return coeffs

    @classmethod
    def callback(
        cls,
        coeffs_ptr: _Pointer,
        array_region: CroppingRegion,
        plane_region: CroppingRegion,
        component_ID: int,
        transform_ID: int,
        transform_ptr: _Pointer,
    ) -> int:
        """Callback function for filling whole image with background color.

        Parameters
        ----------
        coeffs_ptr: _Pointer
            Pointer to the coefficient array for the callback.
        array_region: CroppingRegion
            The width and height coefficient array and its offset relative to
            the component plane.
        plane_region: CroppingRegion
            The width and height of the component plane of the coefficient
            array.
        component_ID: int
            The component number (i.e. 0, 1, or 2)
        transform_ID: int
            The index of the transformation in the array of transformation
            given to the transform function.
        transform_ptr: _Pointer
            Pointer to the transform structure used for the transformation.

        Returns
        ----------
        int
            CFUNCTYPE function must return an int.
        """
        background_data = cls.get_transform_data(transform_ptr)

        if component_ID == 0:
            dc_component = background_data.lum
            subsampling = 0
        else:
            dc_component = 0
            subsampling = background_data.subsample
        coeffs = cls.get_np_coeffs(coeffs_ptr, array_region)
        coeffs[:][:][:] = 0

        for x in range(0, array_region.w // tjMCUWidth[subsampling]):
            for y in range(0, array_region.h // tjMCUHeight[subsampling]):
                coeffs[y][x][0] = dc_component

        return 1

    @staticmethod
    def callback_data(jpeg_subsample: int, luminance: int) -> BlankStruct:
        return BlankStruct(jpeg_subsample, luminance)

    @classmethod
    def transform(
        cls,
        region: CroppingRegion,
        callback_data: BlankStruct,
    ) -> BlankTransformStruct:
        return BlankTransformStruct(
            region,
            cls._operation,
            cls._options,
            pointer(callback_data),
            CUSTOMFILTER(cls.callback),
        )


class JpegFiller:
    """Lossless filling of a JPEG image with a constant color.

    Uses libjpeg-turbo's 2.x transform API with a custom callback to replace
    all DCT coefficients, producing a solid-color image that preserves
    the original encoding tables. The 2.x API is also supported by
    libjpeg-turbo 3.x for backwards compatibility.
    """

    def __init__(self, lib_path: Union[str, Path]):
        lib = cdll.LoadLibrary(str(lib_path))

        self._init_transform = lib.tjInitTransform
        self._init_transform.argtypes = []
        self._init_transform.restype = c_void_p

        self._destroy = lib.tjDestroy
        self._destroy.argtypes = [c_void_p]
        self._destroy.restype = c_int

        self._decompress_header = lib.tjDecompressHeader3
        self._decompress_header.argtypes = [
            c_void_p, POINTER(c_ubyte), c_ulong,
            POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_int),
        ]
        self._decompress_header.restype = c_int

        self._transform = lib.tjTransform
        self._transform.argtypes = [
            c_void_p, POINTER(c_ubyte), c_ulong, c_int,
            POINTER(c_void_p), POINTER(c_ulong), POINTER(BlankTransformStruct), c_int,
        ]
        self._transform.restype = c_int

        self._free = lib.tjFree
        self._free.argtypes = [c_void_p]
        self._free.restype = None

        self._get_error_str = lib.tjGetErrorStr2
        self._get_error_str.argtypes = [c_void_p]
        self._get_error_str.restype = c_char_p

        self._blank_image_transform = BlankImage()

    def fill_image(
        self,
        jpeg_buf: bytes,
        background_luminance: float = 1.0,
    ) -> bytes:
        """Lossless fill jpeg image with background luminance.

        Parameters
        ----------
        jpeg_buf: bytes
            Input jpeg image.
        background_luminance: float
            Luminance level (0 -1 ) to fill background when extending image.
            Default to 1, resulting in white background.

        Returns
        ----------
        bytes
            Filled jpeg image.
        """
        handle = self._init_transform()
        try:
            jpeg_array: np.ndarray = np.frombuffer(jpeg_buf, dtype=np.uint8)
            src_addr = cast(jpeg_array.__array_interface__["data"][0], POINTER(c_ubyte))

            width = c_int()
            height = c_int()
            subsample = c_int()
            colorspace = c_int()
            status = self._decompress_header(
                handle, src_addr, jpeg_array.size,
                byref(width), byref(height), byref(subsample), byref(colorspace),
            )
            if status != 0:
                self._raise_error(handle)

            callback_data = self._blank_image_transform.callback_data(
                subsample.value,
                self._map_luminance_to_dc_dct_coefficient(
                    jpeg_buf, background_luminance
                ),
            )
            dest_array = c_void_p()
            dest_size = c_ulong()
            transform = self._blank_image_transform.transform(
                CroppingRegion(0, 0, width, height),
                callback_data,
            )

            transform_status = self._transform(
                handle, src_addr, jpeg_array.size, 1,
                byref(dest_array), byref(dest_size), byref(transform),
                TJFLAG_ACCURATEDCT,
            )

            dest_buf = create_string_buffer(dest_size.value)
            assert dest_array.value is not None
            memmove(dest_buf, dest_array.value, dest_size.value)

            self._free(dest_array)

            if transform_status != 0:
                self._raise_error(handle)

            return dest_buf.raw

        finally:
            self._destroy(handle)

    @classmethod
    def _map_luminance_to_dc_dct_coefficient(
        cls, jpeg_data: bytes, luminance: float
    ) -> int:
        """Map a luminance level (0 - 1) to quantified dc dct coefficient."""
        luminance = min(max(luminance, 0), 1)
        dc_dqt_coefficient = cls._get_dc_dqt_element(jpeg_data, 0)
        return int(round((luminance * 2047 - 1024) / dc_dqt_coefficient))

    @staticmethod
    def _find_dqt(jpeg_data: bytes, dqt_index: int) -> Optional[int]:
        """Return byte offset to quantification table with index dqt_index."""
        offset = 0
        while offset < len(jpeg_data):
            dct_table_offset = jpeg_data[offset:].find(b"\xff\xdb")
            if dct_table_offset == -1:
                break
            dct_table_offset += offset
            dct_table_length = unpack(
                ">H", jpeg_data[dct_table_offset + 2 : dct_table_offset + 4]
            )[0]
            dct_table_id_offset = dct_table_offset + 4
            table_index = jpeg_data[dct_table_id_offset] >> 4
            if table_index == dqt_index:
                return dct_table_offset
            offset = dct_table_offset + dct_table_length
        return None

    @classmethod
    def _get_dc_dqt_element(cls, jpeg_data: bytes, dqt_index: int) -> int:
        """Return dc quantification element from jpeg_data for quantification
        table dqt_index."""
        dqt_offset = cls._find_dqt(jpeg_data, dqt_index)
        if dqt_offset is None:
            raise ValueError(f"Quantisation table {dqt_index} not found in header")
        precision_offset = dqt_offset + 4
        precision = jpeg_data[precision_offset] & 0x0F
        if precision == 0:
            unpack_type = ">b"
        elif precision == 1:
            unpack_type = ">h"
        else:
            raise ValueError("Not valid precision definition in dqt")
        dc_offset = dqt_offset + 5
        dc_length = calcsize(unpack_type)
        dc_value = unpack(unpack_type, jpeg_data[dc_offset : dc_offset + dc_length])[0]
        return dc_value

    def _raise_error(self, handle: c_void_p) -> None:
        error_str = self._get_error_str(handle)
        msg = error_str.decode() if error_str else "Unknown TurboJPEG error"
        raise IOError(msg)
