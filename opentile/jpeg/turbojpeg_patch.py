#    Copyright 2021 SECTRA AB
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

"""Extension of turbojpeg to enable replacing a frame with a constant color using the
same tables."""

import os
from ctypes import (
    POINTER,
    Structure,
    _Pointer,
    byref,
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
from ctypes.util import find_library
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from turbojpeg import (
    CUSTOMFILTER,
    TJFLAG_ACCURATEDCT,
    TJXOP_NONE,
    TJXOPT_PERFECT,
    CroppingRegion,
    TurboJPEG,
    tjMCUHeight,
    tjMCUWidth,
)


def find_turbojpeg_path() -> Optional[Path]:
    # Only windows installs libraries on strange places
    if os.name != "nt":
        return None
    turbojpeg_lib_path = find_library("turbojpeg")
    if turbojpeg_lib_path is not None:
        return Path(turbojpeg_lib_path)
    turbojpeg_lib_dir = os.environ.get("TURBOJPEG")
    if turbojpeg_lib_dir is not None:
        turbojpeg_lib_path = Path(turbojpeg_lib_dir).joinpath("turbojpeg.dll")
        if turbojpeg_lib_path.exists():
            return turbojpeg_lib_path
    raise ModuleNotFoundError(
        "Could not find turbojpeg.dll in the directories specified "
        "in the `Path` or `TURBOJPEG` environmental variable. Please add the "
        "directory with turbojpeg.dll to the `Path` or `TURBOJPEG` "
        "environmental variable."
    )


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
    def callback_data(jpeg_subsample: c_int, luminance: int) -> BlankStruct:
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


class TurboJPEG_patch(TurboJPEG):
    def __init__(self, lib_turbojpeg_path: Optional[Union[str, Path]] = None):
        if lib_turbojpeg_path is not None:
            lib_turbojpeg_str_path = str(lib_turbojpeg_path)
        else:
            lib_turbojpeg_str_path = str(self._find_turbojpeg())
        super().__init__(lib_turbojpeg_str_path)
        turbo_jpeg = cdll.LoadLibrary(lib_turbojpeg_str_path)
        self.__blank_transform = turbo_jpeg.tjTransform
        self.__blank_transform.argtypes = [
            c_void_p,
            POINTER(c_ubyte),
            c_ulong,
            c_int,
            POINTER(c_void_p),
            POINTER(c_ulong),
            POINTER(BlankTransformStruct),
            c_int,
        ]
        self.__blank_transform.restype = c_int
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
        List[bytes]
            Filled jpeg images.
        """
        handle = self._init_transform()
        try:
            jpeg_array: np.ndarray = np.frombuffer(jpeg_buf, dtype=np.uint8)
            src_addr = self._getaddr(jpeg_array)
            image_width = c_int()
            image_height = c_int()
            jpeg_subsample = c_int()
            jpeg_colorspace = c_int()

            # Decompress header to get input image size and subsample value
            decompress_header_status = self._decompress_header(
                handle,
                src_addr,
                jpeg_array.size,
                byref(image_width),
                byref(image_height),
                byref(jpeg_subsample),
                byref(jpeg_colorspace),
            )

            if decompress_header_status != 0:
                self._report_error(handle)

            # Use callback to fill in background post-transform
            callback_data = self._blank_image_transform.callback_data(
                jpeg_subsample,
                self._map_luminance_to_dc_dct_coefficient(
                    jpeg_buf, background_luminance
                ),
            )
            # Pointers to output image buffers and buffer size
            dest_array = c_void_p()
            dest_size = c_ulong()
            transform = self._blank_image_transform.transform(
                CroppingRegion(0, 0, image_width, image_height),
                callback_data,
            )
            # Do the transforms
            transform_status = self.__blank_transform(
                handle,
                src_addr,
                jpeg_array.size,
                1,
                byref(dest_array),
                byref(dest_size),
                byref(transform),
                TJFLAG_ACCURATEDCT,
            )

            # Copy the transform results into python bytes
            dest_buf = create_string_buffer(dest_size.value)
            assert dest_array.value is not None
            memmove(dest_buf, dest_array.value, dest_size.value)

            # Free the output image buffers
            self._free(dest_array)

            if transform_status != 0:
                self._report_error(handle)

            return dest_buf.raw

        finally:
            self._destroy(handle)

    @classmethod
    def _map_luminance_to_dc_dct_coefficient(
        cls, jpeg_data: bytes, luminance: float
    ) -> int:
        return cls._TurboJPEG__map_luminance_to_dc_dct_coefficient(  # type: ignore # NOQA
            jpeg_data, luminance
        )

    def _find_turbojpeg(self) -> str:
        return self._TurboJPEG__find_turbojpeg()  # type: ignore

    def _init_transform(self) -> c_void_p:
        return self._TurboJPEG__init_transform()  # type: ignore

    def _getaddr(self, nda: np.ndarray) -> _Pointer:
        return self._TurboJPEG__getaddr(nda)  # type: ignore

    def _decompress_header(
        self,
        handle: c_void_p,
        src_addr,
        jpeg_array_size: int,
        image_width,
        image_height,
        jpeg_subsample,
        jpeg_colorspace,
    ) -> c_int:
        return self._TurboJPEG__decompress_header(  # type: ignore
            handle,
            src_addr,
            jpeg_array_size,
            image_width,
            image_height,
            jpeg_subsample,
            jpeg_colorspace,
        )

    def _report_error(self, handle: c_void_p) -> None:
        self._TurboJPEG__report_error(handle)  # type: ignore

    def _free(self, dest_array: c_void_p) -> None:
        self._TurboJPEG__free(dest_array)  # type: ignore

    def _destroy(self, handle: c_void_p) -> c_int:
        return self._TurboJPEG__destroy(handle)  # type: ignore

    @staticmethod
    def _need_fill_background(
        crop_region: CroppingRegion,
        image_size: Tuple[int, int],
        background_luminance: float,
    ) -> bool:
        return TurboJPEG._TurboJPEG__need_fill_background(  # type: ignore
            crop_region, image_size, background_luminance
        )

    @staticmethod
    def _define_cropping_regions(
        crop_parameters: List[Tuple[int, int, int, int]]
    ) -> List[CroppingRegion]:
        return TurboJPEG._TurboJPEG__define_cropping_regions(  # type: ignore
            crop_parameters
        )
