from ctypes import *
from typing import List, Tuple

import numpy as np
from turbojpeg import (TJFLAG_ACCURATEDCT, TJXOP_NONE, TJXOPT_CROP,
                       TJXOPT_GRAY, TJXOPT_PERFECT, CroppingRegion, TurboJPEG)

CUSTOMFILTER = CFUNCTYPE(
    c_int,
    POINTER(c_short),
    CroppingRegion,
    CroppingRegion,
    c_int,
    c_int,
    c_void_p
)


class BackgroundStruct(Structure):
    """Struct to send data to fill_background callback function.

    Parameters
    ----------
    w: c_int
        Width of the input image.
    h: c_int
        Height of the input image.
    lum: c_int
        Luminance value to use as background when extending the image.
    """
    _fields_ = [
        ("w", c_int),
        ("h", c_int),
        ("lum", c_int)
    ]


class TransformStruct(Structure):
    _fields_ = [
        ("r", CroppingRegion),
        ("op", c_int),
        ("options", c_int),
        ("data", POINTER(BackgroundStruct)),
        ("customFilter", CUSTOMFILTER)
    ]


def fill_background(
    coeffs_ptr: POINTER(c_short),
    arrayRegion: CroppingRegion,
    planeRegion: CroppingRegion,
    componentID: c_int,
    transformID: c_int,
    transform_ptr: c_void_p
) -> c_int:
    """Callback function for filling extended crop images with background
    color. The callback can be called multiple times for each component, each
    call providing a region (defined by arrayRegion) of the image.

    Parameters
    ----------
    coeffs_ptr: POINTER(c_short)
        Pointer to the coefficient array for the callback.
    arrayRegion: CroppingRegion
        The width and height coefficient array and its offset relative to
        the component plane.
    planeRegion: CroppingRegion
        The width and height of the component plane of the coefficient array.
    componentID: c_int
        The component number (i.e. 0, 1, or 2)
    transformID: c_int
        The index of the transformation in the array of transformation given to
        the transform function.
    transform_ptr: c_voipd_p
        Pointer to the transform structure used for the transformation.

    Returns
    ----------
    c_int
        CFUNCTYPE function must return an int.
    """

    # Only modify luminance data, so we dont need to worry about subsampling
    if componentID == 0:
        MCU_WIDTH = 8
        MCU_HEIGHT = 8
        MCU_SIZE = 64
        # print("-----------")
        # print(
        #     f"arrayRegion x {arrayRegion.x}, y {arrayRegion.y}",
        #     f" w {arrayRegion.w}, h {arrayRegion.h}"
        # )
        # print(
        #     f"planeRegion  w {planeRegion.w}, h {planeRegion.h}"
        # )
        # print(f"componentID {componentID}")
        # print(f"transformID {transformID}")

        coeff_array_size = arrayRegion.w * arrayRegion.h

        # Read the coefficients in the pointer as a np array (no copy)
        ArrayType = c_short*coeff_array_size
        array_pointer = cast(coeffs_ptr, POINTER(ArrayType))
        coeffs = np.frombuffer(array_pointer.contents, dtype=np.int16)
        coeffs.shape = (
            arrayRegion.h//MCU_WIDTH,
            arrayRegion.w//MCU_HEIGHT,
            MCU_SIZE
        )

        # Cast the content of the transform pointer into a transform structure
        transform = cast(transform_ptr, POINTER(TransformStruct)).contents
        # Cast the content of the callback data pointer in the transform
        # structure to a background structure
        background_data = cast(
            transform.data, POINTER(BackgroundStruct)
        ).contents
        # print(
        #     f"callback data: w {background_data.w}, h {background_data.h}, "
        #     f"luminance {background_data.lum}"
        # )

        # The coeff array is typically just one MCU heigh, but it is up to the
        # libjpeg implementation how to do it. The part of the coeff array that
        # is 'left' of 'non-background' data should thus be handled separately
        # from the part 'under'. (Most of the time, the coeff array will be
        # either 'left' or 'under', but both could happen)

        # mcus left of image, currently 'does not stop' when 'under' the image

        for x in range(background_data.w//MCU_WIDTH, planeRegion.w//MCU_WIDTH):
            for y in range(0, arrayRegion.h//MCU_WIDTH):
                coeffs[y][x] = background_data.lum

        # mcus under image,
        # TBD

    return 1


class TurboJPEG_patch(TurboJPEG):
    def __init__(self, lib_path=None):
        super().__init__(lib_path)
        turbo_jpeg = cdll.LoadLibrary(
            self.__find_turbojpeg() if lib_path is None else lib_path)
        self.__transform = turbo_jpeg.tjTransform
        self.__transform.argtypes = [
            c_void_p,
            POINTER(c_ubyte),
            c_ulong, c_int,
            POINTER(c_void_p),
            POINTER(c_ulong),
            POINTER(TransformStruct),
            c_int
        ]
        self.__transform.restype = c_int

    def crop_multiple(
        self,
        jpeg_buf: bytes,
        crop_parameters: List[Tuple[int, int, int, int]],
        background_luminance: int = 508,
        gray: bool = False
    ) -> List[bytes]:
        """Lossless crop and/or extension operations on jpeg image.
        Crop origin(s) needs be divisable by the MCU block size and inside
        the input image, or OSError: Invalid crop request is raised.

        Parameters
        ----------
        jpeg_buf: bytes
            Input jpeg image.
        crop_parameters: List[Tuple[int, int, int, int]]
            List of crop parameters defining start x and y origin and width
            and height of each crop operation.
        background_luminance: int
            Luminance level to fill background when extending image. Default to
            full, resulting in white background.
        gray: bool
            Produce greyscale output

        Returns
        ----------
        List[bytes]
            Cropped and/or extended jpeg images.
        """
        handle: c_void_p = self._TurboJPEG__init_transform()
        try:
            jpeg_array: np.ndarray = np.frombuffer(jpeg_buf, dtype=np.uint8)
            src_addr = self._TurboJPEG__getaddr(jpeg_array)
            image_width = c_int()
            image_height = c_int()
            jpeg_subsample = c_int()
            jpeg_colorspace = c_int()

            # Decompress header to get input image size and subsample value
            decompress_header_status: int = self._TurboJPEG__decompress_header(
                handle,
                src_addr,
                jpeg_array.size,
                byref(image_width),
                byref(image_height),
                byref(jpeg_subsample),
                byref(jpeg_colorspace)
            )

            if decompress_header_status != 0:
                self._TurboJPEG__report_error(handle)

            # Define cropping regions from input parameters and image size
            crop_regions = self.__define_cropping_regions(crop_parameters)

            number_of_operations = len(crop_regions)

            # Define crop transforms from cropping_regions
            crop_transforms = (TransformStruct * number_of_operations)()
            for i, crop_region in enumerate(crop_regions):
                # The fill_background callback is slow, only use it if needed
                if self.__need_fill_background(
                    crop_region,
                    (image_width.value, image_height.value),
                    background_luminance
                ):
                    # Use callback to fill in background post-transform
                    crop_transforms[i] = TransformStruct(
                        crop_region,
                        TJXOP_NONE,
                        TJXOPT_PERFECT | TJXOPT_CROP | (gray and TJXOPT_GRAY),
                        pointer(BackgroundStruct(
                            image_width,
                            image_height,
                            background_luminance
                        )),
                        CUSTOMFILTER(fill_background)
                    )
                else:
                    crop_transforms[i] = TransformStruct(
                        crop_region,
                        TJXOP_NONE,
                        TJXOPT_PERFECT | TJXOPT_CROP | (gray and TJXOPT_GRAY)
                    )

            # Pointers to output image buffers and buffer size
            dest_array = (c_void_p * number_of_operations)()
            dest_size = (c_ulong * number_of_operations)()

            # Do the transforms
            transform_status = self.__transform(
                handle,
                src_addr,
                jpeg_array.size,
                number_of_operations,
                dest_array,
                dest_size,
                crop_transforms,
                TJFLAG_ACCURATEDCT
            )

            if transform_status != 0:
                self._TurboJPEG__report_error(handle)

            # Copy the transform results into python bytes
            results: List[bytes] = []
            for i in range(number_of_operations):
                dest_buf = create_string_buffer(dest_size[i])
                memmove(dest_buf, dest_array[i], dest_size[i])
                results.append(dest_buf.raw)

            # Free the output image buffers
            for dest in dest_array:
                self._TurboJPEG__free(dest)

            return results

        finally:
            self._TurboJPEG__destroy(handle)

    @staticmethod
    def __define_cropping_regions(
        crop_parameters: List[Tuple[int, int, int, int]]
    ) -> List[CroppingRegion]:
        """Return list of crop regions from crop parameters

        Parameters
        ----------
        crop_parameters: List[Tuple[int, int, int, int]]
            List of crop parameters defining start x and y origin and width
            and height of each crop operation.

        Returns
        ----------
        List[CroppingRegion]
            List of crop operations, size is equal to the product of number of
            crop operations to perform in x and y direction.
        """
        return [
            CroppingRegion(x=crop[0], y=crop[1], w=crop[2], h=crop[3])
            for crop in crop_parameters
        ]

    @staticmethod
    def __need_fill_background(
        crop_region: CroppingRegion,
        image_size: Tuple[int, int],
        background_luminance: int
    ) -> bool:
        """Return true if crop operation require background fill operation.

        Parameters
        ----------
        crop_region: CroppingRegion
            The crop region to check.
        image_size: [int, int]
            Size of input image.
        background_luminance: int
            Requested background luminance.

        Returns
        ----------
        bool
            True if crop operation require background fill operation.
        """
        return (
            (
                (crop_region.x + crop_region.w > image_size[0])
                or
                (crop_region.y + crop_region.h > image_size[1])
            )
            and (background_luminance != 0)
        )


