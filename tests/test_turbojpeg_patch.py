import unittest
from ctypes import *

import numpy as np
import pytest
from opentile.turbojpeg_patch import (CUSTOMFILTER, TJXOP_NONE, TJXOPT_CROP,
                                      TJXOPT_GRAY, TJXOPT_PERFECT,
                                      BackgroundStruct, CroppingRegion,
                                      TransformStruct)
from opentile.turbojpeg_patch import TurboJPEG_patch as TurboJPEG
from opentile.turbojpeg_patch import (fill_background,
                                      fill_whole_image_with_background)

turbo_path = 'C:/libjpeg-turbo64/bin/turbojpeg.dll'
test_file_path = 'C:/temp/opentile/turbojpeg/frame_1024x512.jpg'


@pytest.mark.turbojpeg
class TurboJpegTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls):
        cls.jpeg = TurboJPEG(turbo_path)
        cls.test_file = open(test_file_path, 'rb')
        cls.buffer = cls.test_file.read()

    @classmethod
    def tearDownClass(cls):
        cls.test_file.close()

    def test__need_fill_background(self):
        image_size = (2048, 1024)
        crop_region = CroppingRegion(0, 0, 512, 512)
        self.assertFalse(
            self.jpeg._TurboJPEG_patch__need_fill_background(
                crop_region,
                image_size
            )
        )

        crop_region = CroppingRegion(0, 0, 2048, 1024)
        self.assertFalse(
            self.jpeg._TurboJPEG_patch__need_fill_background(
                crop_region,
                image_size
            )
        )

        crop_region = CroppingRegion(1024, 0, 1024, 1024)
        self.assertFalse(
            self.jpeg._TurboJPEG_patch__need_fill_background(
                crop_region,
                image_size
            )
        )

        crop_region = CroppingRegion(0, 0, 2048, 2048)
        self.assertTrue(
            self.jpeg._TurboJPEG_patch__need_fill_background(
                crop_region,
                image_size
            )
        )

    def test__define_cropping_regions(self):
        crop_parameters = [(0, 1, 2, 3), (4, 5, 6, 7)]
        expected_cropping_regions = [
            CroppingRegion(0, 1, 2, 3),
            CroppingRegion(4, 5, 6, 7)
        ]
        cropping_regions = self.jpeg._TurboJPEG_patch__define_cropping_regions(
                crop_parameters
        )
        for index, region in enumerate(cropping_regions):
            expected = expected_cropping_regions[index]
            self.assertEqual(
                (expected.x, expected.y, expected.w, expected.h),
                (region.x, region.y, region.w, region.h)
            )

    def test_crop_multiple_compare(self):
        crop_parameters = [(0, 0, 512, 512), (512, 0, 512, 512)]
        singe_crops = [
            self.jpeg.crop(self.buffer, *crop_parameter)
            for crop_parameter in crop_parameters
        ]
        multiple_crops = self.jpeg.crop_multiple(self.buffer, crop_parameters)
        self.assertEqual(singe_crops, multiple_crops)

    def test_crop_multiple_extend(self):
        crop_parameters = [(0, 0, 1024, 1024)]
        crop = self.jpeg.crop_multiple(self.buffer, crop_parameters)[0]
        width, height, _, _ = self.jpeg.decode_header(crop)
        self.assertEqual((1024, 1024), (width, height))

    def test_fill_background(self):
        mcu_size = 64
        original_width = 8
        original_height = 8
        extended_width = 16
        extended_height = 16
        callback_row_heigth = 8
        background_luminance = 255
        gray = False
        componentID = 0
        transformID = 0

        crop_region = CroppingRegion(0, 0, extended_width, extended_height)

        # Create coefficent array, filled with 0:s. The data is arranged in
        # mcus, i.e. first 64 values are for mcu (0, 0), second 64 values for
        # mcu (1, 0)
        coeffs = np.zeros(extended_width*extended_height, dtype=c_short)
        # Fill the mcu corresponding to the original image with 1:s.
        coeffs[0:original_width*original_height] = 1

        # Make a copy of the original data and change the coefficents for the
        # extended mcus ((0, 0), (1, 0), (1, 1)) manually.
        expected_results = np.copy(coeffs)
        for index in range(mcu_size, extended_width*extended_height, mcu_size):
            expected_results[index] = background_luminance

        planeRegion = CroppingRegion(0, 0, extended_width, extended_width)

        transform_struct = TransformStruct(
            crop_region,
            TJXOP_NONE,
            TJXOPT_PERFECT | TJXOPT_CROP | (gray and TJXOPT_GRAY),
            pointer(BackgroundStruct(
                original_width,
                original_height,
                0,
                background_luminance
            )),
            CUSTOMFILTER(fill_background)
        )

        # Iterate the callback with one mcu-row of data.
        for row in range(extended_height//callback_row_heigth):
            data_start = row * callback_row_heigth * extended_width
            data_end = (row+1) * callback_row_heigth * extended_width
            arrayRegion = CroppingRegion(
                0,
                row*callback_row_heigth,
                extended_width,
                callback_row_heigth
            )
            callback_result = fill_background(
                coeffs[data_start:data_end].ctypes.data,
                arrayRegion,
                planeRegion,
                componentID,
                transformID,
                pointer(transform_struct)
            )

        # Compare the modified data with the expected result
        self.assertTrue(np.array_equal(expected_results, coeffs))

    def test_blank_background(self):
        mcu_size = 64
        original_width = 0
        original_height = 0
        extended_width = 16
        extended_height = 16
        callback_row_heigth = 8
        background_luminance = 508
        gray = False
        componentID = 0
        transformID = 0

        crop_region = CroppingRegion(0, 0, extended_width, extended_height)

        # Create coefficent array, filled with 1:s. The data is arranged in
        # mcus, i.e. first 64 values are for mcu (0, 0), second 64 values for
        # mcu (1, 0)
        coeffs = np.ones(extended_width*extended_height, dtype=c_short)

        # The expected result is fileld with 0:s and luminance dc component
        # changed
        expected_results = np.zeros(
            extended_width*extended_height, dtype=c_short
        )
        for index in range(0, extended_width*extended_height, mcu_size):
            expected_results[index] = background_luminance
        planeRegion = CroppingRegion(0, 0, extended_width, extended_width)

        transform_struct = TransformStruct(
            crop_region,
            TJXOP_NONE,
            TJXOPT_PERFECT | TJXOPT_CROP | (gray and TJXOPT_GRAY),
            pointer(BackgroundStruct(
                original_width,
                original_height,
                0,
                background_luminance
            )),
            CUSTOMFILTER(fill_whole_image_with_background)
        )

        # Iterate the callback with one mcu-row of data.
        for row in range(extended_height//callback_row_heigth):
            data_start = row * callback_row_heigth * extended_width
            data_end = (row+1) * callback_row_heigth * extended_width
            arrayRegion = CroppingRegion(
                0,
                row*callback_row_heigth,
                extended_width,
                callback_row_heigth
            )
            callback_result = fill_whole_image_with_background(
                coeffs[data_start:data_end].ctypes.data,
                arrayRegion,
                planeRegion,
                componentID,
                transformID,
                pointer(transform_struct)
            )

        # Compare the modified data with the expected result
        self.assertTrue(np.array_equal(expected_results, coeffs))
