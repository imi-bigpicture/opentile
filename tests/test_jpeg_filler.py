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

from ctypes import c_short, pointer
from io import BytesIO

import numpy as np
import pytest
from PIL import Image
from turbojpeg import (
    CUSTOMFILTER,
    TJXOP_NONE,
    TJXOPT_CROP,
    TJXOPT_GRAY,
    TJXOPT_PERFECT,
    BackgroundStruct,
    CroppingRegion,
    TransformStruct,
    fill_background,
)

from opentile.jpeg.jpeg import find_turbojpeg_path
from opentile.jpeg.jpeg_filler import (
    BlankImage,
    BlankStruct,
    JpegFiller,
)

test_file_path = "tests/testdata/turbojpeg/frame_2048x512.jpg"


@pytest.fixture()
def filler():
    yield JpegFiller(find_turbojpeg_path())


@pytest.fixture()
def buffer():
    with open(test_file_path, "rb") as file:
        yield file.read()


@pytest.mark.unittest
class TestJpegFiller:

    @pytest.mark.parametrize(
        ["luminance", "expected_value"],
        [
            (0.0, 0),
            (0.5, 128),
            (1.0, 255),
        ],
    )
    def test_fill_image(
        self,
        filler: JpegFiller,
        buffer: bytes,
        luminance: float,
        expected_value: int,
    ):
        # Act
        filled = filler.fill_image(buffer, background_luminance=luminance)

        # Assert
        image = Image.open(BytesIO(filled))
        pixels = np.array(image.convert("L"))
        assert (pixels == expected_value).all()

    def test_fill_background_callback(self):
        # Arrange
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

        # Create coefficient array, filled with 0:s. The data is arranged in
        # mcus, i.e. first 64 values are for mcu (0, 0), second 64 values for
        # mcu (1, 0)
        coeffs = np.zeros(extended_width * extended_height, dtype=c_short)
        # Fill the mcu corresponding to the original image with 1:s.
        coeffs[0 : original_width * original_height] = 1

        # Make a copy of the original data and change the coefficients for the
        # extended mcus ((0, 0), (1, 0), (1, 1)) manually.
        expected_results = np.copy(coeffs)
        for index in range(mcu_size, extended_width * extended_height, mcu_size):
            expected_results[index] = background_luminance

        planeRegion = CroppingRegion(0, 0, extended_width, extended_width)

        transform_struct = TransformStruct(
            crop_region,
            TJXOP_NONE,
            TJXOPT_PERFECT | TJXOPT_CROP | (gray and TJXOPT_GRAY),
            pointer(
                BackgroundStruct(original_width, original_height, background_luminance)
            ),
            CUSTOMFILTER(fill_background),
        )

        # Act
        # Iterate the callback with one mcu-row of data.
        for row in range(extended_height // callback_row_heigth):
            data_start = row * callback_row_heigth * extended_width
            data_end = (row + 1) * callback_row_heigth * extended_width
            arrayRegion = CroppingRegion(
                0, row * callback_row_heigth, extended_width, callback_row_heigth
            )
            _ = fill_background(
                coeffs[data_start:data_end].ctypes.data,
                arrayRegion,
                planeRegion,
                componentID,
                transformID,
                pointer(transform_struct),
            )

        # Assert
        # Compare the modified data with the expected result
        assert np.array_equal(expected_results, coeffs)

    def test_blank_background_callback(self):
        # Arrange
        mcu_size = 64
        extended_width = 16
        extended_height = 16
        callback_row_heigth = 8
        background_luminance = 508
        transformID = 0
        blank_image_transform = BlankImage()

        crop_region = CroppingRegion(0, 0, extended_width, extended_height)

        # Create coefficient array, filled with 1:s. The data is arranged in
        # mcus, i.e. first 64 values are for mcu (0, 0), second 64 values for
        # mcu (1, 0)
        coeffs = np.ones(extended_width * extended_height, dtype=c_short)

        # The expected result is field with 0:s and luminance dc component
        # changed

        planeRegion = CroppingRegion(0, 0, extended_width, extended_width)

        transform_struct = blank_image_transform.transform(
            crop_region,
            BlankStruct(0, background_luminance),
        )

        # Act
        # Iterate through components
        for componentID in range(3):
            # Expected result is array with 0
            expected_results = np.zeros(extended_width * extended_height, dtype=c_short)
            # For luminance add background luminance to expected result
            if componentID == 0:
                for index in range(0, extended_width * extended_height, mcu_size):
                    expected_results[index] = background_luminance
            # Iterate the callback with one mcu-row of data.
            for row in range(extended_height // callback_row_heigth):
                data_start = row * callback_row_heigth * extended_width
                data_end = (row + 1) * callback_row_heigth * extended_width
                arrayRegion = CroppingRegion(
                    0, row * callback_row_heigth, extended_width, callback_row_heigth
                )
                _ = blank_image_transform.callback(
                    coeffs[data_start:data_end].ctypes.data,  # type: ignore
                    arrayRegion,
                    planeRegion,
                    componentID,
                    transformID,
                    pointer(transform_struct),  # type: ignore
                )

            # Assert
            # Compare the modified component with the expected result
            assert np.array_equal(expected_results, coeffs)
