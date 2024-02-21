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
from io import BufferedReader
import numpy as np
import pytest
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

from opentile.jpeg.turbojpeg_patch import (
    BlankImage,
    BlankStruct,
    TurboJPEG_patch,
    find_turbojpeg_path,
)

test_file_path = "tests/testdata/turbojpeg/frame_2048x512.jpg"


@pytest.fixture()
def jpeg():
    yield TurboJPEG_patch(find_turbojpeg_path())


@pytest.fixture()
def test_file():
    with open(test_file_path, "rb") as file:
        yield file


@pytest.fixture()
def buffer(test_file: BufferedReader):
    yield test_file.read()


@pytest.mark.turbojpeg
class TestTurboJpeg:

    @pytest.mark.parametrize(
        ["region", "background", "expected_result"],
        [
            [CroppingRegion(0, 0, 512, 512), 1.0, False],
            [CroppingRegion(0, 0, 2048, 1024), 1.0, False],
            [CroppingRegion(1024, 0, 1024, 1024), 1.0, False],
            [CroppingRegion(0, 0, 2048, 2048), 1.0, True],
            [CroppingRegion(0, 0, 2048, 2048), 0.5, False],
        ],
    )
    def test_need_fill_background(
        self,
        jpeg: TurboJPEG_patch,
        region: CroppingRegion,
        background: float,
        expected_result: bool,
    ):
        # Arrange
        image_size = (2048, 1024)

        # Act
        result = jpeg._need_fill_background(region, image_size, background)

        # Assert
        assert result == expected_result

    def test_define_cropping_regions(self, jpeg: TurboJPEG_patch):
        # Arrange
        crop_parameters = [(0, 1, 2, 3), (4, 5, 6, 7)]
        expected_cropping_regions = [
            CroppingRegion(0, 1, 2, 3),
            CroppingRegion(4, 5, 6, 7),
        ]

        # Act
        cropping_regions = jpeg._define_cropping_regions(crop_parameters)

        # Assert
        for index, region in enumerate(cropping_regions):
            expected = expected_cropping_regions[index]
            assert (expected.x, expected.y, expected.w, expected.h) == (
                region.x,
                region.y,
                region.w,
                region.h,
            )

    def test_crop_multiple_compare(self, jpeg: TurboJPEG_patch, buffer: bytes):
        # Arrange
        crop_parameters = [(0, 0, 512, 512), (512, 0, 512, 512)]
        single_crops = [
            jpeg.crop(buffer, *crop_parameter) for crop_parameter in crop_parameters
        ]

        # Act
        multiple_crops = jpeg.crop_multiple(buffer, crop_parameters)

        # Assert
        assert single_crops == multiple_crops

    def test_crop_multiple_extend(self, jpeg: TurboJPEG_patch, buffer: bytes):
        # Arrange
        crop_parameters = [(0, 0, 1024, 1024)]

        # Act
        crop = jpeg.crop_multiple(buffer, crop_parameters)[0]

        # Assert
        width, height, _, _ = jpeg.decode_header(crop)
        assert (1024, 1024) == (width, height)

    def test_fill_background(self):
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

    def test_blank_background(self):
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
