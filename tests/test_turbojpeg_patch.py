import unittest

import pytest
from ndpi_tiler.turbojpeg_patch import TurboJPEG_patch as TurboJPEG
from turbojpeg import CroppingRegion

turbo_path = 'C:/libjpeg-turbo64/bin/turbojpeg.dll'
test_file_path = 'C:/temp/ndpi/turbojpeg/frame_1024x512.jpg'


@pytest.mark.unittest
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
                image_size,
                512
            )
        )

        crop_region = CroppingRegion(0, 0, 2048, 1024)
        self.assertFalse(
            self.jpeg._TurboJPEG_patch__need_fill_background(
                crop_region,
                image_size,
                512
            )
        )

        crop_region = CroppingRegion(1024, 0, 1024, 1024)
        self.assertFalse(
            self.jpeg._TurboJPEG_patch__need_fill_background(
                crop_region,
                image_size,
                512
            )
        )

        crop_region = CroppingRegion(0, 0, 2048, 2048)
        self.assertTrue(
            self.jpeg._TurboJPEG_patch__need_fill_background(
                crop_region,
                image_size,
                512
            )
        )

        crop_region = CroppingRegion(0, 0, 2048, 2048)
        self.assertFalse(
            self.jpeg._TurboJPEG_patch__need_fill_background(
                crop_region,
                image_size,
                0
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
