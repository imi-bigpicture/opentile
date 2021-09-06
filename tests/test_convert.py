import glob
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Tuple, TypedDict

import pydicom
import pytest
from ndpi_tiler import NdpiFileImporter, __version__
from PIL import Image, ImageChops
from wsidicom import WsiDataset, WsiDicom

ndpi_test_data_dir = os.environ.get("NDPI_TESTDIR", "C:/temp/ndpi")
sub_data_dir = "convert"
ndpi_data_dir = ndpi_test_data_dir + '/' + sub_data_dir
uids = [pydicom.uid.generate_uid() for i in range(10)]

include_series = {
    'VOLUME': (0, {
        3: uids[0],
        4: uids[1]
    })
}


class WsiFolder(TypedDict):
    path: Path
    wsi_dicom: WsiDicom


@pytest.mark.convert
class NdpiConvertTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_folders: Dict[Path, Tuple[WsiDicom, TemporaryDirectory]]
        self.tile_size: Tuple[int, int]

    @classmethod
    def setUpClass(cls):
        cls.tile_size = (1024, 1024)
        cls.test_folders = {}
        folders = cls._get_folders()
        for folder in folders:
            cls.test_folders[folder] = cls.open(folder)

    @classmethod
    def tearDownClass(cls):
        for (wsi, tempdir) in cls.test_folders.values():
            wsi.close()
            tempdir.cleanup()

    @classmethod
    def open(cls, path: Path) -> Tuple[WsiDicom, TemporaryDirectory]:
        folder = Path(path).joinpath('ndpi/input.ndpi')

        base_dataset = WsiDataset.create_test_base_dataset()
        file_importer = NdpiFileImporter(
            folder,
            base_dataset,
            include_series,
            cls.tile_size,
            'C:/libjpeg-turbo64/bin/turbojpeg.dll'
        )
        tempdir = TemporaryDirectory()
        WsiDicom.convert(Path(tempdir.name), file_importer)
        wsi = WsiDicom.open(str(tempdir.name))
        return (wsi, tempdir)

    @classmethod
    def _get_folders(cls):
        return [
            Path(ndpi_data_dir).joinpath(item)
            for item in os.listdir(ndpi_data_dir)
        ]

    def test_read_region(self):
        for folder, (wsi, tempdir) in self.test_folders.items():
            json_files = glob.glob(
                str(folder.absolute())+"/read_region/*.json")

            for json_file in json_files:
                with open(json_file, "rt") as f:
                    region = json.load(f)

                im = wsi.read_region(
                    (region["location"]["x"], region["location"]["y"]),
                    region["level"],
                    (region["size"]["width"], region["size"]["height"])
                )

                expected_im = Image.open(Path(json_file).with_suffix(".png"))

                diff = ImageChops.difference(im, expected_im)

                bbox = diff.getbbox()
                self.assertIsNone(bbox, msg=json_file)

    def test_read_thumbnail(self):
        for folder, (wsi, tempdir) in self.test_folders.items():
            json_files = glob.glob(
                str(folder.absolute())+"/read_thumbnail/*.json")

            for json_file in json_files:
                with open(json_file, "rt") as f:
                    region = json.load(f)
                im = wsi.read_thumbnail(
                    (region["size"]["width"], region["size"]["height"])
                )
                expected_im = Image.open(Path(json_file).with_suffix(".png"))

                diff = ImageChops.difference(im, expected_im)

                bbox = diff.getbbox()
                self.assertIsNone(bbox, msg=json_file)
