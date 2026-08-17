#    Copyright 2022-2024 SECTRA AB
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

"""Download the WSI test slides used by the test suite from their source hosts.
"""

import os
from hashlib import sha256
from pathlib import Path
from typing import Any

import requests

FILES: dict[str, dict[str, Any]] = {
    "svs/CMU-1/CMU-1.svs": {
        "url": "https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1.svs",
        "description": "Brightfield, JPEG",
        "license": "CC0-1.0",
        "sha256": "00a3d54482cd707abf254fe69dccc8d06b8ff757a1663f1290c23418c480eb30",
    },
    "ndpi/CMU-1/CMU-1.ndpi": {
        "url": "https://openslide.cs.cmu.edu/download/openslide-testdata/Hamamatsu/CMU-1.ndpi",
        "description": "Small scan with valid JPEG headers, brightfield, circa 2009",
        "license": "CC0-1.0",
        "sha256": "edf4a1ccf395c7000ae93ad3b44c07d97043810e00be0c1d167dd09bbe436e46",
    },
    "huron/Huron-1/Huron-1.tif": {
        "url": "https://openslide.cs.cmu.edu/download/openslide-testdata/Huron/Huron-1.tif",
        "description": "H&E stain, brightfield, 20x objective, JPEG, scanned on Huron LE176",
        "license": "CC0-1.0",
        "credit": "Huron Digital Pathology",
        "sha256": "295506f9872becce44033baa830dfd7a7644d104a832e6a7f73565be990c465d",
    },
    "scn/scn1/Leica-1.scn": {
        "url": "https://openslide.cs.cmu.edu/download/openslide-testdata/Leica/Leica-1.scn",
        "description": "Brightfield, single ROI, 2010/10/01 schema",
        "license": "distributable",
        "credit": "Yves Sucaet",
        "sha256": "63a3c00fef5215497a9725cf19092a93f7f5ff855ce8561af74b087631831a97",
    },
    "scn/scn2/Leica-2.scn": {
        "url": "https://openslide.cs.cmu.edu/download/openslide-testdata/Leica/Leica-2.scn",
        "description": "Mouse kidney, H&E stain, brightfield, multiple ROIs with identical resolutions, 2010/10/01 schema",
        "license": "distributable",
        "credit": "Ira Hensen, CECAD Imaging Facility, Cologne",
        "sha256": "f10523f88afac728f6d7d18ba39926be56766b739e68f00cace6ceea5c6c2cea",
    },
    "scn/scn3/Leica-3.scn": {
        "url": "https://openslide.cs.cmu.edu/download/openslide-testdata/Leica/Leica-3.scn",
        "description": "Mouse kidney, H&E stain, brightfield, multiple ROIs with different resolutions, 2010/10/01 schema",
        "license": "distributable",
        "credit": "Ira Hensen, CECAD Imaging Facility, Cologne",
        "sha256": "5ec8867f3edf90e685b8436e515bc24dbbefae545fb556dab568d7691592849b",
    },
    "argos/Argos-1/Argos-1.avs": {
        "url": "https://openslide.cs.cmu.edu/download/openslide-testdata/Argos/Argos-1.avs",
        "description": "Brightfield",
        "license": "CC0-1.0",
        "sha256": "3b68b5be6270344ad1823bb6c74c242d60493be90926002f6cdf6e96ca8ff28d",
    },
    "argos/Argos-1-Stacked/Argos-1-Stacked.avs": {
        "url": "https://openslide.cs.cmu.edu/download/openslide-testdata/Argos/Argos-1-Stacked.avs",
        "description": "Brightfield, Z-stack",
        "license": "CC0-1.0",
        "sha256": "1b88645342040b62bb3c3feaa1db76f9419cd7749327690b1221475a4d04c34a",
    },
    "qptiff/HandEcompressed/HandEcompressed_Scan1.qptiff": {
        "url": "https://downloads.openmicroscopy.org/images/Vectra-QPTIFF/perkinelmer/PKI_scans/HandEcompressed_Scan1.qptiff",
        "description": "Brightfield, RGB, JPEG compression",
        "license": "CC-BY-4.0",
        "credit": "PerkinElmer",
        "attribution": "(c) PerkinElmer (http://www.perkinelmer.com). Licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/. The image content is unchanged.",
        "sha256": "e2f5fae28409da7ca208ac0d53af988696313df67c3b0b6d6e0ac3ce7dfbe0d9",
    },
    "qptiff/LuCa-7color/LuCa-7color_Scan1.qptiff": {
        "url": "https://downloads.openmicroscopy.org/images/Vectra-QPTIFF/perkinelmer/PKI_scans/LuCa-7color_Scan1.qptiff",
        "description": "Fluorescence, 5 layers",
        "license": "CC-BY-4.0",
        "credit": "PerkinElmer",
        "attribution": "(c) PerkinElmer (http://www.perkinelmer.com). Licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/. The image content is unchanged.",
        "sha256": "0a0558670a479d826f13e467fd7e50cb6d691c9b4009d86bf579d1a0f3c526dc",
    },
}

DEFAULT_SLIDE_FOLDER = "tests/testdata/slides"
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
HASH_CHUNK_SIZE = 1024 * 1024


def download_file(url: str, filename: Path):
    with requests.get(url, stream=True, timeout=30) as request:
        request.raise_for_status()
        with open(filename, "wb") as file:
            for chunk in request.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                file.write(chunk)


def file_sha256(path: Path) -> str:
    """Return the sha256 of a file, read in chunks so large files are not held in
    memory."""
    hasher = sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(HASH_CHUNK_SIZE), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    print("Downloading and/or checking testdata.")
    test_data_folder = os.environ.get("OPENTILE_TESTDIR")
    if test_data_folder is None:
        slide_folder = Path(DEFAULT_SLIDE_FOLDER)
        print(
            'Env "OPENTILE_TESTDIR"" not set, downloading to default folder '
            f"{slide_folder}."
        )
    else:
        slide_folder = Path(test_data_folder).joinpath("slides")
        print(f"Downloading to {slide_folder}")
    os.makedirs(slide_folder, exist_ok=True)
    for file, file_settings in FILES.items():
        file_path = slide_folder.joinpath(file)
        if file_path.exists():
            print(f"{file} found, skipping download")
        else:
            url = file_settings["url"]
            print(f"{file} not found, downloading from {url}")
            os.makedirs(file_path.parent, exist_ok=True)
            download_file(url, file_path)

        if file_sha256(file_path) != file_settings["sha256"]:
            raise ValueError(f"Checksum failed for {file_path}")
        print(f"{file_path} checksum OK")


if __name__ == "__main__":
    main()
