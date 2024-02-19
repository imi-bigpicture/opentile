#    Copyright 2022-2023 SECTRA AB
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

import os
from pathlib import Path
from typing import Any, Dict
import requests
from hashlib import md5

FILES: Dict[str, Dict[str, Any]] = {
    "svs/CMU-1/CMU-1.svs": {
        "url": "https://data.cytomine.coop/open/openslide/aperio-svs/CMU-1.svs",  # NOQA
        "md5": {"CMU-1.svs": "751b0b86a3c5ff4dfc8567cf24daaa85"},
    },
    "ndpi/CMU-1/CMU-1.ndpi": {
        "url": "https://data.cytomine.coop/open/openslide/hamamatsu-ndpi/CMU-1.ndpi",  # NOQA
        "md5": {"CMU-1.ndpi": "fb89dea54f85fb112e418a3cf4c7888a"},
    },
}

DEFAULT_SLIDE_FOLDER = "tests/testdata/slides"
DOWNLOAD_CHUNK_SIZE = 8192


def download_file(url: str, filename: Path):
    with requests.get(url, stream=True) as request:
        request.raise_for_status()
        with open(filename, "wb") as file:
            for chunk in request.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                file.write(chunk)


def main():
    print("Downloading and/or checking testdata from cytomine.")

    slide_folder = os.environ.get("OPENTILE_TESTDIR")
    if slide_folder is None:
        slide_folder = Path(DEFAULT_SLIDE_FOLDER)
        print(
            'Env "OPENTILE_TESTDIR"" not set, downloading to default folder '
            f"{slide_folder}."
        )
    else:
        slide_folder = Path(slide_folder)
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

        for relative_path, hash in file_settings["md5"].items():
            saved_file_path = file_path.parent.joinpath(relative_path)
            if not saved_file_path.exists():
                raise ValueError(
                    f"Did not find {saved_file_path}. Try removing the "
                    "parent folder and try again."
                )
            with open(saved_file_path, "rb") as saved_file_io:
                data = saved_file_io.read()
                if not hash == md5(data).hexdigest():
                    raise ValueError(f"Checksum failed for {saved_file_path}")
                else:
                    print(f"{saved_file_path} checksum OK")


if __name__ == "__main__":
    main()
