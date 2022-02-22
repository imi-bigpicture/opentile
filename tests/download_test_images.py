#    Copyright 2022 SECTRA AB
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
import requests
from hashlib import md5

SVS_PATH = 'slides/svs/CMU-1/CMU-1.svs'
SVS_URL = 'https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1.svs'
SVS_MD5 = '751b0b86a3c5ff4dfc8567cf24daaa85'
NDPI_PATH = 'slides/ndpi/CMU-1/CMU-1.ndpi'
NDPI_URL = 'https://openslide.cs.cmu.edu/download/openslide-testdata/Hamamatsu/CMU-1.ndpi'
NDPI_MD5 = 'fb89dea54f85fb112e418a3cf4c7888a'
DEFAULT_DIR = 'testdata'
DOWNLOAD_CHUNK_SIZE=8192

def download_file(url: str, filename: Path):
    with requests.get(url, stream=True) as request:
        request.raise_for_status()
        with open(filename, 'wb') as file:
            for chunk in request.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                file.write(chunk)

def main():
    print("Downloading and/or checking testdata from openslide.")
    test_data_path = os.environ.get("OPENTILE_TESTDIR")
    if test_data_path is None:
        test_data_dir = Path(DEFAULT_DIR)
        print(
            "Env 'OPENTILE_TESTDIR' not set, downloading to default folder "
            f"{test_data_dir}."
        )
    else:
        test_data_dir = Path(test_data_path)
        print(f"Downloading to {test_data_dir}")
    os.makedirs(test_data_dir, exist_ok=True)
    files = {
        test_data_dir.joinpath(SVS_PATH): (SVS_URL, SVS_MD5),
        test_data_dir.joinpath(NDPI_PATH): (NDPI_URL, NDPI_MD5)
    }
    for file, (url, checksum) in files.items():
        if not file.exists():
            print(f"{file} not found, downloading from {url}")
            os.makedirs(file.parent, exist_ok=True)
            download_file(url, file)
        else:
            print(f"{file} found, skipping download")
        with open(file, 'rb') as saved_file:
            data = saved_file.read()
            if not checksum == md5(data).hexdigest():
                raise ValueError(f"Checksum faild for {file}")
            else:
                print(f"{file} checksum OK")

if __name__ == "__main__":
    main()