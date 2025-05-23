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

import os
from pathlib import Path

test_data_dir = os.environ.get("OPENTILE_TESTDIR", "tests/testdata")
slide_folder = Path(test_data_dir).joinpath("slides")
svs_file_path = slide_folder.joinpath("svs/CMU-1/CMU-1.svs")
svs_z_file_path = slide_folder.joinpath("svs/zstack1/zstack1.svs")
philips_file_path = slide_folder.joinpath("philips_tiff/philips1/input.tif")
ndpi_file_path = slide_folder.joinpath("ndpi/CMU-1/CMU-1.ndpi")
ndpi_z_file_path = slide_folder.joinpath("ndpi/zstack1/zstack1.ndpi")
histech_file_path = slide_folder.joinpath(
    "3dhistech_tiff/CMU-1/CMU-1_Default_Extended.tif"
)
ome_tiff_file_path = slide_folder.joinpath("ome_tiff/CMU-1/CMU-1.ome.tiff")
