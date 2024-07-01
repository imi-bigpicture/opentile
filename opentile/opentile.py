#    Copyright 2021-2024 SECTRA AB
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

"""Main interface for OpenTile."""

from pathlib import Path
from typing import (
    Dict,
    Iterator,
    Optional,
    Tuple,
    Type,
    Union,
)

from tifffile import TiffFileError

from opentile.file import OpenTileFile
from opentile.formats import (
    HistechTiffTiler,
    NdpiTiler,
    OmeTiffTiler,
    PhilipsTiffTiler,
    SvsTiler,
)
from opentile.tiler import Tiler
from upath import UPath


class OpenTile:
    _tilers: Dict[str, Type[Tiler]] = {
        "ndpi": NdpiTiler,
        "svs": SvsTiler,
        "phillips tiff": PhilipsTiffTiler,
        "3dhistech tiff": HistechTiffTiler,
        "ome-tiff tiler": OmeTiffTiler,
    }

    @classmethod
    def open(
        cls,
        filepath: Union[str, Path, UPath],
        tile_size: int = 512,
        file_options: Optional[Dict[str, str]] = None,
        turbo_path: Optional[Union[str, Path]] = None,
    ) -> Tiler:
        """Return a file type specific tiler for tiff file in filepath.
        Tile size and turbo jpeg path are optional but required for some file
        types.

        Parameters
        ----------
        filepath: Union[str, Path, UPath]
            Path to tiff file.
        tile_size: int = 512
            Tile size for creating tiles, if needed for file format.
        file_options: Optional[Dict[str, str]] = None
            Options to pass to filesystem when opening file.
        turbo_path: Optional[Union[str, Path]] = None
            Path to turbojpeg (dll or so).
        """
        try:
            file = OpenTileFile(filepath, file_options)
        except TiffFileError as exception:
            raise NotImplementedError(
                f"File {filepath} failed to open with TiffFile", exception
            )
        _, supported_tiler = next(cls._get_supported_tilers(file), (None, None))
        if supported_tiler is NdpiTiler:
            return NdpiTiler(file, tile_size, turbo_path)
        if supported_tiler is SvsTiler:
            return SvsTiler(file, turbo_path)
        if supported_tiler is PhilipsTiffTiler:
            return PhilipsTiffTiler(file, turbo_path)
        if supported_tiler is HistechTiffTiler:
            return HistechTiffTiler(file)
        if supported_tiler is OmeTiffTiler:
            return OmeTiffTiler(file)
        raise NotImplementedError(f"Support for tiff file {filepath} not implemented.")

    @classmethod
    def detect_format(
        cls,
        filepath: Union[str, Path, UPath],
        file_options: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:

        try:
            with OpenTileFile(filepath, file_options) as file:
                tiler_name, _ = next(cls._get_supported_tilers(file), (None, None))
                return tiler_name
        except TiffFileError:
            return None

    @classmethod
    def _get_supported_tilers(
        cls, file: OpenTileFile
    ) -> Iterator[Tuple[str, Type[Tiler]]]:
        return (
            (tiler_name, tiler)
            for (tiler_name, tiler) in cls._tilers.items()
            if tiler.supported(file.tiff)
        )
