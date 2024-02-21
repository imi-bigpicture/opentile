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
from typing import Any, BinaryIO, Dict, Optional, Tuple, Type, Union

from tifffile import TiffFile, TiffFileError

from opentile.formats import (
    HistechTiffTiler,
    NdpiTiler,
    OmeTiffTiler,
    PhilipsTiffTiler,
    SvsTiler,
)
from fsspec.core import open
from opentile.tiler import Tiler
from upath import UPath


class OpenTile:
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
        format, supported_tiler = cls.get_tiler(filepath, file_options)
        if supported_tiler is NdpiTiler:
            return NdpiTiler(filepath, tile_size, turbo_path)

        if supported_tiler is SvsTiler:
            return SvsTiler(filepath, turbo_path)

        if supported_tiler is PhilipsTiffTiler:
            return PhilipsTiffTiler(filepath, turbo_path)

        if supported_tiler is HistechTiffTiler:
            return HistechTiffTiler(filepath)

        if supported_tiler is OmeTiffTiler:
            return OmeTiffTiler(filepath)

        raise NotImplementedError("Non supported tiff file")

    @staticmethod
    def get_tiler(
        filepath: Union[str, Path, UPath],
        file_options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[str], Optional[Type[Tiler]]]:
        """Return tiler that supports the tiff file in filepath, or None if
        not supported"""
        tilers: Dict[str, Type[Tiler]] = {
            "ndpi": NdpiTiler,
            "svs": SvsTiler,
            "phillips tiff": PhilipsTiffTiler,
            "3dhistech tiff": HistechTiffTiler,
            "ome-tiff tiler": OmeTiffTiler,
        }
        try:
            file: BinaryIO = open(str(filepath), **file_options or {})  # type: ignore
            tiff_file = TiffFile(file)
            return next(
                (tiler_name, tiler)
                for (tiler_name, tiler) in tilers.items()
                if tiler.supported(tiff_file)
            )
        except (TiffFileError, StopIteration):
            return None, None

    @classmethod
    def detect_format(
        cls,
        filepath: Union[str, Path, UPath],
        file_options: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        format, supported_tiler = cls.get_tiler(filepath, file_options)
        return format
