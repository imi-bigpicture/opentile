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

from contextlib import contextmanager
from pathlib import Path
from typing import (
    Dict,
    Generator,
    Iterator,
    Optional,
    Tuple,
    Type,
    Union,
)

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
            with cls._open_tifffile(filepath, file_options) as tiff_file:
                _, supported_tiler = next(cls._get_supported_tilers(tiff_file))
                if supported_tiler is NdpiTiler:
                    return NdpiTiler(tiff_file, tile_size, turbo_path)
                if supported_tiler is SvsTiler:
                    return SvsTiler(tiff_file, turbo_path)
                if supported_tiler is PhilipsTiffTiler:
                    return PhilipsTiffTiler(tiff_file, turbo_path)
                if supported_tiler is HistechTiffTiler:
                    return HistechTiffTiler(tiff_file)
                if supported_tiler is OmeTiffTiler:
                    return OmeTiffTiler(tiff_file)
        except (TiffFileError, StopIteration):
            pass
        raise NotImplementedError("Non supported tiff file")

    @classmethod
    def detect_format(
        cls,
        filepath: Union[str, Path, UPath],
        file_options: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:

        try:
            with cls._open_tifffile(filepath, file_options) as tiff_file:
                tiler_name, _ = next(cls._get_supported_tilers(tiff_file))
                return tiler_name
        except (TiffFileError, StopIteration):
            return None

    @staticmethod
    @contextmanager
    def _open_tifffile(
        filepath: Union[str, Path, UPath], file_options: Optional[Dict[str, str]] = None
    ) -> Generator[TiffFile, None, None]:
        with open(str(filepath), **file_options or {}) as file:
            with TiffFile(file) as tiff_file:  # type: ignore
                yield tiff_file

    @classmethod
    def _get_supported_tilers(
        cls, tiff_file: TiffFile
    ) -> Iterator[Tuple[str, Type[Tiler]]]:
        return (
            (tiler_name, tiler)
            for (tiler_name, tiler) in cls._tilers.items()
            if tiler.supported(tiff_file)
        )
