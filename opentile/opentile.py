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

from collections.abc import Iterator
from pathlib import Path
from typing import (
    Optional,
    Union,
)

from upath import UPath

from opentile.file import OpenTileFile
from opentile.formats import (
    ArgosTiffTiler,
    HistechTiffTiler,
    HuronTiffTiler,
    MikroscanTiffTiler,
    MoticTiffTiler,
    NdpiTiler,
    OmeTiffTiler,
    PhilipsTiffTiler,
    SvsTiler,
    TrestleTiffTiler,
    VentanaTiffTiler,
)
from opentile.tiff_format import TiffFormat
from opentile.tiler import Tiler


class OpenTile:
    _tilers: dict[TiffFormat, type[Tiler]] = {
        TiffFormat.NDPI: NdpiTiler,
        TiffFormat.SVS: SvsTiler,
        TiffFormat.PHILIPS_TIFF: PhilipsTiffTiler,
        TiffFormat.HISTECH_TIFF: HistechTiffTiler,
        TiffFormat.OME_TIFF: OmeTiffTiler,
        TiffFormat.TRESTLE: TrestleTiffTiler,
        TiffFormat.VENTANA: VentanaTiffTiler,
        TiffFormat.HURON: HuronTiffTiler,
        TiffFormat.MIKROSCAN: MikroscanTiffTiler,
        TiffFormat.MOTIC: MoticTiffTiler,
        TiffFormat.ARGOS: ArgosTiffTiler,
    }

    @classmethod
    def open(
        cls,
        filepath: Union[str, Path, UPath],
        tile_size: int = 512,
        file_options: Optional[dict[str, str]] = None,
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
        file = OpenTileFile(filepath, file_options)
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
            return OmeTiffTiler(file, tile_size, turbo_path)
        if supported_tiler is TrestleTiffTiler:
            return TrestleTiffTiler(file)
        if supported_tiler is VentanaTiffTiler:
            return VentanaTiffTiler(file, turbo_path)
        if supported_tiler is HuronTiffTiler:
            return HuronTiffTiler(file)
        if supported_tiler is MikroscanTiffTiler:
            return MikroscanTiffTiler(file)
        if supported_tiler is MoticTiffTiler:
            return MoticTiffTiler(file, turbo_path)
        if supported_tiler is ArgosTiffTiler:
            return ArgosTiffTiler(file, turbo_path)
        raise NotImplementedError(f"Support for tiff file {filepath} not implemented.")

    @classmethod
    def detect_format(
        cls,
        filepath: Union[str, Path, UPath],
        file_options: Optional[dict[str, str]] = None,
    ) -> Optional[TiffFormat]:

        try:
            with OpenTileFile(filepath, file_options) as file:
                tiff_format, _ = next(cls._get_supported_tilers(file), (None, None))
                return tiff_format
        except Exception:
            return None

    @classmethod
    def _get_supported_tilers(
        cls, file: OpenTileFile
    ) -> Iterator[tuple[TiffFormat, type[Tiler]]]:
        return (
            (tiff_format, tiler)
            for (tiff_format, tiler) in cls._tilers.items()
            if tiler.supported(file.tiff)
        )
