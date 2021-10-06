import os
from pathlib import Path
from typing import Optional, Tuple

from tifffile import TiffFile, TiffFileError

from opentile.common import Tiler
from opentile.ndpi_tiler import NdpiTiler
from opentile.philips_tiff_tiler import PhilipsTiffTiler
from opentile.svs_tiler import SvsTiler


class OpenTile:
    @staticmethod
    def detect_format(filepath: Path) -> Optional[str]:
        """Return string describing tiff file format in file, or None
        if not supported."""
        try:
            tiff_file = TiffFile(filepath)
            if tiff_file.is_ndpi:
                file_format = 'ndpi'
            elif tiff_file.is_svs:
                file_format = 'svs'
            elif tiff_file.is_philips:
                file_format = 'philips_tiff'
            else:
                file_format = None
        except TiffFileError:
            file_format = None
        return file_format

    @classmethod
    def open(
        cls,
        filepath: Path,
        tile_size: Tuple[int, int] = None,
    ) -> Tiler:
        """Return a file type specific tiler for tiff file in filepath.
        Tile size and turbo jpeg path are optional but required for some file
        types.

        Parameters
        ----------
        filepath: str
            Path to tiff file.
        tile_size: Tuple[int, int] = None
            Tile size for creating tiles, if needed for file format.
        """
        file_format = cls.detect_format(filepath)
        if file_format == 'ndpi':
            if tile_size is None:
                raise ValueError("Tile size needed for ndpi")
            return NdpiTiler(
                filepath,
                tile_size,
                cls.find_turbojpeg_path()
            )

        if file_format == 'svs':
            return SvsTiler(
                filepath
            )

        if file_format == 'philips_tiff':
            return PhilipsTiffTiler(
                filepath,
                cls.find_turbojpeg_path()
            )

        raise NotImplementedError('Non supported tiff file')

    @staticmethod
    def find_turbojpeg_path() -> Optional[Path]:
        # Only windows installs libraries on strange places
        if os.name != 'nt':
            return None
        try:
            bin_path = Path(os.environ['TURBOJPEG'])
        except KeyError:
            raise ValueError(
                "Enviroment variable 'TURBOJPEG' "
                "needs to be set to turbojpeg bin path."
            )
        if not bin_path.is_dir():
            raise ValueError(
                "Enviroment variable 'TURBOJPEG' "
                "is not set to a directory."
            )
        try:
            dll_file = [
                file for file in bin_path.iterdir()
                if file.is_file()
                and 'turbojpeg' in file.name
                and file.suffix == '.dll'
            ][0]
        except IndexError:
            raise ValueError(
                f'Could not find turbojpeg dll in {bin_path}.'
            )
        return dll_file
