from pathlib import Path
from typing import Optional, Tuple, Union

from tifffile import TiffFile, TiffFileError

from opentile.common import Tiler
from opentile.ndpi_tiler import NdpiTiler
from opentile.philips_tiff_tiler import PhilipsTiffTiler
from opentile.svs_tiler import SvsTiler
from opentile.turbojpeg_patch import find_turbojpeg_path


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
        tile_size: Union[int, Tuple[int, int]] = None,
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
        if not isinstance(tile_size, tuple):
            tile_size = (tile_size, tile_size)
        file_format = cls.detect_format(filepath)
        if file_format == 'ndpi':
            if tile_size is None:
                raise ValueError("Tile size needed for ndpi")
            return NdpiTiler(
                filepath,
                tile_size,
                find_turbojpeg_path()
            )

        if file_format == 'svs':
            return SvsTiler(
                filepath
            )

        if file_format == 'philips_tiff':
            return PhilipsTiffTiler(
                filepath,
                find_turbojpeg_path()
            )

        raise NotImplementedError('Non supported tiff file')
