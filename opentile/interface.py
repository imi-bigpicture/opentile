from pathlib import Path
from typing import Tuple

from tifffile import TiffFile

from opentile.common import Tiler
from opentile.ndpi_tiler import NdpiTiler
from opentile.phillips_tiff_tiler import PhillipsTiffTiler
from opentile.svs_tiler import SvsTiler


class OpenTile:
    @classmethod
    def open(
        self,
        filepath: str,
        tile_size: Tuple[int, int] = None,
        turbo_path: Path = None
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
        turbo_path: Path = None
            Path to turbo jpeg library, if needed for transforming tiles for
            file format.
        """
        tiff_file = TiffFile(filepath)
        if tiff_file.is_ndpi:
            if tile_size is None or turbo_path is None:
                raise ValueError("Tile size and turbo path needed for ndpi")
            return NdpiTiler(
                tiff_file,
                tile_size,
                turbo_path
            )

        if tiff_file.is_svs:
            return SvsTiler(
                tiff_file
            )

        if tiff_file.is_phillips:
            if turbo_path is None:
                raise ValueError("Turbo path needed for phillips tiff")
            return PhillipsTiffTiler(
                tiff_file,
                turbo_path
            )

        raise NotImplementedError('Non supported tiff file')
