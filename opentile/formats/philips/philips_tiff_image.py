#    Copyright 2021-2023 SECTRA AB
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

"""Image implementation for Philips tiff files."""

from typing import List, Optional

from tifffile import COMPRESSION, TiffPage

from opentile.file import LockableFileHandle
from opentile.geometry import Size, SizeMm
from opentile.jpeg import Jpeg
from opentile.tiff_image import NativeTiledTiffImage


class PhilipsTiffImage(NativeTiledTiffImage):
    def __init__(
        self,
        page: TiffPage,
        fh: LockableFileHandle,
        base_size: Size,
        base_mpp: SizeMm,
        jpeg: Jpeg,
    ):
        """OpenTiledPage for Philips Tiff image.

        Parameters
        ----------
        page: TiffPage
            TiffPage defining the page.
        fh: LockableFileHandle
            Filehandler to read data from.
        base_size: Size
            Size of base level in pyramid.
        base_mpp: SizeMm
            Mpp (um/pixel) for base level in pyramid.
        jpeg: Jpeg
            Jpeg instance to use.
        """
        super().__init__(page, fh)
        self._jpeg = jpeg
        self._base_size = base_size
        self._base_mpp = base_mpp
        self._pyramid_index = self._calculate_pyramidal_index(self._base_size)
        self._mpp = self._calculate_mpp(self._base_mpp)
        self._blank_tile = self._create_blank_tile()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._page}, {self._fh}, "
            f"{self._base_size}, {self._base_mpp}, {self._jpeg})"
        )

    @property
    def pixel_spacing(self) -> SizeMm:
        """Return pixel spacing in mm per pixel."""
        return self.mpp * 1000

    @property
    def supported_compressions(self) -> Optional[List[COMPRESSION]]:
        return [COMPRESSION.JPEG]

    @property
    def mpp(self) -> SizeMm:
        """Return pixel spacing in um per pixel."""
        return self._mpp

    @property
    def blank_tile(self) -> bytes:
        """Return blank tile."""
        return self._blank_tile

    def _create_blank_tile(self, luminance: float = 1.0) -> bytes:
        """Create a blank tile from a valid tile. Uses the first found
        valid frame (first frame with non-zero value in databytescounts) and
        fills that image with white.

        Parameters
        ----------
        luminance: float = 1.0
            Luminance for tile, 0 = black, 1 = white.

        Returns
        ----------
        bytes:
            Frame bytes from blank tile.

        """
        # Todo, figure out what color to fill with.
        try:
            # Get first frame in page that is not 0 bytes
            valid_frame_index = next(
                index
                for index, datalength in enumerate(self.page.databytecounts)
                if datalength != 0
            )
        except StopIteration as exception:
            raise ValueError("Could not find valid frame in image.") from exception
        tile = self._read_frame(valid_frame_index)
        if self.page.jpegtables is not None:
            tile = Jpeg.add_jpeg_tables(tile, self.page.jpegtables, False)
        tile = self._jpeg.fill_frame(tile, luminance)
        return tile

    def _read_frame(self, index: int) -> bytes:
        """Read frame at frame index from image. Return blank tile if tile is
        sparse (length of frame is zero or frame indexis outside length of
        frames)

        Parameters
        ----------
        index: int
            Frame index to read from image.

        Returns
        ----------
        bytes:
            Frame bytes from frame index or blank tile.

        """
        if (
            index >= len(self.page.databytecounts)
            or self.page.databytecounts[index] == 0
        ):
            # Sparse tile
            return self.blank_tile
        return super()._read_frame(index)
