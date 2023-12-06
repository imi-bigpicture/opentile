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

"""Image implementations for svs tiff files."""

from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from imagecodecs import JPEG2K, JPEG8, jpeg2k_encode, jpeg8_encode, jpeg8_decode
from PIL import Image
from tifffile.tifffile import COMPRESSION, PHOTOMETRIC, TiffPage

from opentile.geometry import Point, Region, Size, SizeMm
from opentile.jpeg import Jpeg
from opentile.tiff_image import LockableFileHandle, NativeTiledTiffImage, TiffImage


class SvsStripedImage(TiffImage):
    _pyramid_index = 0

    def __init__(self, page: TiffPage, fh: LockableFileHandle, jpeg: Jpeg):
        """OpenTiledPage for jpeg striped Svs image, e.g. overview image.

        Parameters
        ----------
        page: TiffPage
            TiffPage defining the page.
        fh: LockableFileHandle
            Filehandler to read data from.
        jpeg: Jpeg
            Jpeg instance to use.

        """
        super().__init__(page, fh, True)
        self._jpeg = jpeg

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._page}, {self._fh}, {self._jpeg}"

    @property
    def pixel_spacing(self) -> Optional[SizeMm]:
        return None

    @property
    def supported_compressions(self) -> Optional[List[COMPRESSION]]:
        return [COMPRESSION.JPEG]

    def get_tile(self, tile_position: Tuple[int, int]) -> bytes:
        """Return tile for tile position.

        Parameters
        ----------
        tile_position: Tuple[int, int]
            Tile position to get.

        Returns
        ----------
        bytes
            Produced tile at position.
        """
        if tile_position != (0, 0):
            raise ValueError("Non-tiled image, expected tile_position (0, 0)")
        indices = range(len(self.page.dataoffsets))
        scans = (self._read_frame(index) for index in indices)
        jpeg_tables = self.page.jpegtables
        frame = self._jpeg.concatenate_scans(
            scans, jpeg_tables, self._add_rgb_colorspace_fix
        )
        return frame

    def get_decoded_tile(self, tile_position: Tuple[int, int]) -> np.ndarray:
        """Return decoded tile for tile position.

        Parameters
        ----------
        tile_position: Tuple[int, int]
            Tile position to get.

        Returns
        ----------
        bytes
            Produced tile at position.
        """
        return jpeg8_decode(self.get_tile(tile_position))


class SvsLZWImage(TiffImage):
    _pyramid_index = 0

    def __init__(self, page: TiffPage, fh: LockableFileHandle, jpeg: Jpeg):
        """OpenTiledPage for lzw striped Svs image, e.g. label image.

        Parameters
        ----------
        page: TiffPage
            TiffPage defining the page.
        fh: LockableFileHandle
            Filehandler to read data from.
        jpeg: Jpeg
            Jpeg instance to use.

        """
        super().__init__(page, fh)
        self._jpeg = jpeg

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._page}, {self._fh}, {self._jpeg}"

    @property
    def compression(self) -> COMPRESSION:
        """Return compression of page."""
        return COMPRESSION.JPEG

    @property
    def pixel_spacing(self) -> Optional[SizeMm]:
        return None

    @property
    def supported_compressions(self) -> Optional[List[COMPRESSION]]:
        return [COMPRESSION.LZW]

    def get_tile(self, tile_position: Tuple[int, int]) -> bytes:
        """Return tile for tile position.

        Parameters
        ----------
        tile_position: Tuple[int, int]
            Tile position to get.

        Returns
        ----------
        bytes
            Produced tile at position.
        """
        return jpeg8_encode(self.get_decoded_tile(tile_position))

    def get_decoded_tile(self, tile_position: Tuple[int, int]) -> np.ndarray:
        """Return decoded tile for tile position.

        Parameters
        ----------
        tile_position: Tuple[int, int]
            Tile position to get.

        Returns
        ----------
        bytes
            Produced tile at position.
        """
        if tile_position != (0, 0):
            raise ValueError("Non-tiled image, expected tile_position (0, 0)")

        tile = np.concatenate(
            [self._get_row(index) for index in range(len(self.page.dataoffsets))],
            axis=1,
        )
        return np.squeeze(tile)

    def _get_row(self, index: int) -> np.ndarray:
        row = self.page.decode(self._read_frame(index), index)[0]
        assert isinstance(row, np.ndarray)
        return row


class SvsTiledImage(NativeTiledTiffImage):
    def __init__(
        self,
        page: TiffPage,
        fh: LockableFileHandle,
        base_size: Size,
        base_mpp: SizeMm,
        parent: Optional[TiffImage] = None,
    ):
        """Svs Tiff tiled image.

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
        parent: Optional[TiffImage] = None
            Parent TiffImage
        """

        super().__init__(page, fh, True)
        self._base_size = base_size
        self._base_mpp = base_mpp
        self._pyramid_index = self._calculate_pyramidal_index(self._base_size)
        self._mpp = self._calculate_mpp(self._base_mpp)
        self._parent = parent
        (
            self._right_edge_corrupt,
            self._bottom_edge_corrupt,
        ) = self._detect_corrupt_edges()
        self._fixed_tiles: Dict[Point, bytes] = {}

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._page}, {self._fh}, "
            f"{self._base_size}, {self._base_mpp})"
        )

    @property
    def pixel_spacing(self) -> SizeMm:
        """Return pixel spacing in mm per pixel."""
        return self.mpp / 1000

    @property
    def supported_compressions(self) -> Optional[List[COMPRESSION]]:
        return [COMPRESSION.JPEG, COMPRESSION.APERIO_JP2000_RGB]

    @property
    def mpp(self) -> SizeMm:
        """Return pixel spacing in um per pixel."""
        return self._mpp

    @property
    def right_edge_corrupt(self) -> bool:
        return self._right_edge_corrupt

    @property
    def bottom_edge_corrupt(self) -> bool:
        return self._bottom_edge_corrupt

    def _detect_corrupt_edge(self, edge: Region) -> bool:
        """Return true if tiles at edge are corrupt (any tile has a frame
        that has 0 data length)
        Parameters
        ----------
        edge: Region
            Edge to check for corrupt tiles.

        Returns
        ----------
        bool:
            True if edge contains corrupt tiles.
        """
        for tile in edge.iterate_all():
            frame_index = self._tile_point_to_frame_index(tile)
            if self._page.databytecounts[frame_index] == 0:
                return True
        return False

    def _detect_corrupt_edges(self) -> Tuple[bool, bool]:
        """Returns tuple bool indiciting if right and/or bottom edge of page is
        corrupt. Bottom and right edge tiles in svs pyramid levels > 1 can
        sometimes corrupt. This manifests as tiles with 0-length (easy to
        detect) and tiles with scrambled image data (not so easy detect).

        Returns
        ----------
        Tuple[bool, bool]:
            Tuple indicating if right and/or bottom edge is corrupt.

        """
        if self.pyramid_index == 0:
            return False, False
        right_edge = Region(
            Point(self.tiled_size.width - 1, 0), Size(1, self.tiled_size.height - 1)
        )
        right_edge_corrupt = self._detect_corrupt_edge(right_edge)

        bottom_edge = Region(
            Point(0, self.tiled_size.height - 1), Size(self.tiled_size.width - 1, 1)
        )
        bottom_edge_corrupt = self._detect_corrupt_edge(bottom_edge)

        return right_edge_corrupt, bottom_edge_corrupt

    def _tile_is_at_right_edge(self, tile_point: Point) -> bool:
        """Return true if tile is at right edge of tiled image."""
        return tile_point.x == self.tiled_size.width - 1

    def _tile_is_at_bottom_edge(self, tile_point: Point) -> bool:
        """Return true if tile is at bottom edge of tiled image."""
        return tile_point.y == self.tiled_size.height - 1

    def _get_scaled_tile(self, tile_point: Point) -> bytes:
        """Create a tile by downscaling from a lower (higher resolution)
        level.

        Parameters
        ----------
        tile_point: Point
            Position of tile in this pyramid level.

        Returns
        ----------
        bytes:
            Scaled tile bytes.

        """
        if self._parent is None:
            raise ValueError("No parent level to get tiles from")
        scale = int(pow(2, self.pyramid_index - self._parent.pyramid_index))
        scaled_tile_region = Region(tile_point, Size(1, 1)) * scale

        # Get decoded tiles
        decoded_tiles = self._parent.get_decoded_tiles(
            [tile.to_tuple() for tile in scaled_tile_region.iterate_all()]
        )
        image_data = np.zeros(
            (self.tile_size * scale).to_tuple() + (3,), dtype=np.uint8
        )
        # Insert decoded_tiles into image_data
        for y in range(scale):
            for x in range(scale):
                image_data[
                    y * self.tile_size.width : (y + 1) * self.tile_size.width,
                    x * self.tile_size.height : (x + 1) * self.tile_size.height,
                ] = next(decoded_tiles)

        # Resize image_data using Pillow
        image: Image.Image = Image.fromarray(image_data)
        image = image.resize(
            self.tile_size.to_tuple(), resample=Image.Resampling.BILINEAR
        )

        # Return compressed image
        if self.compression == COMPRESSION.JPEG:
            if self._page.photometric == PHOTOMETRIC.RGB:
                colorspace = JPEG8.CS.RGB
            elif self._page.photometric == PHOTOMETRIC.YCBCR:
                colorspace = JPEG8.CS.YCbCr
            else:
                raise NotImplementedError("Non-supported color space")
            subsampling = self._page.subsampling
            return jpeg8_encode(
                np.array(image),
                level=95,
                colorspace=colorspace,
                subsampling=subsampling,
                lossless=False,
                bitspersample=8,
            )
        if self.compression == COMPRESSION.APERIO_JP2000_RGB:
            return jpeg2k_encode(
                np.array(image),
                level=80,
                codecformat=JPEG2K.CODEC.J2K,
                colorspace=JPEG2K.CLRSPC.SRGB,
                bitspersample=8,
                reversible=False,
                mct=True,
            )
        raise NotImplementedError("Non-supported compression")

    def _get_fixed_tile(self, tile_point: Point) -> bytes:
        """Get or create a fixed tile inplace for a corrupt tile.

        Parameters
        ----------
        tile_point: Point
            Position of tile to get.

        Returns
        ----------
        bytes:
            Fixed tile bytes.

        """
        if tile_point not in self._fixed_tiles:
            fixed_tile = self._get_scaled_tile(tile_point)
            self._fixed_tiles[tile_point] = fixed_tile
        return self._fixed_tiles[tile_point]

    def get_tile(self, tile_position: Tuple[int, int]) -> bytes:
        """Return image bytes for tile at tile position. If tile is marked as
        corrupt, return a fixed tile. Add color space fix if jpeg compression.

        Parameters
        ----------
        tile_position: Tuple[int, int]
            Tile position to get.

        Returns
        ----------
        bytes
            Produced tile at position.
        """
        tile_point = Point.from_tuple(tile_position)
        if self._tile_is_corrupt(tile_point):
            return self._get_fixed_tile(tile_point)

        return super().get_tile(tile_position)

    def get_tiles(self, tile_positions: Sequence[Tuple[int, int]]) -> Iterator[bytes]:
        """Return list of image bytes for tiles at tile positions.

        Parameters
        ----------
        tile_positions: Sequence[Tuple[int, int]]
            Tile positions to get.

        Returns
        ----------
        Iterator[bytes]
            List of tile bytes.
        """
        tile_points = [
            Point.from_tuple(tile_position) for tile_position in tile_positions
        ]
        if any(self._tile_is_corrupt(tile_point) for tile_point in tile_points):
            return (self.get_tile(tile) for tile in tile_positions)
        return super().get_tiles(tile_positions)

    def _tile_is_corrupt(self, tile_point: Point) -> bool:
        """Return true if tile is corrupt

        Parameters
        ----------
        tile_point: Point
            Tile to check

        Returns
        ----------
        bool
            True if tile is corrupt.
        """
        return (
            self.right_edge_corrupt and self._tile_is_at_right_edge(tile_point)
        ) or (self.bottom_edge_corrupt and self._tile_is_at_bottom_edge(tile_point))
