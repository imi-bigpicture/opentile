#    Copyright 2021 SECTRA AB
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

import io
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, cast

import numpy as np
from PIL import Image
from tifffile.tifffile import (FileHandle, TiffFile, TiffPage,
                               svs_description_metadata)

from opentile.common import NativeTiledPage, OpenTilePage, Tiler
from opentile.geometry import Point, Region, Size, SizeMm
from opentile.jpeg import Jpeg


class SvsStripedPage(OpenTilePage):
    _pyramid_index = 0

    def __init__(self, page: TiffPage, fh: FileHandle, jpeg: Jpeg):
        """OpenTiledPage for jpeg striped Svs page, e.g. overview page.

        Parameters
        ----------
        page: TiffPage
            TiffPage defining the page.
        fh: FileHandle
            Filehandler to read data from.
        jpeg: Jpeg
            Jpeg instance to use.

        """
        super().__init__(page, fh)
        self._jpeg = jpeg

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._page}, {self._fh}, {self._jpeg}"
        )

    @property
    def pixel_spacing(self) -> Optional[SizeMm]:
        return None

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
            raise ValueError("Non-tiled page, expected tile_position (0, 0)")
        indices = range(len(self.page.dataoffsets))
        scans = (self._read_frame(index) for index in indices)
        jpeg_tables = self.page.jpegtables
        frame = self._jpeg.concatenate_scans(
            scans,
            jpeg_tables,
            True
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
        return self._jpeg.decode(self.get_tile(tile_position))


class SvsLZWPage(OpenTilePage):
    _pyramid_index = 0

    def __init__(self, page: TiffPage, fh: FileHandle, jpeg: Jpeg):
        """OpenTiledPage for lzw striped Svs page, e.g. label page.

        Parameters
        ----------
        page: TiffPage
            TiffPage defining the page.
        fh: FileHandle
            Filehandler to read data from.
        jpeg: Jpeg
            Jpeg instance to use.

        """
        super().__init__(page, fh)
        self._jpeg = jpeg

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._page}, {self._fh}, {self._jpeg}"
        )

    @property
    def compression(self) -> str:
        """Return compression of page."""
        return 'COMPRESSION.JPEG'

    @property
    def pixel_spacing(self) -> Optional[SizeMm]:
        return None

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
        return self._jpeg.encode(self.get_decoded_tile(tile_position))

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
            raise ValueError("Non-tiled page, expected tile_position (0, 0)")

        tile = np.concatenate([
            self._get_row(index)
            for index in range(len(self.page.dataoffsets))
        ], axis=1)
        return np.squeeze(tile)

    def _get_row(self, index: int) -> np.ndarray:
        row = self.page.decode(
            self._read_frame(index),
            index
        )[0]
        assert(isinstance(row, np.ndarray))
        return row


class SvsTiledPage(NativeTiledPage):
    def __init__(
        self,
        page: TiffPage,
        fh: FileHandle,
        base_shape: Size,
        base_mpp: SizeMm,
        parent: Optional['SvsTiledPage'] = None
    ):
        """OpenTiledPage for Svs Tiff-page.

        Parameters
        ----------
        page: TiffPage
            TiffPage defining the page.
        fh: FileHandle
            Filehandler to read data from.
        base_shape: Size
            Size of base level in pyramid.
        base_mpp: SizeMm
            Mpp (um/pixel) for base level in pyramid.
        parent: Optional['SvsTiledPage'] = None
            Parent SvsTiledPage
        """
        super().__init__(page, fh)
        self._base_shape = base_shape
        self._base_mpp = base_mpp
        self._pyramid_index = self._calculate_pyramidal_index(self._base_shape)
        self._mpp = self._calculate_mpp(self._base_mpp)
        self._parent = parent
        self._right_edge_corrupt, self._bottom_edge_corrupt = (
            self._detect_corrupt_edges()
        )
        self._fixed_tiles: Dict[Point, bytes] = {}

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._page}, {self._fh}, "
            f"{self._base_shape}, {self._base_mpp})"
        )

    @property
    def pixel_spacing(self) -> SizeMm:
        """Return pixel spacing in mm per pixel."""
        return self.mpp / 1000

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
            Point(self.tiled_size.width-1, 0),
            Size(1, self.tiled_size.height-1)
        )
        right_edge_corrupt = self._detect_corrupt_edge(right_edge)

        bottom_edge = Region(
            Point(0, self.tiled_size.height-1),
            Size(self.tiled_size.width-1, 1)
        )
        bottom_edge_corrupt = self._detect_corrupt_edge(bottom_edge)

        return right_edge_corrupt, bottom_edge_corrupt

    def _tile_is_at_right_edge(self, tile_point: Point) -> bool:
        """Return true if tile is at right edge of tiled image."""
        return tile_point.x == self.tiled_size.width - 1

    def _tile_is_at_bottom_edge(self, tile_point: Point) -> bool:
        """Return true if tile is at bottom edge of tiled image."""
        return tile_point.y == self.tiled_size.height - 1

    def _get_scaled_tile(
        self,
        tile_point: Point
    ) -> bytes:
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
            (self.tile_size*scale).to_tuple() + (3,),
            dtype=np.uint8
        )
        # Insert decoded_tiles into image_data
        for y in range(scale):
            for x in range(scale):
                image_data_index = y*scale + x
                image_data[
                    y*self.tile_size.width:(y+1)*self.tile_size.width,
                    x*self.tile_size.height:(x+1)*self.tile_size.height
                ] = decoded_tiles[image_data_index]

        # Resize image_data using Pillow
        image: Image.Image = Image.fromarray(image_data)
        image = image.resize(
            self.tile_size.to_tuple(),
            resample=Image.Resampling.BILINEAR
        )

        # Return compressed image
        if self.compression == 'COMPRESSION.JPEG':
            image_format = 'jpeg'
            image_options = {'quality': 95}
        elif self.compression == 'COMPRESSION.APERIO_JP2000_RGB':
            image_format = 'jpeg2000'
            image_options = {'irreversible': True}
        else:
            raise NotImplementedError("Not supported compression")
        with io.BytesIO() as buffer:
            image.save(buffer, format=image_format, **image_options)
            frame = buffer.getvalue()

        if self.compression == 'COMPRESSION.APERIO_JP2000_RGB':
            # PIL encodes in jp2, find start of j2k and return from there.
            START_TAGS = bytes([0xFF, 0x4F, 0xFF, 0x51])
            start_index = frame.find(START_TAGS)
            return frame[start_index:]
        return frame

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
        # Check if tile is corrupted
        tile_corrupt = (
            self.right_edge_corrupt and
            self._tile_is_at_right_edge(tile_point)
        ) or (
            self.bottom_edge_corrupt and
            self._tile_is_at_bottom_edge(tile_point)
        )
        if tile_corrupt:
            return self._get_fixed_tile(tile_point)

        tile = super().get_tile(tile_position)
        if self.compression == 'COMPRESSION.JPEG':
            tile = Jpeg.add_color_space_fix(tile)
        return tile


class SvsTiler(Tiler):
    def __init__(
        self,
        filepath: Union[str, Path],
        turbo_path: Optional[Union[str, Path]] = None
    ):
        """Tiler for svs file.

        Parameters
        ----------
        filepath: Union[str, Path]
            Filepath to a svs TiffFile.
        """
        super().__init__(Path(filepath))
        self._fh = self._tiff_file.filehandle
        self._turbo_path = turbo_path
        self._jpeg = Jpeg(self._turbo_path)

        for series_index, series in enumerate(self.series):
            if series.name == 'Baseline':
                self._level_series_index = series_index
            elif series.name == 'Label':
                self._label_series_index = series_index
            elif series.name == 'Macro':
                self._overview_series_index = series_index
        mpp = svs_description_metadata(self.base_page.description)['MPP']
        self._base_mpp = SizeMm(mpp, mpp)
        self._pages: Dict[
            Tuple[int, int, int], OpenTilePage
        ] = {}
        if 'InterColorProfile' in self._tiff_file.pages.first.tags:
            icc_profile = (
                self._tiff_file.pages.first.tags['InterColorProfile'].value
            )
            assert(isinstance(icc_profile, bytes) or icc_profile is None)
            self._icc_profile = icc_profile

    @property
    def base_mpp(self) -> SizeMm:
        """Return pixel spacing in um/pixel for base level."""
        return self._base_mpp

    @classmethod
    def supported(cls, tiff_file: TiffFile) -> bool:
        return tiff_file.is_svs

    def _get_level_page(
        self,
        level: int,
        page: int = 0
    ) -> SvsTiledPage:
        series = self._level_series_index
        if level > 0:
            parent = self.get_page(series, level-1, page)
            parent = cast(SvsTiledPage, parent)
        else:
            parent = None
        svs_page = SvsTiledPage(
            self._get_tiff_page(series, level, page),
            self._fh,
            self.base_size,
            self.base_mpp,
            parent
        )
        return svs_page

    def get_page(
        self,
        series: int,
        level: int,
        page: int = 0
    ) -> OpenTilePage:
        """Return SvsTiledPage for series, level, page."""
        if not (series, level, page) in self._pages:

            if series == self._overview_series_index:
                svs_page = SvsStripedPage(
                    self._get_tiff_page(series, level, page),
                    self._fh,
                    self._jpeg
                )
            elif series == self._label_series_index:
                svs_page = SvsLZWPage(
                    self._get_tiff_page(series, level, page),
                    self._fh,
                    self._jpeg
                )
            elif series == self._level_series_index:
                svs_page = self._get_level_page(level, page)
            else:
                raise NotImplementedError()

            self._pages[series, level, page] = svs_page
        return self._pages[series, level, page]
