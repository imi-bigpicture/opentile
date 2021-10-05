import io
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image
from tifffile.tifffile import (FileHandle, TiffPage,
                               svs_description_metadata)

from opentile.common import NativeTiledPage, Tiler
from opentile.geometry import Point, Region, Size, SizeMm
from opentile.utils import Jpeg


class SvsTiledPage(NativeTiledPage):
    def __init__(
        self,
        page: TiffPage,
        fh: FileHandle,
        base_shape: Size,
        base_mpp: SizeMm,
        parent: 'SvsTiledPage' = None
    ):
        """OpenTiledPage for Svs Tiff-page.

        Parameters
        ----------
        page: TiffPage
            TiffPage defining the page.
        fh: NdpiFileHandle
            Filehandler to read data from.
        base_shape: Size
            Size of base level in pyramid.
        base_mpp: SizeMm
            Mpp (um/pixel) for base level in pyramid.
        parent: 'SvsTiledPage' = None
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
        return self.mpp * 1000

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

    def _add_jpeg_tables(
        self,
        frame: bytes
    ) -> bytes:
        """Add jpeg tables to frame. Tables are insterted before 'start of
        scan'-tag, and leading 'start of image' and ending 'end of image' tags
        are removed from the header prior to insertion. Adds colorspace fix at
        end of header.

        Parameters
        ----------
        frame: bytes
            'Abbreviated' jpeg frame lacking jpeg tables.

        Returns
        ----------
        bytes:
            'Interchange' jpeg frame containg jpeg tables.

        """
        start_of_scan = frame.find(Jpeg.start_of_scan())
        with io.BytesIO() as buffer:
            buffer.write(frame[0:start_of_scan])
            buffer.write(self.page.jpegtables[2:-2])  # No start and end tags
            # colorspace fix: Adobe APP14 marker with transform flag 0
            # indicating image is encoded as RGB (not YCbCr)
            buffer.write(
                b"\xFF\xEE\x00\x0E\x41\x64\x6F\x62"
                b"\x65\x00\x64\x80\x00\x00\x00\x00"
            )  

            buffer.write(frame[start_of_scan:None])
            return buffer.getvalue()

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
        scale = int(pow(2, self.pyramid_index - self._parent.pyramid_index))
        scaled_tile_region = Region(tile_point, Size(1, 1)) * scale

        # Get decoded tiles
        decoded_tiles = self._parent.get_decoded_tiles(
            tile.to_tuple() for tile in scaled_tile_region.iterate_all()
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
        image = Image.fromarray(image_data)
        image = image.resize(
            self.tile_size.to_tuple(),
            resample=Image.BILINEAR
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
            return buffer.getvalue()

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
        corrupt, return a fixed tile.

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
        if self.right_edge_corrupt and self._tile_is_at_right_edge(tile_point):
            return self._get_fixed_tile(tile_point)
        if self.bottom_edge_corrupt and self._tile_is_at_bottom_edge(
            tile_point
        ):
            return self._get_fixed_tile(tile_point)
        return super().get_tile(tile_position)


class SvsTiler(Tiler):
    def __init__(self, filepath: Path):
        """Tiler for svs file.

        Parameters
        ----------
        filepath: Path
            Filepath to a svs TiffFile.
        """
        super().__init__(filepath)
        self._fh = self._tiff_file.filehandle

        for series_index, series in enumerate(self.series):
            if series.name == 'Baseline':
                self._level_series_index = series_index
            elif series.name == 'Label':
                self._label_series_index = series_index
            elif series.name == 'Macro':
                self._overview_series_index = series_index
        mpp = svs_description_metadata(self.base_page.description)['MPP']
        self._base_mpp = SizeMm(mpp, mpp)

    @property
    def base_mpp(self) -> SizeMm:
        """Return pixel spacing in um/pixel for base level."""
        return self._base_mpp

    def get_page(
        self,
        series: int,
        level: int,
        page: int = 0
    ) -> SvsTiledPage:
        """Return SvsTiledPage for series, level, page."""
        if not (series, level, page) in self._pages:
            tiff_page = self.series[series].levels[level].pages[page]
            if level > 0:
                parent = self.get_page(series, level-1, page)
            else:
                parent = None

            self._pages[series, level, page] = SvsTiledPage(
                tiff_page,
                self._fh,
                self.base_size,
                self.base_mpp,
                parent
            )
        return self._pages[series, level, page]
