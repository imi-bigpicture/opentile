#    Copyright 2026 SECTRA AB
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

"""Image implementations for Trestle tiff files."""

from collections.abc import Iterator, Sequence
from typing import Optional, Union

import numpy as np
import opentile_jpeg
from imagecodecs import jpeg_decode
from tifffile import COMPRESSION, TiffPage

from opentile.file import OpenTileFile
from opentile.geometry import Point, PointF, Region, Size, SizeMm
from opentile.jpeg import JpegInfo, JpegProcess
from opentile.tiff_image_bases import (
    ExplicitScaleLevelTiffImage,
    NativeTiledTiffImage,
    OverlappingLevelTiffImage,
)


class TrestleTiledImage(ExplicitScaleLevelTiffImage, NativeTiledTiffImage):
    """A Trestle level stored without overlap at the requested tile size, served as
    stored."""

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._page}, {self._file}, "
            f"{self._base_mpp}, {self._scale}, {self._origin_offset})"
        )

    @property
    def supported_compressions(self) -> Optional[list[COMPRESSION]]:
        return None


class TrestleOverlappingImage(OverlappingLevelTiffImage):
    """A Trestle level served as stored, with an `overlap` placement for the consumer
    to compose. `regrid_if_possible()` gives the level re-gridded instead."""

    def regrid_if_possible(
        self, tile_size: Size
    ) -> Union["TrestleRegriddedImage", "TrestleOverlappingImage"]:
        """The level re-gridded into ``tile_size`` tiles if that is lossless, else this
        level.

        Parameters
        ----------
        tile_size: Size
            Size of the tiles to serve.

        Returns
        ----------
        Union[TrestleRegriddedImage, TrestleOverlappingImage]
            The re-gridded level, or this level.
        """
        step = self._lossless_regrid_step(tile_size)
        if step is None:
            return self
        return TrestleRegriddedImage(
            self._page,
            self._file,
            self._base_mpp,
            self._scale,
            self._origin_offset,
            self,
            step,
            tile_size,
        )

    def _lossless_regrid_step(self, tile_size: Size) -> Optional[Size]:
        """The stride of the stored tiles if they can be re-gridded losslessly into
        ``tile_size`` tiles, else None.

        Eligible when the stored tiles are baseline 8-bit JPEG placed as a regular
        grid of whole tiles (one placement per stored tile, at ``grid * step``) and
        the step and ``tile_size`` are multiples of the JPEG MCU.

        Parameters
        ----------
        tile_size: Size
            Size of the tiles to serve.

        Returns
        ----------
        Optional[Size]
            The step, or None if the level can not be re-gridded losslessly.
        """
        info = self.encoded_info
        if not (
            self.compression == COMPRESSION.JPEG
            and isinstance(info, JpegInfo)
            and info.process == JpegProcess.BASELINE
            and info.bit_depth == 8
        ):
            return None
        stored = self.tile_size
        placements = self.overlap.placements
        if any(
            len(pieces) != 1 or pieces[0].source != Region(Point(0, 0), stored)
            for pieces in placements.values()
        ):
            return None
        # The step is the position of the neighbour. A level one stored tile wide
        # (or tall) has none; its single tile sits at 0, so any step serves, and the
        # stored tile size is used.
        right, below = placements.get(Point(1, 0)), placements.get(Point(0, 1))
        tiled_size = self.tiled_size
        if (right is None and tiled_size.width > 1) or (
            below is None and tiled_size.height > 1
        ):
            return None
        step_x = float(right[0].position.x) if right else float(stored.width)
        step_y = float(below[0].position.y) if below else float(stored.height)
        if not (step_x.is_integer() and step_y.is_integer()):
            return None
        step_size = Size(int(step_x), int(step_y))
        if any(
            pieces[0].position.x != point.x * step_size.width
            or pieces[0].position.y != point.y * step_size.height
            for point, pieces in placements.items()
        ):
            return None
        mcu_width, mcu_height = info.mcu_size.to_tuple()
        if (
            step_size.width % mcu_width
            or step_size.height % mcu_height
            or tile_size.width % mcu_width
            or tile_size.height % mcu_height
        ):
            return None
        return step_size


class TrestleRegriddedImage(ExplicitScaleLevelTiffImage):
    """A Trestle level whose stored JPEG tiles, placed on a regular grid, are served
    re-gridded into non-overlapping tiles of the requested size.

    Each stored tile keeps its ``step`` top-left footprint on the de-overlapped
    canvas (the last tile in each row and column also keeps its tail). A served tile
    is a window of that canvas, built by copying JPEG blocks (`opentile_jpeg.compose`)
    without decoding, so it adds no loss. The step and tile size must therefore be
    multiples of the JPEG MCU.

    The part of an edge tile past the image is white, except the partial MCU on the
    image edge, which keeps the stored tile's padding.
    """

    _TILES_PER_COMPOSE = 16
    """Output tiles composed per call; bounds the stored tiles held decoded (to
    coefficients) at once, at the cost of re-reading stored tiles shared by tiles in
    different calls."""

    def __init__(
        self,
        page: TiffPage,
        file: OpenTileFile,
        base_mpp: SizeMm,
        scale: float,
        origin_offset: PointF,
        source: TrestleOverlappingImage,
        step: Size,
        tile_size: Size,
    ):
        """Use `TrestleOverlappingImage.regrid_if_possible()` to create.

        Parameters
        ----------
        page: TiffPage
            TiffPage defining the level.
        file: OpenTileFile
            File to read data from.
        base_mpp: SizeMm
            Mpp (um/pixel) for the base level in the pyramid.
        scale: float
            Downsample of this level relative to the base level.
        origin_offset: PointF
            Position of the level's first pixel on the base level, in base-level
            pixels (see `LevelTiffImage.origin_offset`).
        source: TrestleOverlappingImage
            The level as stored. Its tiles are re-gridded, and the size of its
            `overlap` is the image size served.
        step: Size
            Stride of the stored tiles on the de-overlapped canvas (stored tile size
            minus the overlap). Must be a multiple of the JPEG MCU.
        tile_size: Size
            Size of the served tiles. Must be a multiple of the JPEG MCU.
        """
        # Set before `super().__init__()`, which reads the image size from it.
        self._source = source
        if not isinstance(source.encoded_info, JpegInfo):
            raise ValueError("Lossless re-gridding needs JPEG tiles.")
        super().__init__(
            page, file, base_mpp, scale, origin_offset, tile_size=tile_size
        )
        self._step = step
        self._mcu = source.encoded_info.mcu_size
        self._trailing_size = self._trailing_stored_size()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._page}, {self._file}, {self._base_mpp}, "
            f"{self._scale}, {self._step}, {self.tile_size})"
        )

    def _read_image_size(self) -> Size:
        return self._source.overlap.image_size

    @property
    def supported_compressions(self) -> Optional[list[COMPRESSION]]:
        return [COMPRESSION.JPEG]

    def get_tile(self, tile_position: tuple[int, int]) -> bytes:
        return next(self.get_tiles([tile_position]))

    def get_tiles(self, tile_positions: Sequence[tuple[int, int]]) -> Iterator[bytes]:
        tile_points = [
            self._validated_tile_point(tile_position)
            for tile_position in tile_positions
        ]
        for start in range(0, len(tile_points), self._TILES_PER_COMPOSE):
            yield from self._compose(
                tile_points[start : start + self._TILES_PER_COMPOSE]
            )

    def get_all_tiles(self, raw: bool = False) -> Iterator[bytes]:
        # There are no stored frames matching the served grid, so raw is ignored.
        return self.get_tiles(
            [tile.to_tuple() for tile in self._tiled_region.iterate_all()]
        )

    def get_decoded_tile(self, tile_position: tuple[int, int]) -> np.ndarray:
        return jpeg_decode(self.get_tile(tile_position))

    def _compose(self, tile_points: Sequence[Point]) -> Iterator[bytes]:
        """Compose served tiles from the stored tiles covering them, reading each
        stored tile once per call."""
        tile_width, tile_height = self.tile_size.to_tuple()
        step_x, step_y = self._step.to_tuple()
        stored_width, stored_height = self._source.tile_size.to_tuple()
        last_x = self._source.tiled_size.width - 1
        last_y = self._source.tiled_size.height - 1
        cells: dict[tuple[int, int], None] = {}
        for tile_point in tile_points:
            left, top = tile_point.x * tile_width, tile_point.y * tile_height
            # A stored tile covers the window when its full extent (content plus the
            # tail the trailing tile keeps) intersects it.
            for grid_y in range(
                max(0, (top - stored_height) // step_y + 1),
                min(last_y, (top + tile_height - 1) // step_y) + 1,
            ):
                for grid_x in range(
                    max(0, (left - stored_width) // step_x + 1),
                    min(last_x, (left + tile_width - 1) // step_x) + 1,
                ):
                    cells[(grid_x, grid_y)] = None
        frames = self._source.get_tiles(list(cells))
        sources = [
            (self._crop_past_image(frame, x == last_x, y == last_y), x, y)
            for frame, (x, y) in zip(frames, cells)
        ]
        yield from opentile_jpeg.compose(
            sources,
            self._step.to_tuple(),
            self.tile_size.to_tuple(),
            [(point.x * tile_width, point.y * tile_height) for point in tile_points],
            self._source.tile_size.to_tuple(),
            background=self._background,
        )

    @property
    def _background(self) -> tuple[int, int, int]:
        """White, the fill for the parts of edge tiles past the image."""
        return (255, 255, 255)

    def _trailing_stored_size(self) -> Size:
        """Size of the whole MCUs holding image in the last stored column (width) and
        row (height). The rest of those stored tiles is past the image."""
        mcu_width, mcu_height = self._mcu.to_tuple()
        stored_width, stored_height = self._source.tile_size.to_tuple()
        last_x = self._source.tiled_size.width - 1
        last_y = self._source.tiled_size.height - 1
        inside_width = self.image_size.width - last_x * self._step.width
        inside_height = self.image_size.height - last_y * self._step.height
        return Size(
            min(stored_width, -(-inside_width // mcu_width) * mcu_width),
            min(stored_height, -(-inside_height // mcu_height) * mcu_height),
        )

    def _crop_past_image(
        self, frame: bytes, last_column: bool, last_row: bool
    ) -> bytes:
        """Crop a stored tile of the last column or row to the MCUs holding image.
        `opentile_jpeg.compose` then fills the part past the image with the
        background, as no source covers it."""
        stored_width, stored_height = self._source.tile_size.to_tuple()
        width = self._trailing_size.width if last_column else stored_width
        height = self._trailing_size.height if last_row else stored_height
        if width == stored_width and height == stored_height:
            return frame
        return opentile_jpeg.crop(frame, 0, 0, width, height)
