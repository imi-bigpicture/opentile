#    Copyright 2021, 2022, 2023 SECTRA AB
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


from typing import List, Tuple, Union

from opentile.geometry import Point, Size


class NdpiTile:
    """Defines a tile by position and coordinates and size for cropping out
    out frame."""

    def __init__(self, position: Point, tile_size: Size, frame_size: Size) -> None:
        """Create a ndpi tile and calculate cropping parameters.

        Parameters
        ----------
        position: Point
            Tile position.
        tile_size: Size
            Tile size.
        frame_size: Size
            Frame size.

        """
        self._position = position
        self._tile_size = tile_size
        self._frame_size = frame_size

        self._tiles_per_frame = Size.max(
            self._frame_size // self._tile_size, Size(1, 1)
        )
        position_inside_frame: Point = (self.position * self._tile_size) % Size.max(
            self._frame_size, self._tile_size
        )
        self._left = position_inside_frame.x
        self._top = position_inside_frame.y
        self._frame_position = (
            self.position // self._tiles_per_frame
        ) * self._tiles_per_frame

    def __eq__(self, other: object) -> bool:
        if isinstance(other, NdpiTile):
            return (
                self.position == other.position
                and self._tile_size == other._tile_size
                and self._frame_size == other._frame_size
            )
        return NotImplemented

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self.position}, {self._tile_size}, "
            f"{self._frame_size})"
        )

    def __str__(self) -> str:
        return f"{type(self).__name__} of position {self.position}"

    @property
    def position(self) -> Point:
        """Return position of tile."""
        return self._position

    @property
    def frame_position(self) -> Point:
        """Return frame position for tile."""
        return self._frame_position

    @property
    def left(self) -> int:
        """Return left coordinate for tile inside frame."""
        return self._left

    @property
    def top(self) -> int:
        """Return top coordinate for tile inside frame."""
        return self._top

    @property
    def width(self) -> int:
        """Return width for tile inside frame."""
        return self._tile_size.width

    @property
    def height(self) -> int:
        """Return height for tile inside frame."""
        return self._tile_size.height

    @property
    def frame_size(self) -> Size:
        """Return frame size."""
        return self._frame_size


class NdpiFrameJob:
    """A list of tiles to create from a frame. Tiles need to have the same
    frame position."""

    def __init__(self, tiles: Union[NdpiTile, List[NdpiTile]]) -> None:
        """Create a frame job from given tile(s).

        Parameters
        ----------
        tiles: Union[NdpiTile, List[NdpiTile]]
            Tile(s) to base the frame job on.

        """
        if isinstance(tiles, NdpiTile):
            tiles = [tiles]
        first_tile = tiles.pop(0)
        self._position = first_tile.frame_position
        self._frame_size = first_tile.frame_size
        self._tiles: List[NdpiTile] = [first_tile]
        for tile in tiles:
            self.append(tile)

    def __repr__(self) -> str:
        return f"{type(self).__name__}{self.tiles})"

    def __str__(self) -> str:
        return f"{type(self).__name__} of tiles {self.tiles}"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, NdpiFrameJob):
            return self.tiles == other.tiles
        return NotImplemented

    @property
    def position(self) -> Point:
        """The frame position of the frame job."""
        return self._position

    @property
    def frame_size(self) -> Size:
        """Frame size required for reading tiles in NdpiFrameJob."""
        return self._frame_size

    @property
    def tiles(self) -> List[NdpiTile]:
        """Tiles in NdpiFrameJob."""
        return self._tiles

    @property
    def crop_parameters(self) -> List[Tuple[int, int, int, int]]:
        """Parameters for croping tiles from frame in NdpiFrameJob."""
        return [(tile.left, tile.top, tile.width, tile.height) for tile in self._tiles]

    def append(self, tile: NdpiTile) -> None:
        """Add a tile to the tile job."""
        if tile.frame_position != self.position:
            raise ValueError(f"{tile} does not match {self} frame position")
        self._tiles.append(tile)
