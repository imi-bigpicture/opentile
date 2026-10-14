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

"""Where the tiles of an image are stored in its file."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class TileLocations:
    """Where the tiles of an image are stored in its file, for a consumer to read a
    tile as `get_tile()` serves it with a single read, without going through the
    image.

    The tile at position (x, y) is stored as the `lengths[i]` bytes at
    `positions[i]` of the file, where `i = y * tiled_size.width + x`, and is read as
    `get_tile()` serves it with the `frame_reader` of the image. A tile whose length
    is `READ_THROUGH_IMAGE` is not served as it is stored (e.g. a sparse or corrupt
    tile served as another), and is to be read with `get_tile()`.

    The arrays are made read-only, as consumers may hold views of them instead of
    copies.
    """

    READ_THROUGH_IMAGE: ClassVar[int] = -1
    """Length of a tile that is not served as it is stored."""

    positions: NDArray[np.int64]
    """Position in the file of each tile as stored, in row-major order."""
    lengths: NDArray[np.int64]
    """Number of bytes of each tile as stored, or `READ_THROUGH_IMAGE`."""

    def __post_init__(self) -> None:
        self.positions.flags.writeable = False
        self.lengths.flags.writeable = False

    @classmethod
    def from_frames(
        cls,
        frame_indices: NDArray[np.int64],
        frame_offsets: Sequence[int],
        frame_bytecounts: Sequence[int],
        minimum_length: int,
        read_through: Iterable[int],
    ) -> "TileLocations":
        """Create tile locations from the frame of each tile and where the frames are
        stored.

        A tile is to be read through the image if its frame is not stored (its index
        is past the stored frames), is shorter than `minimum_length` (which includes
        a frame of length 0, which is not stored either), or is in `read_through`.

        Parameters
        ----------
        frame_indices: NDArray[np.int64]
            Index of the frame of each tile, in row-major order of the tiles.
        frame_offsets: Sequence[int]
            Position in the file of each stored frame.
        frame_bytecounts: Sequence[int]
            Number of bytes of each stored frame.
        minimum_length: int
            Shortest frame that is read as it is stored, at least 1. A frame read
            with a prefix in place of its first bytes must be longer than those.
        read_through: Iterable[int]
            Index (in row-major order) of each tile that `get_tile()` serves as
            something else than its stored frame.

        Returns
        -------
        TileLocations
            Where the tiles are stored.
        """
        stored = frame_indices < len(frame_bytecounts)
        stored_frames = frame_indices[stored]
        positions = np.zeros(len(frame_indices), np.int64)
        positions[stored] = np.asarray(frame_offsets, np.int64)[stored_frames]
        lengths = np.full(len(frame_indices), cls.READ_THROUGH_IMAGE, np.int64)
        lengths[stored] = np.asarray(frame_bytecounts, np.int64)[stored_frames]
        lengths[lengths < max(1, minimum_length)] = cls.READ_THROUGH_IMAGE
        lengths[np.fromiter(read_through, np.int64)] = cls.READ_THROUGH_IMAGE
        return cls(positions, lengths)
