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

"""Tile positions solved from measured offsets between neighbouring tiles.

Scanners that store overlapping tiles may record how each tile lines up with its
neighbours. Each such measurement is a `TileOffset`; `TilePositionSolver` finds the
tile positions that agree best with all of them.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from opentile.geometry import Point, PointF


@dataclass(frozen=True)
class TileOffset:
    """A measured offset between two neighbouring tiles."""

    first: Point
    """Grid position of one of the tiles."""
    second: Point
    """Grid position of the other tile."""
    offset: PointF
    """Position of `second` minus position of `first`, in pixels. The order only sets
    the sign: both positions are solved for, and swapping the tiles negates the
    offset."""
    weight: float = 1.0
    """Relative trust in the measurement, at least 0. Offsets with weight 0 are
    ignored."""


class TilePositionSolver:
    """Places the tiles of a grid so that they best agree with measured offsets.

    Minimises ``sum(weight * |position(second) - position(first) - offset| ** 2)``
    over all offsets, with the anchor tile held at (0, 0). The two axes are
    independent of each other.

    The tiles must fill a rectangular grid, and every offset must be between direct
    neighbours. The normal equations form a weighted graph Laplacian over the tiles,
    which with the anchor fixed is symmetric positive definite as long as no weight
    is negative and every tile is linked to the anchor through offsets with non-zero
    weight. All of this is checked on construction.

    The grid is cut into lines along its shorter side. Ordered line by line, the
    Laplacian is block tridiagonal:

    - a diagonal block per line holds the links along that line, and
    - the block between two adjacent lines is diagonal, as each tile only links to
      the tile straight across.

    Eliminating one line after the other (the block Thomas algorithm) gives the exact
    solution up to rounding. Each line's Schur complement is inverted explicitly, as
    the inverse is used in both the forward and the backward pass. That is safe here,
    as the complements are small and well conditioned.

    The cost is about ``lines * line_length ** 3`` operations and
    ``2 * lines * line_length ** 2`` floats of memory, fine for long areas but
    growing fast for large square ones.
    """

    MAX_REPORTED = 5
    """Number of offending tiles or offsets to name in an error message."""

    def __init__(
        self,
        tiles: Sequence[Point],
        offsets: Sequence[TileOffset],
        anchor: Point,
    ):
        """Build the system to solve.

        Parameters
        ----------
        tiles: Sequence[Point]
            Grid positions of the tiles to place, each listed once. They must fill a
            rectangle.
        offsets: Sequence[TileOffset]
            Offsets between pairs of `tiles` that are direct grid neighbours. Callers
            typically add weak offsets at the nominal pitch between neighbours that
            lack a measurement, so that every tile is linked to the anchor.
        anchor: Point
            Tile whose position is fixed at (0, 0).

        Raises
        ------
        ValueError
            If a tile is listed more than once in `tiles`; if `anchor` or a tile in
            `offsets` is not among `tiles`; if the tiles do not fill a rectangle; if
            an offset is not between direct neighbours, is not finite, or has a
            negative or non-finite weight; or if a tile is not linked to the anchor
            through offsets with non-zero weight.
        """
        self._index = {tile: i for i, tile in enumerate(tiles)}
        if self._tile_count != len(tiles):
            duplicates = [
                tile for tile, i in self._index.items() if tiles.index(tile) != i
            ]
            raise ValueError(
                f"Tiles listed more than once: {duplicates[: self.MAX_REPORTED]}"
            )
        if anchor not in self._index:
            raise ValueError(f"Anchor {anchor} is not among the tiles.")
        self._anchor = anchor
        self._anchor_index = self._index[anchor]
        origin, width, height = self._check_rectangle()
        self._check_offsets(offsets)

        self._first_indices = np.array(
            [self._index[o.first] for o in offsets], dtype=np.int64
        )
        self._second_indices = np.array(
            [self._index[o.second] for o in offsets], dtype=np.int64
        )
        self._weights = np.array([o.weight for o in offsets], dtype=np.float64)
        self._check_connected()

        self._degree = np.bincount(
            self._first_indices, self._weights, self._tile_count
        ) + np.bincount(self._second_indices, self._weights, self._tile_count)
        measured = np.array(
            [[o.offset.x, o.offset.y] for o in offsets], dtype=np.float64
        ).reshape(-1, 2)
        # Right-hand side of the normal equations: the net weighted pull of the
        # offsets on each tile, +offset on its second tile and -offset on its first.
        # The anchor is fixed and takes no part in the solve.
        weighted = self._weights[:, None] * measured
        self._offset_pull = self._scatter(
            self._second_indices, weighted
        ) - self._scatter(self._first_indices, weighted)
        self._offset_pull[self._anchor_index] = 0.0

        xs = np.array([tile.x for tile in self._index], dtype=np.int64) - origin.x
        ys = np.array([tile.y for tile in self._index], dtype=np.int64) - origin.y
        # Lines run along the shorter side, so that the dense blocks stay small.
        if width <= height:
            self._line_of, self._position_of = ys, xs
            self._line_count, self._line_length = height, width
        else:
            self._line_of, self._position_of = xs, ys
            self._line_count, self._line_length = width, height

    def solve(self) -> dict[Point, PointF]:
        """Solve for the tile positions.

        Returns
        -------
        dict[Point, PointF]
            Position of each tile relative to the anchor, in pixels.
        """
        line_blocks, couplings, reduced_pull = self._blocks()
        # Forward elimination: replace each line's block by its Schur complement and
        # fold the previous line's pull into its own.
        inverses = np.empty_like(line_blocks)
        inverses[0] = np.linalg.inv(line_blocks[0])
        for line in range(1, self._line_count):
            coupling = couplings[line]
            schur = line_blocks[line] - (
                coupling[:, None] * inverses[line - 1] * coupling[None, :]
            )
            reduced_pull[line] -= coupling[:, None] * (
                inverses[line - 1] @ reduced_pull[line - 1]
            )
            inverses[line] = np.linalg.inv(schur)
        # Back substitution, from the last line to the first.
        line_positions = np.empty_like(reduced_pull)
        line_positions[-1] = inverses[-1] @ reduced_pull[-1]
        for line in range(self._line_count - 2, -1, -1):
            line_positions[line] = inverses[line] @ (
                reduced_pull[line]
                - couplings[line + 1][:, None] * line_positions[line + 1]
            )

        solution = line_positions[self._line_of, self._position_of]
        solution[self._anchor_index] = 0.0
        return {
            tile: PointF(*map(float, solution[i])) for tile, i in self._index.items()
        }

    @property
    def _tile_count(self) -> int:
        """Number of tiles to place."""
        return len(self._index)

    def _check_rectangle(self) -> tuple[Point, int, int]:
        """Raise if the tiles do not fill their bounding rectangle.

        Returns
        -------
        tuple[Point, int, int]
            Top-left tile, width and height of the rectangle.
        """
        xs = [tile.x for tile in self._index]
        ys = [tile.y for tile in self._index]
        width = max(xs) - min(xs) + 1
        height = max(ys) - min(ys) + 1
        # With no duplicates, the count matches only if every position is filled.
        if width * height != self._tile_count:
            raise ValueError(
                f"The {self._tile_count} tiles do not fill their {width} x {height} "
                "bounding rectangle."
            )
        return Point(min(xs), min(ys)), width, height

    def _check_offsets(self, offsets: Sequence[TileOffset]) -> None:
        """Raise if an offset refers to a tile that is not among the tiles, is not
        between direct neighbours, is not finite, or has a negative or non-finite
        weight."""
        unknown = list(
            dict.fromkeys(
                tile
                for offset in offsets
                for tile in (offset.first, offset.second)
                if tile not in self._index
            )
        )
        if unknown:
            raise ValueError(
                "Offsets refer to tiles that are not among the tiles: "
                f"{unknown[: self.MAX_REPORTED]}"
            )
        distant = [
            (offset.first, offset.second)
            for offset in offsets
            if abs(offset.second.x - offset.first.x)
            + abs(offset.second.y - offset.first.y)
            != 1
        ]
        if distant:
            raise ValueError(
                "Offsets between tiles that are not direct neighbours: "
                f"{distant[: self.MAX_REPORTED]}"
            )
        invalid = [
            offset
            for offset in offsets
            if not (
                math.isfinite(offset.offset.x)
                and math.isfinite(offset.offset.y)
                and math.isfinite(offset.weight)
                and offset.weight >= 0
            )
        ]
        if invalid:
            raise ValueError(
                "Offsets must be finite, with finite weights of at least 0: "
                f"{invalid[: self.MAX_REPORTED]}"
            )

    def _check_connected(self) -> None:
        """Raise if a tile is not linked to the anchor through weighted offsets."""
        neighbours: list[list[int]] = [[] for _ in range(self._tile_count)]
        for first, second, weight in zip(
            self._first_indices, self._second_indices, self._weights
        ):
            if weight != 0:
                neighbours[first].append(int(second))
                neighbours[second].append(int(first))
        reached = {self._anchor_index}
        pending = [self._anchor_index]
        while pending:
            for neighbour in neighbours[pending.pop()]:
                if neighbour not in reached:
                    reached.add(neighbour)
                    pending.append(neighbour)
        unconnected = [tile for tile, i in self._index.items() if i not in reached]
        if unconnected:
            raise ValueError(
                f"Tiles not linked to the anchor {self._anchor}: "
                f"{unconnected[: self.MAX_REPORTED]}"
            )

    def _scatter(
        self, indices: NDArray[np.int64], values: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Sum per-offset (offsets, 2) values into per-tile (tiles, 2) sums."""
        return np.stack(
            [
                np.bincount(indices, values[:, axis], self._tile_count)
                for axis in (0, 1)
            ],
            axis=1,
        )

    def _blocks(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """The Laplacian and pull, split by line.

        Returns
        -------
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
            The (lines, length, length) diagonal blocks; the (lines, length) diagonals
            of the blocks linking each line to the previous one (zero for the first);
            and the (lines, length, 2) pull. The anchor's row and column are
            replaced by the identity and its pull is 0, fixing it at 0.
        """
        first_line = self._line_of[self._first_indices]
        second_line = self._line_of[self._second_indices]
        first_position = self._position_of[self._first_indices]
        second_position = self._position_of[self._second_indices]

        line_blocks = np.zeros((self._line_count, self._line_length, self._line_length))
        line_blocks[self._line_of, self._position_of, self._position_of] = self._degree
        along = first_line == second_line
        for rows, columns in (
            (first_position, second_position),
            (second_position, first_position),
        ):
            np.add.at(
                line_blocks,
                (first_line[along], rows[along], columns[along]),
                -self._weights[along],
            )
        # Offsets across lines join tiles at the same position, so they are stored
        # with the later of the two lines.
        couplings = np.zeros((self._line_count, self._line_length))
        across = ~along
        np.add.at(
            couplings,
            (
                np.maximum(first_line[across], second_line[across]),
                first_position[across],
            ),
            -self._weights[across],
        )
        pull = np.zeros((self._line_count, self._line_length, 2))
        pull[self._line_of, self._position_of] = self._offset_pull

        anchor_line = self._line_of[self._anchor_index]
        anchor_position = self._position_of[self._anchor_index]
        line_blocks[anchor_line, anchor_position, :] = 0.0
        line_blocks[anchor_line, :, anchor_position] = 0.0
        line_blocks[anchor_line, anchor_position, anchor_position] = 1.0
        couplings[anchor_line, anchor_position] = 0.0
        if anchor_line + 1 < self._line_count:
            couplings[anchor_line + 1, anchor_position] = 0.0
        return line_blocks, couplings, pull
