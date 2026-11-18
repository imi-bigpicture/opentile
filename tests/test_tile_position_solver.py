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

import numpy as np
import pytest

from opentile.geometry import Point, PointF
from opentile.tile_position_solver import TileOffset, TilePositionSolver


def grid(cols: int, rows: int) -> list[Point]:
    return [Point(col, row) for row in range(rows) for col in range(cols)]


def random_grid_offsets(
    cols: int, rows: int, seed: int, weak_weight: float = 0.01
) -> list[TileOffset]:
    """Neighbour offsets on a grid: noisy measurements on some joints with weights
    near 1, weak nominal links on the rest, like the Ventana tiler builds."""
    rng = np.random.default_rng(seed)
    offsets = []
    for cell in grid(cols, rows):
        for neighbour, pitch in (
            (Point(cell.x + 1, cell.y), (900.0, 0.0)),
            (Point(cell.x, cell.y + 1), (0.0, 1100.0)),
        ):
            if neighbour.x >= cols or neighbour.y >= rows:
                continue
            if rng.random() < 0.6:
                noise = rng.normal(0, 3, 2)
                weight = rng.uniform(0.8, 1.0)
            else:
                noise = np.zeros(2)
                weight = weak_weight
            offsets.append(
                TileOffset(cell, neighbour, PointF(*(np.array(pitch) + noise)), weight)
            )
    return offsets


def dense_solve(
    tiles: list[Point], offsets: list[TileOffset], anchor: Point
) -> dict[Point, PointF]:
    """Reference solution: the normal equations solved directly."""
    index = {tile: i for i, tile in enumerate(tiles)}
    laplacian = np.zeros((len(tiles), len(tiles)))
    pull = np.zeros((len(tiles), 2))
    for offset in offsets:
        i, j = index[offset.first], index[offset.second]
        laplacian[i, i] += offset.weight
        laplacian[j, j] += offset.weight
        laplacian[i, j] -= offset.weight
        laplacian[j, i] -= offset.weight
        pull[j] += offset.weight * np.array([offset.offset.x, offset.offset.y])
        pull[i] -= offset.weight * np.array([offset.offset.x, offset.offset.y])
    free = [i for i in range(len(tiles)) if i != index[anchor]]
    solution = np.zeros((len(tiles), 2))
    solution[free] = np.linalg.solve(laplacian[np.ix_(free, free)], pull[free])
    return {tile: PointF(*solution[i]) for tile, i in index.items()}


def assert_positions_close(
    actual: dict[Point, PointF], expected: dict[Point, PointF], abs: float
):
    assert actual.keys() == expected.keys()
    for tile in expected:
        assert actual[tile].x == pytest.approx(expected[tile].x, abs=abs)
        assert actual[tile].y == pytest.approx(expected[tile].y, abs=abs)


@pytest.mark.unittest
class TestTilePositionSolver:
    def test_consistent_offsets_are_reproduced(self):
        # Arrange - a 2x2 grid whose offsets agree around the loop
        offsets = [
            TileOffset(Point(0, 0), Point(1, 0), PointF(900, -2)),
            TileOffset(Point(0, 1), Point(1, 1), PointF(900, -2)),
            TileOffset(Point(0, 0), Point(0, 1), PointF(3, 1200)),
            TileOffset(Point(1, 0), Point(1, 1), PointF(3, 1200)),
        ]

        # Act
        positions = TilePositionSolver(grid(2, 2), offsets, Point(0, 0)).solve()

        # Assert
        assert positions[Point(1, 1)].x == pytest.approx(903, abs=1e-9)
        assert positions[Point(1, 1)].y == pytest.approx(1198, abs=1e-9)

    def test_conflicting_offsets_meet_in_the_weighted_middle(self):
        # Arrange - two measurements of the same joint disagree by 4 px; one is
        # trusted 3x
        offsets = [
            TileOffset(Point(0, 0), Point(1, 0), PointF(100, 0), 3.0),
            TileOffset(Point(0, 0), Point(1, 0), PointF(104, 0), 1.0),
        ]

        # Act
        positions = TilePositionSolver(grid(2, 1), offsets, Point(0, 0)).solve()

        # Assert
        assert positions[Point(1, 0)].x == pytest.approx(101, abs=1e-9)

    def test_repeated_offsets_across_lines_add_up(self):
        # Arrange - two measurements of a joint between lines, trusted 3:1
        offsets = [
            TileOffset(Point(0, 0), Point(1, 0), PointF(100, 0)),
            TileOffset(Point(0, 0), Point(0, 1), PointF(0, 200), 3.0),
            TileOffset(Point(0, 0), Point(0, 1), PointF(0, 204), 1.0),
            TileOffset(Point(1, 0), Point(1, 1), PointF(0, 201)),
        ]
        tiles = grid(2, 2)

        # Act
        positions = TilePositionSolver(tiles, offsets, Point(0, 0)).solve()

        # Assert
        assert_positions_close(
            positions, dense_solve(tiles, offsets, Point(0, 0)), abs=1e-9
        )

    @pytest.mark.parametrize(
        ["cols", "rows"],
        [
            pytest.param(12, 9, id="wide"),
            pytest.param(9, 12, id="tall"),
            pytest.param(10, 10, id="square"),
            pytest.param(7, 1, id="single row"),
            pytest.param(1, 7, id="single column"),
            pytest.param(40, 25, id="large"),
        ],
    )
    @pytest.mark.parametrize("anchor_at", ["first", "middle", "last"])
    @pytest.mark.parametrize("seed", [0, 1])
    def test_matches_direct_solve(
        self, cols: int, rows: int, anchor_at: str, seed: int
    ):
        # Arrange - a grid mixing measured joints and weak nominal links
        tiles = grid(cols, rows)
        offsets = random_grid_offsets(cols, rows, seed)
        anchor = {
            "first": tiles[0],
            "middle": Point(cols // 2, rows // 2),
            "last": tiles[-1],
        }[anchor_at]
        expected = dense_solve(tiles, offsets, anchor)

        # Act
        positions = TilePositionSolver(tiles, offsets, anchor).solve()

        # Assert
        assert positions[anchor] == PointF(0, 0)
        assert_positions_close(positions, expected, abs=1e-6)

    def test_tiles_need_not_start_at_the_origin(self):
        # Arrange - the same grid, moved to start at (3, 5)
        tiles = grid(6, 4)
        offsets = random_grid_offsets(6, 4, seed=4)
        moved_tiles = [Point(tile.x + 3, tile.y + 5) for tile in tiles]
        moved_offsets = [
            TileOffset(
                Point(o.first.x + 3, o.first.y + 5),
                Point(o.second.x + 3, o.second.y + 5),
                o.offset,
                o.weight,
            )
            for o in offsets
        ]

        # Act
        positions = TilePositionSolver(tiles, offsets, tiles[0]).solve()
        moved = TilePositionSolver(moved_tiles, moved_offsets, moved_tiles[0]).solve()

        # Assert
        assert_positions_close(
            {Point(t.x - 3, t.y - 5): p for t, p in moved.items()},
            positions,
            abs=1e-9,
        )

    def test_single_tile(self):
        # Arrange / Act
        positions = TilePositionSolver([Point(0, 0)], [], Point(0, 0)).solve()

        # Assert
        assert positions == {Point(0, 0): PointF(0, 0)}

    def test_positions_are_plain_floats(self):
        # Arrange
        offsets = [TileOffset(Point(0, 0), Point(1, 0), PointF(100, 2))]

        # Act
        positions = TilePositionSolver(grid(2, 1), offsets, Point(0, 0)).solve()

        # Assert
        position = positions[Point(1, 0)]
        assert type(position.x) is float and type(position.y) is float

    def test_layout_is_checked_before_connectivity(self):
        # Arrange - not a rectangle, and the third tile is not linked either
        tiles = [Point(0, 0), Point(1, 0), Point(0, 1)]
        offsets = [TileOffset(Point(0, 0), Point(1, 0), PointF(1, 0))]

        # Act / Assert
        with pytest.raises(ValueError, match="do not fill"):
            TilePositionSolver(tiles, offsets, Point(0, 0))

    @pytest.mark.parametrize(
        ["tiles", "offsets", "match"],
        [
            pytest.param(
                [Point(0, 0), Point(1, 0), Point(0, 0)],
                [TileOffset(Point(0, 0), Point(1, 0), PointF(1, 0))],
                "more than once",
                id="duplicate tile",
            ),
            pytest.param([Point(1, 0)], [], "Anchor", id="anchor missing"),
            pytest.param(
                grid(2, 1),
                [TileOffset(Point(0, 0), Point(5, 5), PointF(1, 0))],
                "not among the tiles",
                id="unknown tile",
            ),
            pytest.param(
                [Point(0, 0), Point(1, 0), Point(0, 1)],
                [
                    TileOffset(Point(0, 0), Point(1, 0), PointF(1, 0)),
                    TileOffset(Point(0, 0), Point(0, 1), PointF(0, 1)),
                ],
                "do not fill",
                id="not a rectangle",
            ),
            pytest.param(
                grid(3, 1),
                [
                    TileOffset(Point(0, 0), Point(1, 0), PointF(1, 0)),
                    TileOffset(Point(0, 0), Point(2, 0), PointF(2, 0)),
                ],
                "not direct neighbours",
                id="distant offset",
            ),
            pytest.param(
                grid(2, 2),
                [
                    TileOffset(Point(0, 0), Point(1, 0), PointF(1, 0)),
                    TileOffset(Point(0, 0), Point(1, 1), PointF(1, 1)),
                    TileOffset(Point(0, 0), Point(0, 1), PointF(0, 1)),
                ],
                "not direct neighbours",
                id="diagonal offset",
            ),
            pytest.param(
                grid(2, 2),
                [
                    TileOffset(Point(0, 0), Point(1, 0), PointF(1, 0)),
                    TileOffset(Point(0, 1), Point(1, 1), PointF(1, 0)),
                ],
                "not linked",
                id="two separate groups",
            ),
            pytest.param(
                grid(2, 1),
                [TileOffset(Point(0, 0), Point(1, 0), PointF(1, 0), 0.0)],
                "not linked",
                id="zero weight",
            ),
            pytest.param(
                grid(2, 1),
                [TileOffset(Point(0, 0), Point(1, 0), PointF(1, 0), -0.5)],
                "at least 0",
                id="negative weight",
            ),
            pytest.param(
                grid(2, 1),
                [TileOffset(Point(0, 0), Point(1, 0), PointF(1, 0), float("nan"))],
                "at least 0",
                id="nan weight",
            ),
            pytest.param(
                grid(2, 1),
                [TileOffset(Point(0, 0), Point(1, 0), PointF(1, 0), float("inf"))],
                "at least 0",
                id="infinite weight",
            ),
            pytest.param(
                grid(2, 1),
                [TileOffset(Point(0, 0), Point(1, 0), PointF(float("nan"), 0))],
                "must be finite",
                id="nan offset",
            ),
            pytest.param(
                grid(2, 1),
                [TileOffset(Point(0, 0), Point(1, 0), PointF(0, float("-inf")))],
                "must be finite",
                id="infinite offset",
            ),
        ],
    )
    def test_invalid_input_is_an_error(
        self, tiles: list[Point], offsets: list[TileOffset], match: str
    ):
        # Arrange / Act / Assert
        with pytest.raises(ValueError, match=match):
            TilePositionSolver(tiles, offsets, Point(0, 0))
