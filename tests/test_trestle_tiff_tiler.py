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

from hashlib import md5
from types import SimpleNamespace

import numpy as np
import opentile_jpeg
import pytest
from imagecodecs import jpeg8_decode
from tifffile import PHOTOMETRIC

from opentile.exceptions import MissingAssociatedImageError
from opentile.formats import TrestleTiffTiler
from opentile.formats.trestle.trestle_image import (
    TrestleOverlappingImage,
    TrestleRegriddedImage,
    TrestleTiledImage,
)
from opentile.geometry import Point, PointF, Size, SizeMm
from opentile.tiff_image_bases import (
    OverlappingLevelTiffImage,
)
from opentile.tile_overlap import TileOverlap
from tests.filepaths import trestle_file_path


@pytest.fixture()
def tiler():
    try:
        with TrestleTiffTiler(trestle_file_path) as tiler:
            yield tiler
    except FileNotFoundError:
        pytest.skip("Trestle tiff test file not found, skipping")


@pytest.fixture()
def level(tiler: TrestleTiffTiler):
    yield tiler.get_level(0)


@pytest.fixture()
def stored_level(level: TrestleRegriddedImage):
    """The base level as stored: the overlapping tiles that the re-gridded level is
    made from, with their placement."""
    yield level._source


@pytest.fixture()
def not_regriddable(monkeypatch: pytest.MonkeyPatch):
    """Make every level not losslessly re-griddable, as for a file whose JPEG is not
    baseline. CMU-1 is baseline with an MCU-aligned step, so all its levels re-grid."""
    monkeypatch.setattr(
        TrestleOverlappingImage,
        "_lossless_regrid_step",
        lambda self, tile_size: None,
    )


@pytest.mark.unittest
class TestTrestleTiffTiler:
    @pytest.mark.parametrize(
        ["tile_point", "hash"],
        [
            ((50, 50), "1fab49bd7f594f55757af2722c25d84f"),
            ((40, 30), "c1429527aafc7802f3328db79e0dc43b"),
        ],
    )
    def test_get_tile(
        self,
        level: TrestleRegriddedImage,
        tile_point: tuple[int, int],
        hash: str,
    ):
        # Arrange

        # Act
        tile = level.get_tile(tile_point)

        # Assert
        assert md5(tile).hexdigest() == hash

    def test_photometric_interpretation(self, level: TrestleRegriddedImage):
        # Arrange

        # Act
        photometric_interpretation = level.photometric_interpretation

        # Assert
        assert photometric_interpretation == PHOTOMETRIC.YCBCR

    def test_samples_per_pixel(self, level: TrestleRegriddedImage):
        # Arrange

        # Act
        samples_per_pixel = level.samples_per_pixel

        # Assert
        assert samples_per_pixel == 3

    def test_magnification(self, tiler: TrestleTiffTiler):
        # Arrange

        # Act
        magnification = tiler.metadata.magnification

        # Assert
        assert magnification == 10.0

    def test_level_count(self, tiler: TrestleTiffTiler):
        # Arrange

        # Act
        levels = tiler.levels

        # Assert
        assert len(levels) == 7

    @pytest.mark.parametrize(
        ["level_index", "expected_pyramid_index", "expected_image_size"],
        [
            (0, 0, Size(40000, 27712)),
            (1, 1, Size(20000, 13856)),
            (2, 2, Size(10000, 6928)),
        ],
    )
    def test_overlapping_levels_are_regridded(
        self,
        tiler: TrestleTiffTiler,
        level_index: int,
        expected_pyramid_index: int,
        expected_image_size: Size,
    ):
        # Arrange
        level = tiler.get_level(level_index)

        # Act
        image_size = level.image_size

        # Assert - served as a regular non-overlapping grid of the de-overlapped size
        assert isinstance(level, TrestleRegriddedImage)
        assert level.overlap is None
        assert level.pyramid_index == expected_pyramid_index
        assert image_size == expected_image_size
        assert level.tile_size == Size(512, 512)
        assert level.tiled_size == expected_image_size.ceil_div(Size(512, 512))

    def test_all_levels_share_requested_tile_size(self, tiler: TrestleTiffTiler):
        # Arrange - levels 3-6 store 256 px tiles without overlap; at the requested
        # 512 they are re-gridded too (2x2 stored tiles per tile, by block copy)

        # Act
        levels = tiler.levels

        # Assert
        assert all(level.tile_size == Size(512, 512) for level in levels)
        assert all(level.overlap is None for level in levels)
        assert [level.pyramid_index for level in levels] == list(range(7))

    @pytest.mark.parametrize("level_index", [3, 4, 5, 6])
    def test_levels_without_overlap_at_stored_tile_size_serve_stored_tiles(
        self, level_index: int
    ):
        # Arrange
        try:
            tiler = TrestleTiffTiler(trestle_file_path, tile_size=256)
        except FileNotFoundError:
            pytest.skip("Trestle tiff test file not found, skipping")

        # Act
        with tiler:
            level = tiler.get_level(level_index)

            # Assert
            assert isinstance(level, TrestleTiledImage)
            assert level.overlap is None
            assert level.tile_size == Size(256, 256)
            assert level.pyramid_index == level_index
            assert level.scale == 2**level_index
            assert level.origin_offset == PointF(32, 32)

    def test_pixel_spacing(self, level: TrestleRegriddedImage):
        # Arrange

        # Act
        pixel_spacing = level.pixel_spacing

        # Assert
        assert pixel_spacing == SizeMm(0.000574691891670227, 0.0005750624537467957)

    @pytest.mark.parametrize("level_index", range(7))
    def test_pixel_spacing_is_power_of_two_of_base(
        self, tiler: TrestleTiffTiler, level_index: int
    ):
        # Arrange - levels 3-6 are cropped (4992 px wide, not 40000 / 8 = 5000), so
        # their sizes do not give the downsample exactly
        base_spacing = tiler.get_level(0).pixel_spacing
        level = tiler.get_level(level_index)

        # Act
        pixel_spacing = level.pixel_spacing

        # Assert
        assert level.scale == 2**level_index
        assert pixel_spacing == base_spacing * 2**level_index

    @pytest.mark.parametrize(
        ["level_index", "expected_offset"],
        [
            (0, PointF(0, 0)),
            (1, PointF(0, 0)),
            (2, PointF(0, 0)),
            (3, PointF(32, 32)),
            (4, PointF(32, 32)),
            (5, PointF(32, 32)),
            (6, PointF(32, 32)),
        ],
    )
    def test_origin_offset(
        self, tiler: TrestleTiffTiler, level_index: int, expected_offset: PointF
    ):
        # Arrange - levels 3-6 cover 64 base-level pixels less than the base level in
        # each direction (the base level's overlap), cropped equally on both sides
        level = tiler.get_level(level_index)

        # Act
        origin_offset = level.origin_offset

        # Assert
        assert origin_offset == expected_offset

    @pytest.mark.usefixtures("not_regriddable")
    def test_origin_offset_when_not_regridded(self):
        # Arrange - the fallback levels report the same offset
        try:
            tiler = TrestleTiffTiler(trestle_file_path)
        except FileNotFoundError:
            pytest.skip("Trestle tiff test file not found, skipping")

        # Act
        with tiler:
            offsets = [level.origin_offset for level in tiler.levels]

        # Assert
        assert offsets == [PointF(0, 0)] * 3 + [PointF(32, 32)] * 4

    @pytest.mark.usefixtures("not_regriddable")
    def test_levels_not_regridded_keep_requested_tile_size(self):
        # Arrange - no level can be re-gridded losslessly; the plain 256 px levels
        # are served with a zero-overlap placement rather than at their stored tile
        # size
        try:
            tiler = TrestleTiffTiler(trestle_file_path)
        except FileNotFoundError:
            pytest.skip("Trestle tiff test file not found, skipping")

        # Act
        with tiler:
            levels = tiler.levels

            # Assert
            assert all(isinstance(level, OverlappingLevelTiffImage) for level in levels)
            assert all(level.overlap is not None for level in levels)

    @pytest.mark.parametrize(
        ["sizes", "scales", "expected"],
        [
            # CMU-1 widths: levels 3-6 cropped by the base overlap, 64 base pixels
            (
                [40000, 20000, 10000, 4992, 2496, 1248, 624],
                [1, 2, 4, 8, 16, 32, 64],
                [0, 0, 0, 32, 32, 32, 32],
            ),
            # the crop is under a level pixel at downsample 128, but the size matches it
            ([40000, 4992, 312], [1, 8, 128], [0, 32, 32]),
            # an odd base size rounded down is not a crop: no offset
            ([40001, 20000, 10000], [1, 2, 4], [0, 0, 0]),
            # nor rounded up
            ([40001, 20001, 10001], [1, 2, 4], [0, 0, 0]),
        ],
    )
    def test_axis_origin_offsets(
        self, sizes: list[int], scales: list[float], expected: list[float]
    ):
        # Arrange
        base = sizes[0]

        # Act
        offsets = TrestleTiffTiler._axis_origin_offsets(base, sizes, scales)

        # Assert
        assert offsets == expected

    def test_lossless_regrid_step_of_single_tile_column(
        self, level: TrestleRegriddedImage
    ):
        # Arrange - a plain grid one stored tile wide has no right neighbour to
        # give the step; its single column sits at 0, so the stored width serves
        stored = level._source
        single_column = SimpleNamespace(
            compression=stored.compression,
            encoded_info=stored.encoded_info,
            tile_size=Size(256, 256),
            tiled_size=Size(1, 3),
            overlap=TileOverlap.from_regular_grid(
                Size(200, 700), Size(256, 256), Size(0, 0)
            ),
        )

        # Act
        step = TrestleOverlappingImage._lossless_regrid_step(
            single_column,  # type: ignore[arg-type]
            Size(512, 512),
        )

        # Assert
        assert step == Size(256, 256)

    def test_lossless_regrid_step(self, level: TrestleRegriddedImage):
        # Arrange - 448x352 stored tiles overlapping by 64 px, 4:2:0 (16 px MCU)
        stored = level._source

        # Act
        step = stored._lossless_regrid_step(Size(512, 512))
        off_grid = stored._lossless_regrid_step(Size(500, 500))

        # Assert
        assert step == Size(384, 288)
        assert off_grid is None

    @pytest.mark.parametrize(
        ["level_index", "tile_point"],
        [(0, Point(40, 30)), (4, Point(1, 1))],
    )
    def test_regridded_tiles_copy_stored_coefficients(
        self, tiler: TrestleTiffTiler, level_index: int, tile_point: Point
    ):
        # Arrange - every DCT block of a served tile comes from the stored tile owning
        # that position on the de-overlapped canvas: the tile at grid cell g keeps
        # [g * step, (g + 1) * step). Level 0 overlaps; level 4 is a plain 256 px grid.
        level = tiler.get_level(level_index)
        assert isinstance(level, TrestleRegriddedImage)
        step = level._step
        served, _ = opentile_jpeg.decode_coefficients(
            level.get_tile(tile_point.to_tuple())
        )
        mcu = 16
        origin_x, origin_y = tile_point.x * 512, tile_point.y * 512

        # Act / Assert - compare luma per 16 px MCU (2x2 blocks)
        for row in range(0, 512, mcu):
            for column in range(0, 512, mcu):
                x, y = origin_x + column, origin_y + row
                owner = Point(x // step.width, y // step.height)
                stored, _ = opentile_jpeg.decode_coefficients(
                    level._source.get_tile(owner.to_tuple())
                )
                sx, sy = x - owner.x * step.width, y - owner.y * step.height
                np.testing.assert_array_equal(
                    served[0][row // 8 : row // 8 + 2, column // 8 : column // 8 + 2],
                    stored[0][sy // 8 : sy // 8 + 2, sx // 8 : sx // 8 + 2],
                )

    def test_all_tiles_decode_to_tile_size(self, tiler: TrestleTiffTiler):
        # Arrange
        level = tiler.get_level(2)

        # Act
        shapes = {jpeg8_decode(tile).shape for tile in level.get_all_tiles()}

        # Assert
        assert shapes == {(512, 512, 3)}

    def test_edge_tile_is_white_past_image(self, level: TrestleRegriddedImage):
        # Arrange - the last column ends 40000 - 78 * 512 = 64 px into its tile, an
        # MCU boundary, so everything past the image is filled
        last = Point(level.tiled_size.width - 1, level.tiled_size.height - 1)
        inside_width = level.image_size.width - last.x * 512
        inside_height = level.image_size.height - last.y * 512

        # Act
        tile = jpeg8_decode(level.get_tile(last.to_tuple()))

        # Assert
        assert tile.shape == (512, 512, 3)
        assert (tile[:, inside_width:] == 255).all()
        assert (tile[inside_height:, :] == 255).all()

    def test_get_tiles_matches_get_tile(self, level: TrestleRegriddedImage):
        # Arrange - a batch larger than one compose call, out of raster order
        positions = [(x, y) for y in (31, 30) for x in range(30, 40)]

        # Act
        batch = list(level.get_tiles(positions))

        # Assert
        assert batch == [level.get_tile(position) for position in positions]

    def test_tile_outside_level_raises(self, level: TrestleRegriddedImage):
        # Arrange
        outside = (level.tiled_size.width, 0)

        # Act / Assert
        with pytest.raises(ValueError):
            level.get_tile(outside)

    def test_stored_tile_positions_de_overlap(
        self, stored_level: OverlappingLevelTiffImage
    ):
        # Arrange
        overlap = stored_level.overlap
        tile_width, tile_height = stored_level.tile_size.to_tuple()

        # Act - each stored tile contributes a single whole-tile piece
        origin = overlap.placements[Point(0, 0)][0].position
        right = overlap.placements[Point(1, 0)][0].position
        down = overlap.placements[Point(0, 1)][0].position

        # Assert - each tile advances by (tile - overlap), overlap is 64 on base level
        assert origin == PointF(0, 0)
        assert right == PointF(tile_width - 64, 0)
        assert down == PointF(0, tile_height - 64)

    def test_overlapping_tiles_share_pixels(
        self, stored_level: OverlappingLevelTiffImage
    ):
        # Arrange - a tissue tile and its right neighbour
        overlap_x = 64
        tile_width = stored_level.tile_size.width
        left = stored_level.get_decoded_tile((30, 40))
        right = stored_level.get_decoded_tile((31, 40))

        # Act - the right edge of a tile covers the same pixels as the left edge of
        # its neighbour, so de-overlapping by 64 px aligns them
        overlap_difference = np.abs(
            left[:, tile_width - overlap_x :].astype(int) - right[:, :overlap_x]
        ).mean()

        # Assert
        assert overlap_difference < 1.0

    @pytest.mark.parametrize("get", ["get_label", "get_overview", "get_thumbnail"])
    def test_associated_image_raises_missing(self, tiler: TrestleTiffTiler, get: str):
        # Arrange - Trestle stores associated images in separate files

        # Act, Assert
        with pytest.raises(MissingAssociatedImageError):
            getattr(tiler, get)()

    def test_has_no_associated_images(self, tiler: TrestleTiffTiler):
        # Act
        associated = [tiler.labels, tiler.overviews, tiler.thumbnails]

        # Assert
        assert associated == [[], [], []]
