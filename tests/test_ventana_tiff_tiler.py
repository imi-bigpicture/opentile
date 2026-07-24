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

import pytest
from tifffile import PHOTOMETRIC

from opentile.formats import VentanaTiffTiler
from opentile.formats.ventana.ventana_tiff_metadata import VentanaMetadata
from opentile.geometry import Size, SizeMm
from opentile.tiff_image import OverlappingLevelTiffImage

from .filepaths import ventana_file_path


@pytest.fixture()
def metadata() -> VentanaMetadata:
    """A VentanaMetadata built from a multi-area XMP string, without an on-disk file."""
    xmp = """<EncodeInfo>
<iScan Magnification="40" ScanRes="0.2325" UnitNumber="X"/>
<AoiOrigin>
<AOI0 OriginX="4096" OriginY="2720"/>
<AOI1 OriginX="0" OriginY="0"/>
<AOI2 OriginX="10240" OriginY="0"/>
</AoiOrigin>
<SlideStitchInfo>
<ImageInfo AOIScanned="1" Width="1024" Height="1360" NumCols="2" NumRows="2">
<TileJointInfo Tile1="1" Tile2="2" Direction="RIGHT" OverlapX="100" FlagJoined="1" Confidence="99"/>
<TileJointInfo Tile1="1" Tile2="3" Direction="UP" OverlapY="80" FlagJoined="1" Confidence="99"/>
<TileJointInfo Tile1="1" Tile2="2" Direction="RIGHT" OverlapX="999" FlagJoined="1" Confidence="10"/>
</ImageInfo>
<ImageInfo AOIScanned="0" Width="1024" Height="1360" NumCols="5" NumRows="5"/>
<ImageInfo AOIScanned="1" Width="1024" Height="1360" NumCols="3" NumRows="1"/>
</SlideStitchInfo>
</EncodeInfo>"""

    page = SimpleNamespace(tags={"XMP": SimpleNamespace(value=xmp)})
    return VentanaMetadata(page)  # type: ignore[arg-type]


@pytest.mark.unittest
class TestVentanaMetadataAreas:
    def test_skips_unscanned_areas(self, metadata: VentanaMetadata):
        # Arrange

        # Act
        areas = metadata.areas

        # Assert - the AOIScanned="0" area (no tiles on disk) is dropped
        assert [(a.num_cols, a.num_rows) for a in areas] == [(2, 2), (3, 1)]

    def test_origin_snapped_to_tile_grid(self, metadata: VentanaMetadata):
        # Arrange

        # Act - scanned areas keep document order and match AoiOrigin by that index
        first, second = metadata.areas

        # Assert - origins snap to the tile grid: 4096/1024=4, 2720/1360=2, 10240/1024=10
        assert (first.origin_col, first.origin_row) == (4, 2)
        assert (second.origin_col, second.origin_row) == (10, 0)

    def test_measured_overlap_filters_low_confidence(self, metadata: VentanaMetadata):
        # Arrange

        # Act
        area = metadata.areas[0]

        # Assert - the confidence=10 joint is ignored, so the trusted 100/80 remain
        assert area.col_overlaps == [100.0]
        assert area.row_overlaps == [80.0]


@pytest.fixture()
def left_down_metadata() -> VentanaMetadata:
    """Newer Roche/Ventana scanners emit LEFT/DOWN instead of RIGHT/UP. Direction names
    the axis the overlap was measured along, not the tile relation, so these must be
    read the same way (see openslide/openslide#760)."""
    xmp = """<EncodeInfo>
<iScan Magnification="40" ScanRes="0.2325" UnitNumber="X"/>
<AoiOrigin><AOI0 OriginX="0" OriginY="0"/></AoiOrigin>
<SlideStitchInfo>
<ImageInfo AOIScanned="1" Width="1024" Height="1360" NumCols="2" NumRows="2">
<TileJointInfo Tile1="1" Tile2="2" Direction="LEFT" OverlapX="100" FlagJoined="1" Confidence="99"/>
<TileJointInfo Tile1="1" Tile2="3" Direction="DOWN" OverlapY="80" FlagJoined="1" Confidence="99"/>
</ImageInfo>
</SlideStitchInfo>
</EncodeInfo>"""

    page = SimpleNamespace(tags={"XMP": SimpleNamespace(value=xmp)})
    return VentanaMetadata(page)  # type: ignore[arg-type]


@pytest.mark.unittest
class TestVentanaMetadataJointDirections:
    def test_left_and_down_are_measured_like_right_and_up(
        self, left_down_metadata: VentanaMetadata
    ):
        # Arrange

        # Act
        area = left_down_metadata.areas[0]

        # Assert - identical to the RIGHT/UP fixture's 100/80 overlaps
        assert area.col_overlaps == [100.0]
        assert area.row_overlaps == [80.0]


@pytest.fixture()
def tiler():
    try:
        with VentanaTiffTiler(ventana_file_path) as tiler:
            yield tiler
    except FileNotFoundError:
        pytest.skip("Ventana bif test file not found, skipping")


@pytest.fixture()
def level(tiler: VentanaTiffTiler):
    yield tiler.get_level(0)


@pytest.mark.unittest
class TestVentanaTiffTiler:
    @pytest.mark.parametrize(
        ["tile_point", "hash"],
        [
            ((50, 35), "e01c69417daa95bff8ba66df19dc1849"),
            ((60, 40), "c49fe7975d114989c86daa8443981ad1"),
        ],
    )
    def test_get_tile(
        self,
        level: OverlappingLevelTiffImage,
        tile_point: tuple[int, int],
        hash: str,
    ):
        # Arrange

        # Act
        tile = level.get_tile(tile_point)

        # Assert
        assert md5(tile).hexdigest() == hash

    def test_photometric_interpretation(self, level: OverlappingLevelTiffImage):
        # Arrange

        # Act
        photometric_interpretation = level.photometric_interpretation

        # Assert
        assert photometric_interpretation == PHOTOMETRIC.YCBCR

    def test_level_count(self, tiler: VentanaTiffTiler):
        # Arrange

        # Act
        levels = tiler.levels

        # Assert
        assert len(levels) == 10

    def test_magnification(self, tiler: VentanaTiffTiler):
        # Arrange

        # Act
        magnification = tiler.metadata.magnification

        # Assert
        assert magnification == 40.0

    def test_scanner_serial_number(self, tiler: VentanaTiffTiler):
        # Arrange

        # Act
        serial_number = tiler.metadata.scanner_serial_number

        # Assert
        assert serial_number == "BI10N0306"

    @pytest.mark.parametrize(
        ["level_index", "expected_pyramid_index", "expected_composed_size"],
        [
            (0, 0, Size(105823, 93925)),
            (1, 1, Size(52912, 46962)),
        ],
    )
    def test_composed_size(
        self,
        tiler: VentanaTiffTiler,
        level_index: int,
        expected_pyramid_index: int,
        expected_composed_size: Size,
    ):
        # Arrange
        level = tiler.get_level(level_index)

        # Act
        overlap = level.overlap

        # Assert - reduced levels are downsamples of the raw mosaic, so pyramid scale
        # is exact even though the de-overlapped sizes do not halve exactly
        assert overlap is not None
        assert level.pyramid_index == expected_pyramid_index
        assert overlap.image_size == expected_composed_size

    def test_pixel_spacing(self, level: OverlappingLevelTiffImage):
        # Arrange

        # Act
        pixel_spacing = level.pixel_spacing

        # Assert
        assert pixel_spacing == SizeMm(0.0002325, 0.0002325)

    def test_has_label_and_thumbnail(self, tiler: VentanaTiffTiler):
        # Arrange

        # Act
        labels = tiler.labels
        thumbnails = tiler.thumbnails

        # Assert
        assert len(labels) == 1
        assert len(thumbnails) == 1
