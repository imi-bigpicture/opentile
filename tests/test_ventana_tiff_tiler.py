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
from typing import Optional
from xml.etree.ElementTree import Element

import pytest
from defusedxml.ElementTree import fromstring
from tifffile import PHOTOMETRIC

from opentile.formats import VentanaTiffTiler
from opentile.formats.ventana.ventana_tiff_metadata import (
    VentanaArea,
    VentanaElement,
    VentanaMetadata,
)
from opentile.formats.ventana.ventana_tiff_tiler import VentanaBaseOverlap
from opentile.geometry import Point, PointF, Size, SizeMm
from opentile.tiff_image_bases import OverlappingLevelTiffImage
from tests.filepaths import (
    ventana_1_file_path,
    ventana_2_file_path,
    ventana_file_path,
)


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
<TileJointInfo Tile1="1" Tile2="2" Direction="RIGHT" OverlapX="100" OverlapY="0" FlagJoined="1" Confidence="99"/>
<TileJointInfo Tile1="1" Tile2="3" Direction="UP" OverlapX="0" OverlapY="80" FlagJoined="1" Confidence="99"/>
<TileJointInfo Tile1="1" Tile2="2" Direction="RIGHT" OverlapX="999" OverlapY="0" FlagJoined="1" Confidence="10"/>
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
        assert [a.tiled_size for a in areas] == [Size(2, 2), Size(3, 1)]

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

    def test_unscanned_area_needs_no_origin(self):
        # Arrange - only the scanned second area has an origin
        metadata = metadata_from_stitch_info(
            '<AoiOrigin><AOI1 OriginX="2048" OriginY="0"/></AoiOrigin>',
            [
                '<ImageInfo AOIScanned="0" Width="1024" Height="1360" NumCols="5"'
                ' NumRows="5"/>',
                '<ImageInfo AOIScanned="1" Width="1024" Height="1360" NumCols="3"'
                ' NumRows="1"/>',
            ],
        )

        # Act
        areas = metadata.areas

        # Assert
        assert [(a.origin_col, a.origin_row) for a in areas] == [(2, 0)]

    @pytest.mark.parametrize(
        ["aoi_origin", "image_info", "match"],
        [
            pytest.param(
                '<AoiOrigin><AOI0 OriginX="0" OriginY="0"/></AoiOrigin>',
                '<ImageInfo Width="1024" Height="1360" NumCols="2" NumRows="1"/>',
                "ImageInfo has no AOIScanned",
                id="no AOIScanned",
            ),
            pytest.param(
                "",
                '<ImageInfo AOIScanned="1" Width="1024" Height="1360" NumCols="2"'
                ' NumRows="1"/>',
                "no AoiOrigin block",
                id="no AoiOrigin",
            ),
            pytest.param(
                '<AoiOrigin><AOI1 OriginX="0" OriginY="0"/></AoiOrigin>',
                '<ImageInfo AOIScanned="1" Width="1024" Height="1360" NumCols="2"'
                ' NumRows="1"/>',
                "no AoiOrigin AOI0 for scanned ImageInfo 0",
                id="no origin for the area",
            ),
            pytest.param(
                '<AoiOrigin><AOI0 OriginY="0"/></AoiOrigin>',
                '<ImageInfo AOIScanned="1" Width="1024" Height="1360" NumCols="2"'
                ' NumRows="1"/>',
                "AOI0 has no OriginX",
                id="no OriginX",
            ),
            pytest.param(
                '<AoiOrigin><AOI0 OriginX="0"/></AoiOrigin>',
                '<ImageInfo AOIScanned="1" Width="1024" Height="1360" NumCols="2"'
                ' NumRows="1"/>',
                "AOI0 has no OriginY",
                id="no OriginY",
            ),
        ],
    )
    def test_invalid_area_metadata_is_an_error(
        self, aoi_origin: str, image_info: str, match: str
    ):
        # Arrange
        metadata = metadata_from_stitch_info(aoi_origin, [image_info])

        # Act / Assert
        with pytest.raises(ValueError, match=match):
            _ = metadata.areas


def metadata_from_stitch_info(
    aoi_origin: str, image_infos: list[str]
) -> VentanaMetadata:
    """A VentanaMetadata from an AoiOrigin block and ImageInfo elements."""
    xmp = f"""<EncodeInfo>
<iScan Magnification="40" ScanRes="0.2325" UnitNumber="X"/>
{aoi_origin}
<SlideStitchInfo>{"".join(image_infos)}</SlideStitchInfo>
</EncodeInfo>"""
    page = SimpleNamespace(tags={"XMP": SimpleNamespace(value=xmp)})
    return VentanaMetadata(page)  # type: ignore[arg-type]


@pytest.mark.unittest
class TestVentanaArea:
    def test_from_image_info(self):
        # Arrange - snake tiles 1, 2 are the bottom row, 3 sits above 2
        image_info = fromstring(
            '<ImageInfo Width="1024" Height="1360" NumCols="2" NumRows="2">'
            '<TileJointInfo Tile1="1" Tile2="2" Direction="RIGHT" OverlapX="100"'
            ' OverlapY="2" FlagJoined="1" Confidence="99"/>'
            '<TileJointInfo Tile1="2" Tile2="3" Direction="UP" OverlapX="-3"'
            ' OverlapY="80" FlagJoined="1" Confidence="99"/>'
            "</ImageInfo>"
        )

        # Act
        area = VentanaArea.from_image_info(
            VentanaElement(image_info), PointF(4096, 2720)
        )

        # Assert - the origin is on the tile grid: 4096/1024=4, 2720/1360=2
        assert area.tiled_size == Size(2, 2)
        assert (area.origin_col, area.origin_row) == (4, 2)
        assert area.col_overlaps == [100.0]
        assert area.row_overlaps == [80.0]
        assert {(j.first, j.second): j.offset for j in area.joints} == {
            (Point(0, 1), Point(1, 1)): PointF(924.0, -2.0),
            (Point(1, 0), Point(1, 1)): PointF(3.0, 1280.0),
        }

    @pytest.mark.parametrize("attribute", ["Width", "Height", "NumCols", "NumRows"])
    @pytest.mark.parametrize(
        ["value", "match"],
        [
            pytest.param(None, "has no {attribute}", id="missing"),
            pytest.param("0", "{attribute} must be positive", id="zero"),
        ],
    )
    def test_invalid_area_size_is_an_error(
        self, attribute: str, value: Optional[str], match: str
    ):
        # Arrange - a valid area with one attribute removed or set
        attributes = {"Width": "1024", "Height": "1360", "NumCols": "2", "NumRows": "1"}
        if value is None:
            del attributes[attribute]
        else:
            attributes[attribute] = value
        image_info = Element("ImageInfo", attributes)

        # Act / Assert
        with pytest.raises(ValueError, match=match.format(attribute=attribute)):
            VentanaArea.from_image_info(VentanaElement(image_info), PointF(0, 0))

    @pytest.mark.parametrize(
        "origin",
        [
            pytest.param(PointF(1000, 0), id="column"),
            pytest.param(PointF(0, 1400), id="row"),
            pytest.param(PointF(1024.5, 0), id="fraction"),
        ],
    )
    def test_origin_off_tile_grid_is_an_error(self, origin: PointF):
        # Arrange
        image_info = Element(
            "ImageInfo",
            {"Width": "1024", "Height": "1360", "NumCols": "2", "NumRows": "1"},
        )

        # Act / Assert
        with pytest.raises(ValueError, match="is not on the grid of its 1024x1360"):
            VentanaArea.from_image_info(VentanaElement(image_info), origin)

    @pytest.mark.parametrize(
        "attribute",
        ["FlagJoined", "Tile1", "Tile2", "Direction", "OverlapX", "OverlapY"],
    )
    @pytest.mark.parametrize("confidence", ["99", "10"])
    def test_missing_joint_attribute_is_an_error(self, attribute: str, confidence: str):
        # Arrange - a valid joint with one attribute removed. Both a confidence that
        # counts towards the mean overlaps and one that does not are tried, as the
        # overlaps and the joint are read from different attributes.
        image_info = Element(
            "ImageInfo",
            {"Width": "1024", "Height": "1360", "NumCols": "2", "NumRows": "1"},
        )
        joint_attributes = {
            "Tile1": "1",
            "Tile2": "2",
            "Direction": "RIGHT",
            "OverlapX": "100",
            "OverlapY": "2",
            "FlagJoined": "1",
            "Confidence": confidence,
        }
        del joint_attributes[attribute]
        image_info.append(Element("TileJointInfo", joint_attributes))

        # Act / Assert
        with pytest.raises(ValueError, match=f"TileJointInfo has no {attribute}"):
            VentanaArea.from_image_info(VentanaElement(image_info), PointF(0, 0))

    def test_missing_joint_confidence_is_an_error(self):
        # Arrange
        image_info = Element(
            "ImageInfo",
            {"Width": "1024", "Height": "1360", "NumCols": "2", "NumRows": "1"},
        )
        image_info.append(
            Element(
                "TileJointInfo",
                {
                    "Tile1": "1",
                    "Tile2": "2",
                    "Direction": "RIGHT",
                    "OverlapX": "100",
                    "OverlapY": "2",
                    "FlagJoined": "1",
                },
            )
        )

        # Act / Assert
        with pytest.raises(ValueError, match="TileJointInfo has no Confidence"):
            VentanaArea.from_image_info(VentanaElement(image_info), PointF(0, 0))


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
<TileJointInfo Tile1="1" Tile2="2" Direction="LEFT" OverlapX="100" OverlapY="0" FlagJoined="1" Confidence="99"/>
<TileJointInfo Tile1="1" Tile2="3" Direction="DOWN" OverlapX="0" OverlapY="80" FlagJoined="1" Confidence="99"/>
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
def joint_metadata() -> VentanaMetadata:
    """A 2x2 area whose joints measure both the overlap and the shift across it."""
    xmp = """<EncodeInfo>
<iScan Magnification="40" ScanRes="0.2325" UnitNumber="X"/>
<AoiOrigin><AOI0 OriginX="0" OriginY="0"/></AoiOrigin>
<SlideStitchInfo>
<ImageInfo AOIScanned="1" Width="1024" Height="1360" NumCols="2" NumRows="2">
<TileJointInfo Tile1="1" Tile2="2" Direction="RIGHT" OverlapX="100" OverlapY="2" FlagJoined="1" Confidence="99"/>
<TileJointInfo Tile1="1" Tile2="4" Direction="UP" OverlapX="-3" OverlapY="80" FlagJoined="1" Confidence="85"/>
<TileJointInfo Tile1="2" Tile2="3" Direction="UP" OverlapX="-3" OverlapY="80" FlagJoined="0" Confidence="0"/>
<TileJointInfo Tile1="3" Tile2="4" Direction="RIGHT" OverlapX="100" OverlapY="2" FlagJoined="1" Confidence="0"/>
</ImageInfo>
</SlideStitchInfo>
</EncodeInfo>"""
    page = SimpleNamespace(tags={"XMP": SimpleNamespace(value=xmp)})
    return VentanaMetadata(page)  # type: ignore[arg-type]


@pytest.mark.unittest
class TestVentanaMetadataJoints:
    def test_joint_offsets_include_the_shift_across_the_joint(
        self, joint_metadata: VentanaMetadata
    ):
        # Arrange - snake tiles 1, 2 are the bottom row; 4 sits above 1

        # Act
        joints = {(j.first, j.second): j for j in joint_metadata.areas[0].joints}

        # Assert - right of: (width - OverlapX, -OverlapY); below: (-OverlapX,
        # height - OverlapY); an unjoined joint and a joined joint of confidence 0
        # are dropped; confidence 85 is kept
        assert joints[(Point(0, 1), Point(1, 1))].offset == PointF(924.0, -2.0)
        assert joints[(Point(0, 0), Point(0, 1))].offset == PointF(3.0, 1280.0)
        assert joints[(Point(0, 0), Point(0, 1))].confidence == 85.0
        assert len(joints) == 2


@pytest.fixture()
def tiler():
    try:
        with VentanaTiffTiler(ventana_file_path) as tiler:
            yield tiler
    except FileNotFoundError:
        pytest.skip("Ventana bif test file not found, skipping")


@pytest.mark.unittest
class TestVentanaBaseOverlap:
    def test_areas_on_shared_canvas(self):
        # Arrange - two areas on tile row 1, without joints (so at the full pitch)
        metadata = metadata_from_stitch_info(
            "<AoiOrigin>"
            '<AOI0 OriginX="0" OriginY="1360"/>'
            '<AOI1 OriginX="2048" OriginY="1360"/>'
            "</AoiOrigin>",
            [
                '<ImageInfo AOIScanned="1" Width="1024" Height="1360" NumCols="2"'
                ' NumRows="1"/>',
                '<ImageInfo AOIScanned="1" Width="1024" Height="1360" NumCols="1"'
                ' NumRows="1"/>',
            ],
        )

        # Act
        overlap = VentanaBaseOverlap.create(metadata, Size(1024, 1360))

        # Assert - the canvas starts at the top-left-most tile, at row 1
        assert overlap.image_size == Size(3072, 1360)
        assert {
            grid: [piece.position for piece in pieces]
            for grid, pieces in overlap.placements.items()
        } == {
            Point(0, 1): [PointF(0, 0)],
            Point(1, 1): [PointF(1024, 0)],
            Point(2, 1): [PointF(2048, 0)],
        }

    def test_overlapping_areas_are_an_error(self):
        # Arrange - the second area starts on the first area's second tile
        metadata = metadata_from_stitch_info(
            "<AoiOrigin>"
            '<AOI0 OriginX="0" OriginY="0"/>'
            '<AOI1 OriginX="1024" OriginY="0"/>'
            "</AoiOrigin>",
            [
                '<ImageInfo AOIScanned="1" Width="1024" Height="1360" NumCols="2"'
                ' NumRows="1"/>',
                '<ImageInfo AOIScanned="1" Width="1024" Height="1360" NumCols="2"'
                ' NumRows="1"/>',
            ],
        )

        # Act & Assert
        with pytest.raises(ValueError, match="overlap at 1 grid positions"):
            VentanaBaseOverlap.create(metadata, Size(1024, 1360))


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
            # tiles placed from all joints, including the shift across each joint
            # (the scan grid is slightly rotated), so the canvas is a little larger
            # than the axis-aligned 105823 x 93925 of mean overlaps alone; reduced
            # levels round the downsampled extent up (94147 / 2 = 47073.5)
            (0, 0, Size(105950, 94147)),
            (1, 1, Size(52975, 47074)),
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

    def test_associated_images(self, tiler: VentanaTiffTiler):
        # Arrange

        # Act
        overviews = tiler.overviews
        labels = tiler.labels
        thumbnails = tiler.thumbnails

        # Assert - OS-1's series are Baseline/Overview/Probability: only the "Overview"
        # is exposed (as an associated image); there is no "Label"/"Thumbnail" series and
        # the "Probability" heatmap is not exposed.
        assert len(overviews) == 1
        assert len(labels) == 0
        assert len(thumbnails) == 0

    def test_overview(self, tiler: VentanaTiffTiler):
        # Arrange
        overview = tiler.get_overview()

        # Act
        tile = overview.get_tile((0, 0))

        # Assert - the single tiled JPEG overview is served whole
        assert overview.image_size == Size(1008, 3008)
        assert md5(tile).hexdigest() == "46507b641b9f47a7725baefc1377275e"


@pytest.fixture()
def ventana_1_tiler():
    try:
        with VentanaTiffTiler(ventana_1_file_path) as tiler:
            yield tiler
    except FileNotFoundError:
        pytest.skip("Ventana-1 bif test file not found, skipping")


@pytest.fixture()
def ventana_1_level(ventana_1_tiler: VentanaTiffTiler):
    yield ventana_1_tiler.get_level(0)


@pytest.mark.unittest
class TestVentana1TiffTiler:
    """Ventana-1.bif: a single-area, LEFT-joint-direction bif (vs OS-1's RIGHT), with
    no label or thumbnail image."""

    @pytest.mark.parametrize(
        ["tile_point", "hash"],
        [
            ((0, 0), "307da7f6e767c64e7c1a6352cc8a7edc"),
            ((11, 10), "dbf7c918e336cbaebd1f89421221b33a"),
        ],
    )
    def test_get_tile(
        self,
        ventana_1_level: OverlappingLevelTiffImage,
        tile_point: tuple[int, int],
        hash: str,
    ):
        # Arrange

        # Act
        tile = ventana_1_level.get_tile(tile_point)

        # Assert
        assert md5(tile).hexdigest() == hash

    def test_photometric_interpretation(
        self, ventana_1_level: OverlappingLevelTiffImage
    ):
        # Arrange

        # Act
        photometric_interpretation = ventana_1_level.photometric_interpretation

        # Assert
        assert photometric_interpretation == PHOTOMETRIC.YCBCR

    def test_level_count(self, ventana_1_tiler: VentanaTiffTiler):
        # Arrange

        # Act
        levels = ventana_1_tiler.levels

        # Assert
        assert len(levels) == 8

    def test_single_area(self, ventana_1_tiler: VentanaTiffTiler):
        # Arrange
        assert isinstance(ventana_1_tiler.metadata, VentanaMetadata)

        # Act
        areas = ventana_1_tiler.metadata.areas

        # Assert
        assert len(areas) == 1
        assert areas[0].tiled_size == Size(23, 21)

    def test_magnification(self, ventana_1_tiler: VentanaTiffTiler):
        # Arrange

        # Act
        magnification = ventana_1_tiler.metadata.magnification

        # Assert
        assert magnification == 40.0

    def test_scanner_serial_number(self, ventana_1_tiler: VentanaTiffTiler):
        # Arrange

        # Act
        serial_number = ventana_1_tiler.metadata.scanner_serial_number

        # Assert
        assert serial_number == "2000515"

    def test_scanner_manufacturer(self, ventana_1_tiler: VentanaTiffTiler):
        # Arrange

        # Act
        manufacturer = ventana_1_tiler.metadata.scanner_manufacturer

        # Assert
        assert manufacturer == "Ventana Medical Systems, Inc."

    @pytest.mark.parametrize(
        ["level_index", "expected_pyramid_index", "expected_composed_size"],
        [
            (0, 0, Size(23432, 21504)),
            (1, 1, Size(11716, 10752)),
        ],
    )
    def test_composed_size(
        self,
        ventana_1_tiler: VentanaTiffTiler,
        level_index: int,
        expected_pyramid_index: int,
        expected_composed_size: Size,
    ):
        # Arrange
        level = ventana_1_tiler.get_level(level_index)

        # Act
        overlap = level.overlap

        # Assert
        assert overlap is not None
        assert level.pyramid_index == expected_pyramid_index
        assert overlap.image_size == expected_composed_size

    def test_pixel_spacing(self, ventana_1_level: OverlappingLevelTiffImage):
        # Arrange

        # Act
        pixel_spacing = ventana_1_level.pixel_spacing

        # Assert
        assert pixel_spacing == SizeMm(0.00025, 0.00025)

    def test_associated_images(self, ventana_1_tiler: VentanaTiffTiler):
        # Arrange

        # Act
        overviews = ventana_1_tiler.overviews
        labels = ventana_1_tiler.labels
        thumbnails = ventana_1_tiler.thumbnails

        # Assert - like OS-1, only an Overview is exposed (no label/thumbnail series)
        assert len(overviews) == 1
        assert len(labels) == 0
        assert len(thumbnails) == 0

    def test_overview(self, ventana_1_tiler: VentanaTiffTiler):
        # Arrange
        overview = ventana_1_tiler.get_overview()

        # Act
        tile = overview.get_tile((0, 0))

        # Assert - the multi-strip (uncompressed) overview is assembled whole, unlike a
        # tiled associated image which would return only the first strip
        assert overview.image_size == Size(1251, 3685)
        assert md5(tile).hexdigest() == "74821cb734b23b811c5f4f74264bd269"


@pytest.fixture()
def ventana_2_tiler():
    try:
        with VentanaTiffTiler(ventana_2_file_path) as tiler:
            yield tiler
    except FileNotFoundError:
        pytest.skip("OS-2 bif test file not found, skipping")


@pytest.fixture()
def ventana_2_level(ventana_2_tiler: VentanaTiffTiler):
    yield ventana_2_tiler.get_level(0)


@pytest.mark.unittest
class TestVentana2TiffTiler:
    """OS-2.bif: three areas, of which the middle one is not scanned. The scanned
    areas are stored on one tile grid, with blank tiles where no area is."""

    @pytest.mark.parametrize(
        ["tile_point", "hash"],
        [
            pytest.param((5, 36), "41fe3d25a9b1ecf6f8e7c9eacce95a2e", id="first area"),
            pytest.param(
                (77, 30), "3e930e26bf71f3b9d113626af676fa29", id="second area"
            ),
        ],
    )
    def test_get_tile(
        self,
        ventana_2_level: OverlappingLevelTiffImage,
        tile_point: tuple[int, int],
        hash: str,
    ):
        # Arrange

        # Act
        tile = ventana_2_level.get_tile(tile_point)

        # Assert
        assert md5(tile).hexdigest() == hash

    @pytest.mark.parametrize("tile_point", [(0, 0), (10, 10)])
    def test_tile_outside_the_areas_is_blank_and_not_placed(
        self, ventana_2_level: OverlappingLevelTiffImage, tile_point: tuple[int, int]
    ):
        # Arrange
        overlap = ventana_2_level.overlap
        assert overlap is not None

        # Act
        tile = ventana_2_level.get_tile(tile_point)

        # Assert - the stored blank tile, left off the composed canvas
        assert md5(tile).hexdigest() == "664be9a519bfe79bd7d23742f59c7f6e"
        assert Point(*tile_point) not in overlap.placements

    def test_level_count(self, ventana_2_tiler: VentanaTiffTiler):
        # Arrange

        # Act
        levels = ventana_2_tiler.levels

        # Assert
        assert len(levels) == 8

    def test_unscanned_area_is_skipped(self, ventana_2_tiler: VentanaTiffTiler):
        # Arrange
        assert isinstance(ventana_2_tiler.metadata, VentanaMetadata)

        # Act
        areas = ventana_2_tiler.metadata.areas

        # Assert - areas 0 and 2 of the three; area 2 takes origin AOI2, so origins
        # are matched by document order, counting the unscanned area
        assert [
            (area.tiled_size, area.origin_col, area.origin_row) for area in areas
        ] == [(Size(11, 17), 0, 28), (Size(95, 61), 30, 0)]

    def test_scanner_serial_number(self, ventana_2_tiler: VentanaTiffTiler):
        # Arrange

        # Act
        serial_number = ventana_2_tiler.metadata.scanner_serial_number

        # Assert
        assert serial_number == "BI10N0306"

    @pytest.mark.parametrize(
        ["level_index", "expected_pyramid_index", "expected_composed_size"],
        [
            # the stored base level is the raw 125 x 61 grid of 1024 x 1360 tiles,
            # 128000 x 82960
            (0, 0, Size(117503, 76576)),
            (1, 1, Size(58752, 38288)),
        ],
    )
    def test_composed_size(
        self,
        ventana_2_tiler: VentanaTiffTiler,
        level_index: int,
        expected_pyramid_index: int,
        expected_composed_size: Size,
    ):
        # Arrange
        level = ventana_2_tiler.get_level(level_index)

        # Act
        overlap = level.overlap

        # Assert
        assert overlap is not None
        assert level.pyramid_index == expected_pyramid_index
        assert overlap.image_size == expected_composed_size

    def test_pixel_spacing(self, ventana_2_level: OverlappingLevelTiffImage):
        # Arrange

        # Act
        pixel_spacing = ventana_2_level.pixel_spacing

        # Assert
        assert pixel_spacing == SizeMm(0.0002325, 0.0002325)

    def test_associated_images(self, ventana_2_tiler: VentanaTiffTiler):
        # Arrange

        # Act
        overviews = ventana_2_tiler.overviews
        labels = ventana_2_tiler.labels
        thumbnails = ventana_2_tiler.thumbnails

        # Assert - like OS-1, only an Overview is exposed (no label/thumbnail series)
        assert len(overviews) == 1
        assert len(labels) == 0
        assert len(thumbnails) == 0

    def test_overview(self, ventana_2_tiler: VentanaTiffTiler):
        # Arrange
        overview = ventana_2_tiler.get_overview()

        # Act
        tile = overview.get_tile((0, 0))

        # Assert
        assert overview.image_size == Size(1008, 3008)
        assert md5(tile).hexdigest() == "4a2a7a4e9878c2768af0ce978f39acfd"
