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

"""Tiler for reading tiles from Ventana bif files.

Ventana stores a single-file pyramidal tiled BigTIFF whose tiles overlap. The base
level is de-overlapped by `VentanaBaseOverlap`, which solves each tile's position from
the measured joints between neighbouring tiles.

Reduced levels are downsamples of the raw overlapping mosaic and record no overlap of
their own: each stored tile packs a ``downsample x downsample`` block of downsampled
level-0 tiles, each placed at its level-0 position scaled by ``1 / downsample``.
"""

import math
import statistics
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Optional, Union

from tifffile import TiffFile, TiffPage, TiffPageSeries
from upath import UPath

from opentile.exceptions import MissingAssociatedImageError
from opentile.file import OpenTileFile
from opentile.formats.ventana.ventana_tiff_metadata import VentanaArea, VentanaMetadata
from opentile.geometry import Point, PointF, Region, Size
from opentile.metadata import Metadata
from opentile.tiff_format import TiffFormat
from opentile.tiff_image import AssociatedTiffImage, LevelTiffImage, ThumbnailTiffImage
from opentile.tiff_image_bases import (
    NativeTiledAssociatedImage,
    NativeTiledLevelImage,
    NativeTiledThumbnailImage,
    OverlappingLevelTiffImage,
    StripedAssociatedImage,
)
from opentile.tile_overlap import (
    TileOverlap,
    TilePlacement,
)
from opentile.tile_position_solver import TileOffset, TilePositionSolver
from opentile.tiler import Tiler


class VentanaBaseOverlap:
    """Factory of the base-level `TileOverlap` of a Ventana slide, which places each
    stored tile whole at its de-overlapped position.

    Within each scanned area, tile positions are solved by least squares
    (`TilePositionSolver`) from the area's joints, each giving both the overlap and
    the shift across the joint, weighted by confidence. Neighbours without a joint
    (mostly blank glass) are weakly linked at the area's mean pitch, which places
    them without pulling measured tiles off their joints. Positions are rounded to
    whole pixels, as the joints are whole-pixel measurements.

    Each area's top-left tile is placed at the area's origin, so areas keep their
    stored spacing to each other; only the tiles within an area are de-overlapped.
    The composed canvas starts at the top-left-most tile.
    """

    NOMINAL_LINK_WEIGHT = 0.01
    """Solver weight of a mean-pitch link between neighbours without a joint.

    Joints weigh confidence / 100. Mean-pitch links are the majority of links, so
    they are kept weak, or they would pull measured tiles off their joints."""

    @classmethod
    def create(cls, metadata: VentanaMetadata, tile_size: Size) -> TileOverlap:
        """Create the base-level overlap of a slide.

        Parameters
        ----------
        metadata: VentanaMetadata
            The slide's metadata, giving its scanned areas.
        tile_size: Size
            Size of the stored base-level tiles.

        Returns
        -------
        TileOverlap
            The composed size of the base level and the placement of every tile of
            the scanned areas, by stored grid position.

        Raises
        ------
        ValueError
            If two scanned areas cover the same stored grid position, or as
            `VentanaMetadata.areas`.
        """
        positions: dict[Point, Point] = {}
        for area in metadata.areas:
            area_positions = cls._area_positions(area, tile_size)
            shared = positions.keys() & area_positions.keys()
            if shared:
                first = min(shared, key=lambda grid: (grid.y, grid.x))
                raise ValueError(
                    f"Ventana metadata has scanned areas that overlap at {len(shared)}"
                    f" grid positions, first at {first}."
                )
            positions.update(area_positions)

        start_x = min(position.x for position in positions.values())
        start_y = min(position.y for position in positions.values())
        end_x = max(position.x for position in positions.values())
        end_y = max(position.y for position in positions.values())
        whole_tile = Region(Point(0, 0), tile_size)
        placements = {
            grid: [
                TilePlacement(
                    whole_tile, PointF(position.x - start_x, position.y - start_y)
                )
            ]
            for grid, position in positions.items()
        }
        image_size = Size(
            end_x - start_x + tile_size.width, end_y - start_y + tile_size.height
        )
        return TileOverlap(image_size, placements)

    @classmethod
    def _area_positions(cls, area: VentanaArea, tile_size: Size) -> dict[Point, Point]:
        """The position, in whole pixels, of every tile of `area`, by stored grid
        position."""
        cells = [
            Point(col, row)
            for row in range(area.tiled_size.height)
            for col in range(area.tiled_size.width)
        ]
        offsets = [
            TileOffset(joint.first, joint.second, joint.offset, joint.confidence / 100)
            for joint in area.joints
        ]
        offsets.extend(cls._nominal_links(area, tile_size))
        solved = TilePositionSolver(cells, offsets, Point(0, 0)).solve()
        anchor = Point(
            area.origin_col * tile_size.width, area.origin_row * tile_size.height
        )
        return {
            Point(area.origin_col + cell.x, area.origin_row + cell.y): Point(
                round(anchor.x + position.x), round(anchor.y + position.y)
            )
            for cell, position in solved.items()
        }

    @classmethod
    def _nominal_links(cls, area: VentanaArea, tile_size: Size) -> Iterator[TileOffset]:
        """Weak links at the mean pitch between the neighbours of `area` without a
        joint.

        The scan grid is slightly rotated against the stage: a neighbour is also
        shifted across the joint, consistently over the area. The links therefore
        take the area's median shift across measured joints, not zero, or they would
        pull the solution back toward an unrotated grid.
        """
        measured = {(joint.first, joint.second) for joint in area.joints}
        across_right = [
            joint.offset.y for joint in area.joints if joint.first.y == joint.second.y
        ]
        across_down = [
            joint.offset.x for joint in area.joints if joint.first.x == joint.second.x
        ]
        shift_right = statistics.median(across_right) if across_right else 0.0
        shift_down = statistics.median(across_down) if across_down else 0.0
        for row in range(area.tiled_size.height):
            for col in range(area.tiled_size.width):
                cell = Point(col, row)
                right = Point(col + 1, row)
                if col + 1 < area.tiled_size.width and (cell, right) not in measured:
                    yield TileOffset(
                        cell,
                        right,
                        PointF(tile_size.width - area.col_overlaps[col], shift_right),
                        cls.NOMINAL_LINK_WEIGHT,
                    )
                below = Point(col, row + 1)
                if row + 1 < area.tiled_size.height and (cell, below) not in measured:
                    yield TileOffset(
                        cell,
                        below,
                        PointF(shift_down, tile_size.height - area.row_overlaps[row]),
                        cls.NOMINAL_LINK_WEIGHT,
                    )


class VentanaTiffTiler(Tiler):
    def __init__(
        self,
        file: Union[str, Path, UPath, OpenTileFile],
        file_options: Optional[dict[str, Any]] = None,
    ):
        """Tiler for Ventana bif files.

        Parameters
        ----------
        file: Union[str, Path, UPath, OpenTileFile]
            Path to a Ventana bif file, or an opened Ventana OpenTileFile.
        file_options: Optional[dict[str, Any]] = None
            Options to pass to the filesystem when opening the file.
        """
        super().__init__(file, file_options)
        # An already-stitched Ventana tiff has no EncodeInfo/XMP on its level page; its
        # tiles abut and are served as a plain pyramid. A raw bif has the stitch XMP.
        self._stitched = "XMP" not in self._base_page.tags
        self._metadata = VentanaMetadata(self._metadata_page())
        self._base_mpp = self._metadata.mpp
        if self._stitched:
            self._base_composed_size = self._base_size
        else:
            self._base_overlap = self._create_base_overlap(
                Size(self._base_page.tilewidth, self._base_page.tilelength)
            )
            self._base_composed_size = self._base_overlap.image_size

    def _metadata_page(self) -> TiffPage:
        """The page carrying the iScan XMP: the level base page for a raw bif, or an
        associated page (the iScan travels on the label) for a stitched tiff."""
        if "XMP" in self._base_page.tags:
            return self._base_page
        for series in self._file.series:
            for page in series.pages:
                if isinstance(page, TiffPage) and "XMP" in page.tags:
                    return page
        return self._base_page

    @property
    def metadata(self) -> Metadata:
        return self._metadata

    @property
    def format(self) -> TiffFormat:
        return TiffFormat.VENTANA

    @classmethod
    def supported(cls, tiff_file: TiffFile) -> bool:
        return tiff_file.is_bif

    def _create_level(self, level: int, page: int = 0) -> LevelTiffImage:
        tiff_page = self._get_tiff_page(self._level_series_index, level, page)
        if self._stitched:
            return NativeTiledLevelImage(
                tiff_page, self._file, self._base_size, self._base_mpp
            )
        if level == 0:
            return OverlappingLevelTiffImage(
                tiff_page, self._file, self._base_mpp, 1.0, self._base_overlap
            )
        downsample = round(self._base_page.imagewidth / tiff_page.imagewidth)
        overlap = self._create_downsampled_overlap(
            downsample, Size(tiff_page.tilewidth, tiff_page.tilelength)
        )
        return OverlappingLevelTiffImage(
            tiff_page, self._file, self._base_mpp, float(downsample), overlap
        )

    def _create_label(self, page: int = 0) -> AssociatedTiffImage:
        if self._label_series_index is None:
            raise MissingAssociatedImageError("No label series found in this file.")
        return NativeTiledAssociatedImage(
            self._get_tiff_page(self._label_series_index, 0, page), self._file
        )

    def _create_overview(self, page: int = 0) -> AssociatedTiffImage:
        if self._overview_series_index is None:
            raise MissingAssociatedImageError("No overview series found in this file.")
        # The overview is a striped image so it is served as a StripedAssociatedImage
        # that assembles the whole image, not the tiled NativeTiledAssociatedImage
        # (one strip only).
        return StripedAssociatedImage(
            self._get_tiff_page(self._overview_series_index, 0, page),
            self._file,
        )

    def _create_thumbnail(self, page: int = 0) -> ThumbnailTiffImage:
        if self._thumbnail_series_index is None:
            raise MissingAssociatedImageError("No thumbnail series found in this file.")
        return NativeTiledThumbnailImage(
            self._get_tiff_page(self._thumbnail_series_index, 0, page),
            self._file,
            self._base_composed_size,
            self._base_mpp,
        )

    def _is_level_series(self, series: TiffPageSeries) -> bool:
        return series.name == "Baseline"

    def _is_overview_series(self, series: TiffPageSeries) -> bool:
        return series.name == "Overview"

    def _is_label_series(self, series: TiffPageSeries) -> bool:
        return series.name == "Label"

    def _is_thumbnail_series(self, series: TiffPageSeries) -> bool:
        return series.name == "Thumbnail"

    def _create_base_overlap(self, tile_size: Size) -> TileOverlap:
        """The base-level placement: each stored tile placed whole at its
        de-overlapped position (see `VentanaBaseOverlap`)."""
        return VentanaBaseOverlap.create(self._metadata, tile_size)

    def _create_downsampled_overlap(
        self, downsample: int, tile_size: Size
    ) -> TileOverlap:
        """The placement of a reduced level. Each stored tile of `tile_size` packs a
        ``downsample x downsample`` block of downsampled level-0 tiles, each placed at
        its level-0 position scaled by ``1 / downsample`` and cut from the share of
        the stored tile it occupies."""
        # A share is tile_size / downsample, which can be fractional (1360 / 32 =
        # 42.5): cut each share at its exact boundaries rounded out to whole pixels,
        # and place the cut so the share itself lands at its level-0 position.
        share_width = tile_size.width / downsample
        share_height = tile_size.height / downsample
        placements: dict[Point, list[TilePlacement]] = {}
        for level0_point, level0_placements in self._base_overlap.placements.items():
            base_position = level0_placements[0].position
            stored_tile = Point(
                level0_point.x // downsample, level0_point.y // downsample
            )
            column = level0_point.x % downsample
            row = level0_point.y % downsample
            left = math.floor(column * share_width)
            top = math.floor(row * share_height)
            right = min(math.ceil((column + 1) * share_width), tile_size.width)
            bottom = min(math.ceil((row + 1) * share_height), tile_size.height)
            source = Region(Point(left, top), Size(right - left, bottom - top))
            position = PointF(
                base_position.x / downsample - (column * share_width - left),
                base_position.y / downsample - (row * share_height - top),
            )
            placements.setdefault(stored_tile, []).append(
                TilePlacement(source, position)
            )
        # The last share ends exactly at the base size / downsample; round up so it
        # is never cut, as the base level does for its fractional extent.
        image_size = Size(
            math.ceil(self._base_overlap.image_size.width / downsample),
            math.ceil(self._base_overlap.image_size.height / downsample),
        )
        return TileOverlap(image_size, placements)
