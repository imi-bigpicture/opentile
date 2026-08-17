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

"""Metadata parser for Ventana bif files."""

from dataclasses import dataclass
from typing import Any, Optional
from xml.etree.ElementTree import Element

from defusedxml import ElementTree
from tifffile import TiffPage

from opentile.geometry import SizeMm
from opentile.metadata import Metadata


@dataclass(frozen=True)
class VentanaArea:
    """Stitch geometry of one scanned area (``ImageInfo``) of a Ventana slide.

    A slide has one area per scanned region of interest. Each is a serpentine tile
    grid placed at its own origin on the shared de-overlapped canvas.
    """

    num_cols: int
    num_rows: int
    origin_col: int
    """Grid column of the area's top-left tile on the shared canvas."""
    origin_row: int
    """Grid row of the area's top-left tile on the shared canvas."""
    col_overlaps: list[float]
    """Mean measured overlap (px) per column boundary; one fewer than ``num_cols``."""
    row_overlaps: list[float]
    """Mean measured overlap (px) per row boundary; one fewer than ``num_rows``."""


class VentanaMetadata(Metadata):
    def __init__(self, page: TiffPage):
        xmp = page.tags["XMP"].value
        if isinstance(xmp, bytes):
            xmp = xmp.decode("utf-8", "replace")
        self._xmp = xmp
        self._iscan = self._find_iscan(xmp)

    @property
    def magnification(self) -> Optional[float]:
        value = self._iscan.get("Magnification")
        return float(value) if value is not None else None

    @property
    def scan_resolution(self) -> Optional[float]:
        """Scan resolution in um/pixel of the base level."""
        value = self._iscan.get("ScanRes")
        return float(value) if value is not None else None

    @property
    def mpp(self) -> SizeMm:
        """Base level mpp (um/pixel), isotropic, from the iScan ScanRes."""
        if self.scan_resolution is None:
            raise ValueError("No ScanRes in Ventana iScan metadata.")
        return SizeMm(self.scan_resolution, self.scan_resolution)

    @property
    def scanner_manufacturer(self) -> Optional[str]:
        return "Ventana Medical Systems, Inc."

    @property
    def scanner_serial_number(self) -> Optional[str]:
        return self._iscan.get("UnitNumber")

    @property
    def scanner_software_versions(self) -> Optional[list[str]]:
        version = self._iscan.get("BuildVersion")
        return [version] if version is not None else None

    @property
    def barcode(self) -> Optional[str]:
        # iScan carries both a 1D and a 2D barcode; prefer the 1D (linear) value.
        candidates = (
            self._clean_string(self._iscan.get(attr))
            for attr in ("Barcode1D", "Barcode2D")
        )
        return next((value for value in candidates if value is not None), None)

    @property
    def properties(self) -> dict[str, Any]:
        return dict(self._iscan.attrib)

    @property
    def areas(self) -> list[VentanaArea]:
        """The scanned areas (``ImageInfo``) of the slide, each a serpentine tile grid
        with its measured per-boundary overlaps and its origin on the shared canvas.

        Areas flagged ``AOIScanned="0"`` are skipped: their tiles are not present on
        disk.
        """
        root = ElementTree.fromstring(self._xmp.rstrip("\x00"))
        origins = self._aoi_origins(root)
        areas: list[VentanaArea] = []
        for index, image_info in enumerate(root.iter("ImageInfo")):
            if int(image_info.get("AOIScanned", 1)) == 0:
                continue
            origin_x, origin_y = origins.get(index, (0.0, 0.0))
            areas.append(self._area_overlaps(image_info, origin_x, origin_y))
        if not areas:
            raise ValueError("No scanned areas found in Ventana metadata.")
        return areas

    def _area_overlaps(
        self, image_info: Element, origin_x: float, origin_y: float
    ) -> VentanaArea:
        MIN_JOINT_CONFIDENCE = 95
        cols = int(image_info.get("NumCols", 0))
        rows = int(image_info.get("NumRows", 0))
        tile_width = float(image_info.get("Width", 1))
        tile_height = float(image_info.get("Height", 1))

        # Mean measured overlap per grid boundary, plus the area-wide mean as fallback.
        col_sum = [0.0] * max(cols - 1, 0)
        col_count = [0] * max(cols - 1, 0)
        row_sum = [0.0] * max(rows - 1, 0)
        row_count = [0] * max(rows - 1, 0)
        global_col_sum = global_col_count = global_row_sum = global_row_count = 0.0
        for joint in image_info.iter("TileJointInfo"):
            if int(joint.get("FlagJoined", 0)) == 0:
                continue
            if float(joint.get("Confidence", 0)) < MIN_JOINT_CONFIDENCE:
                continue
            # Direction names the axis the overlap was measured along, not the spatial
            # relation of the tiles, so LEFT is equivalent to RIGHT and DOWN to UP
            # (newer Roche/Ventana scanners emit LEFT/DOWN). The boundary is taken from
            # the lower of the two tile indices, which is already direction agnostic.
            direction = joint.get("Direction")
            col_a, row_a = self._serpentine_cell(int(joint.get("Tile1", 1)), cols, rows)
            col_b, row_b = self._serpentine_cell(int(joint.get("Tile2", 1)), cols, rows)
            if direction in ("LEFT", "RIGHT"):
                boundary = min(col_a, col_b)
                overlap = float(joint.get("OverlapX", 0))
                if 0 <= boundary < len(col_sum):
                    col_sum[boundary] += overlap
                    col_count[boundary] += 1
                global_col_sum += overlap
                global_col_count += 1
            elif direction in ("UP", "DOWN"):
                boundary = min(row_a, row_b)
                overlap = float(joint.get("OverlapY", 0))
                if 0 <= boundary < len(row_sum):
                    row_sum[boundary] += overlap
                    row_count[boundary] += 1
                global_row_sum += overlap
                global_row_count += 1

        mean_col = global_col_sum / global_col_count if global_col_count else 0.0
        mean_row = global_row_sum / global_row_count if global_row_count else 0.0
        col_overlaps = [
            col_sum[b] / col_count[b] if col_count[b] else mean_col
            for b in range(len(col_sum))
        ]
        row_overlaps = [
            row_sum[b] / row_count[b] if row_count[b] else mean_row
            for b in range(len(row_sum))
        ]
        return VentanaArea(
            cols,
            rows,
            round(origin_x / tile_width),
            round(origin_y / tile_height),
            col_overlaps,
            row_overlaps,
        )

    @staticmethod
    def _aoi_origins(root: Element) -> dict[int, tuple[float, float]]:
        """Image-space origins keyed by area index, from the ``AoiOrigin`` block's
        ``AOI<n>`` children (empty when the block is absent)."""
        origins: dict[int, tuple[float, float]] = {}
        aoi_origin = root.find(".//AoiOrigin")
        if aoi_origin is None:
            return origins
        for child in aoi_origin:
            if not child.tag.startswith("AOI"):
                continue
            index = int(child.tag[len("AOI") :])
            origins[index] = (
                float(child.get("OriginX", 0)),
                float(child.get("OriginY", 0)),
            )
        return origins

    @staticmethod
    def _serpentine_cell(tile_one_based: int, cols: int, rows: int) -> tuple[int, int]:
        """Image cell (col, row) of a 1-based serpentine tile index. The snake starts at
        the lower-left, so the image row is the mirror of the snake row."""
        index = tile_one_based - 1
        snake_row = index // cols
        snake_col = index % cols
        col = snake_col if snake_row % 2 == 0 else cols - 1 - snake_col
        return col, rows - 1 - snake_row

    @staticmethod
    def _find_iscan(xmp: str) -> Element:
        """Return the iScan element from a Ventana XMP document (its root, or nested in
        an EncodeInfo/Metadata root)."""
        root = ElementTree.fromstring(xmp.rstrip("\x00"))
        if root.tag == "iScan":
            return root
        iscan = root.find(".//iScan")
        if iscan is None:
            raise ValueError("No iScan element found in Ventana XMP.")
        return iscan
