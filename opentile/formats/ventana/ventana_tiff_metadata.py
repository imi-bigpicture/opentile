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

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Optional
from xml.etree.ElementTree import Element

from defusedxml import ElementTree
from tifffile import TiffPage

from opentile.exceptions import MissingMetadataError
from opentile.geometry import Point, PointF, Size, SizeMm
from opentile.metadata import Metadata


class VentanaElement:
    """An XMP element of Ventana metadata, for reading the attributes it must have.

    A missing or invalid attribute raises `ValueError` naming the element and the
    attribute."""

    def __init__(self, element: Element):
        self._element = element

    @property
    def tag(self) -> str:
        return self._element.tag

    def iter(self, tag: str) -> Iterator["VentanaElement"]:
        """The elements with `tag` within this element, in document order."""
        return (VentanaElement(element) for element in self._element.iter(tag))

    def children(self) -> Iterator["VentanaElement"]:
        """The direct children of this element, in document order."""
        return (VentanaElement(element) for element in self._element)

    def required_str(self, name: str) -> str:
        """The value of a required attribute."""
        value = self._element.get(name)
        if value is None:
            raise ValueError(f"Ventana metadata {self.tag} has no {name} attribute.")
        return value

    def required_int(self, name: str) -> int:
        """The value of a required integer attribute."""
        return int(self.required_str(name))

    def required_float(self, name: str) -> float:
        """The value of a required number attribute."""
        return float(self.required_str(name))

    def required_positive_int(self, name: str) -> int:
        """The value of a required attribute that must be a positive integer."""
        number = self.required_int(name)
        if number < 1:
            raise ValueError(
                f"Ventana metadata {self.tag} {name} must be positive, not {number}."
            )
        return number


@dataclass(frozen=True)
class VentanaJoint:
    """A measured offset between two neighbouring tiles of an area."""

    first: Point
    """Area cell of the left or upper tile."""
    second: Point
    """Area cell of the right or lower tile."""
    offset: PointF
    """Position of `second` minus position of `first`, in pixels."""
    confidence: float
    """The scanner's confidence in the joint, from 0 to 100."""


@dataclass(frozen=True)
class VentanaArea:
    """Stitch geometry of one scanned area (``ImageInfo``) of a Ventana slide.

    A slide has one area per scanned region of interest. Each is a tile grid, scanned
    in serpentine order, stored at its own origin in the slide's tile grid.
    """

    tiled_size: Size
    """Number of tile columns and rows of the area."""
    origin_col: int
    """Column of the area's top-left tile in the slide's tile grid."""
    origin_row: int
    """Row of the area's top-left tile in the slide's tile grid."""
    col_overlaps: list[float]
    """Mean measured overlap per column boundary, in pixels; one fewer than the
    columns."""
    row_overlaps: list[float]
    """Mean measured overlap per row boundary, in pixels; one fewer than the rows."""
    joints: list[VentanaJoint]
    """Measured offsets between neighbouring tiles of the area."""

    MIN_OVERLAP_CONFIDENCE = 95
    """Least confidence of a joint whose overlap counts towards the mean overlap of
    its boundary."""

    @classmethod
    def from_image_info(
        cls, image_info: VentanaElement, origin: PointF
    ) -> "VentanaArea":
        """Parse an area from its ``ImageInfo`` element.

        Parameters
        ----------
        image_info: VentanaElement
            The area's ``ImageInfo`` element.
        origin: PointF
            The area's origin in the slide, in pixels (from ``AoiOrigin``). The
            area's tiles are stored at whole grid positions, so it must be a multiple
            of the tile size.

        Returns
        -------
        VentanaArea
            The area's tile grid, origin, mean overlaps and joints.

        Raises
        ------
        ValueError
            If ``NumCols``, ``NumRows``, ``Width`` or ``Height`` is missing or not
            positive; if `origin` is not a multiple of the tile size; if a
            ``TileJointInfo`` has no ``FlagJoined``; or if a joined one has no
            ``Tile1``, ``Tile2``, ``Direction``, ``OverlapX``, ``OverlapY`` or
            ``Confidence``, or a ``Tile1`` or ``Tile2`` that is not positive.
        """
        tiled_size = Size(
            image_info.required_positive_int("NumCols"),
            image_info.required_positive_int("NumRows"),
        )
        tile_size = Size(
            image_info.required_positive_int("Width"),
            image_info.required_positive_int("Height"),
        )
        if origin.x % tile_size.width != 0 or origin.y % tile_size.height != 0:
            raise ValueError(
                f"Ventana metadata area origin {origin} is not on the grid of its"
                f" {tile_size} tiles."
            )
        # The joints the scanner joined, each as its TileJointInfo element and the
        # area cells of its Tile1 and Tile2 (in either spatial order).
        joined = [
            (joint_info, *cls._joint_cells(joint_info, tiled_size))
            for joint_info in image_info.iter("TileJointInfo")
            if joint_info.required_int("FlagJoined") != 0
        ]
        col_overlaps, row_overlaps = cls._mean_overlaps(joined, tiled_size)
        return cls(
            tiled_size,
            int(origin.x // tile_size.width),
            int(origin.y // tile_size.height),
            col_overlaps,
            row_overlaps,
            cls._joints(joined, tile_size),
        )

    @classmethod
    def _joint_cells(
        cls, joint_info: VentanaElement, tiled_size: Size
    ) -> tuple[Point, Point]:
        """The area cells of a joint's ``Tile1`` and ``Tile2``."""
        return (
            cls._serpentine_cell(joint_info.required_positive_int("Tile1"), tiled_size),
            cls._serpentine_cell(joint_info.required_positive_int("Tile2"), tiled_size),
        )

    @classmethod
    def _joints(
        cls, joined: list[tuple[VentanaElement, Point, Point]], tile_size: Size
    ) -> list[VentanaJoint]:
        """The joined joints between grid neighbours, with their measured offsets.

        Joints below `MIN_OVERLAP_CONFIDENCE` are kept, as registering raw tiles
        shows they are as accurate (about 1 pixel). Joints of confidence 0 are left
        out, as they carry no weight.
        """
        joints: list[VentanaJoint] = []
        for info, tile1_cell, tile2_cell in joined:
            joint = cls._neighbour_joint(info, tile1_cell, tile2_cell, tile_size)
            if joint is not None and joint.confidence > 0:
                joints.append(joint)
        return joints

    @classmethod
    def _mean_overlaps(
        cls, joined: list[tuple[VentanaElement, Point, Point]], tiled_size: Size
    ) -> tuple[list[float], list[float]]:
        """Mean measured overlap per column and per row boundary, from the joined
        joints of confidence at least `MIN_OVERLAP_CONFIDENCE`.

        Direction names the axis the overlap was measured along, not the spatial
        relation of the tiles, so LEFT is equivalent to RIGHT and DOWN to UP (newer
        Roche/Ventana scanners emit LEFT/DOWN). The boundary is taken from the lower
        of the two tile cells, which is already direction agnostic.
        """
        col_samples: list[tuple[int, float]] = []
        row_samples: list[tuple[int, float]] = []
        for info, tile1_cell, tile2_cell in joined:
            # Read before the confidence filter, so that every joined joint must
            # have a Direction.
            direction = info.required_str("Direction")
            if info.required_float("Confidence") < cls.MIN_OVERLAP_CONFIDENCE:
                continue
            if direction in ("LEFT", "RIGHT"):
                col_samples.append(
                    (
                        min(tile1_cell.x, tile2_cell.x),
                        info.required_float("OverlapX"),
                    )
                )
            elif direction in ("UP", "DOWN"):
                row_samples.append(
                    (
                        min(tile1_cell.y, tile2_cell.y),
                        info.required_float("OverlapY"),
                    )
                )
        return (
            cls._boundary_means(col_samples, tiled_size.width - 1),
            cls._boundary_means(row_samples, tiled_size.height - 1),
        )

    @staticmethod
    def _boundary_means(
        samples: list[tuple[int, float]], boundary_count: int
    ) -> list[float]:
        """Mean overlap per boundary from (boundary, overlap) samples.

        A boundary without samples takes the mean of all samples, or 0 if there are
        none. Samples for a boundary outside the grid only count towards that mean.
        """
        count = max(boundary_count, 0)
        sums = [0.0] * count
        counts = [0] * count
        for boundary, overlap in samples:
            if 0 <= boundary < count:
                sums[boundary] += overlap
                counts[boundary] += 1
        fallback = (
            sum(overlap for _, overlap in samples) / len(samples) if samples else 0.0
        )
        return [
            sums[boundary] / counts[boundary] if counts[boundary] else fallback
            for boundary in range(count)
        ]

    @staticmethod
    def _neighbour_joint(
        info: VentanaElement, tile1_cell: Point, tile2_cell: Point, tile_size: Size
    ) -> Optional[VentanaJoint]:
        """The joint between two grid neighbours, with its measured offset, or None
        if the joint's tiles are not grid neighbours.

        `info` is the joint's ``TileJointInfo`` element, and `tile1_cell` and
        `tile2_cell` are the area cells of its ``Tile1`` and ``Tile2``, in either
        spatial order.

        A joint measures both axes: the overlap along the joint (OverlapX for a
        horizontal neighbour, OverlapY for a vertical one) and the shift across it.
        Registering the raw tiles of real slides shows the shift across the joint is
        the negated value: the right tile sits at (width - OverlapX, -OverlapY) from
        the left tile, the lower tile at (-OverlapX, height - OverlapY) from the
        upper tile, whatever the Direction and tile order.
        """
        first, second = sorted(
            (tile1_cell, tile2_cell), key=lambda cell: (cell.y, cell.x)
        )
        overlap_x = info.required_float("OverlapX")
        overlap_y = info.required_float("OverlapY")
        confidence = info.required_float("Confidence")
        if first.y == second.y and second.x - first.x == 1:
            offset = PointF(tile_size.width - overlap_x, -overlap_y)
        elif first.x == second.x and second.y - first.y == 1:
            offset = PointF(-overlap_x, tile_size.height - overlap_y)
        else:
            return None
        return VentanaJoint(first, second, offset, confidence)

    @staticmethod
    def _serpentine_cell(tile_one_based: int, tiled_size: Size) -> Point:
        """Area cell of a 1-based serpentine tile index. The snake starts at the
        lower-left, so the area row is the mirror of the snake row."""
        index = tile_one_based - 1
        snake_row = index // tiled_size.width
        snake_col = index % tiled_size.width
        col = snake_col if snake_row % 2 == 0 else tiled_size.width - 1 - snake_col
        return Point(col, tiled_size.height - 1 - snake_row)


class VentanaMetadata(Metadata):
    """Metadata of a Ventana bif file, from the ``iScan`` and ``EncodeInfo`` XMP."""

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
            raise MissingMetadataError("No ScanRes in Ventana iScan metadata.")
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
        """The scanned areas (``ImageInfo``) of the slide.

        Areas flagged ``AOIScanned="0"`` are skipped, as their tiles are not stored.
        The ``ImageInfo`` elements are numbered in document order, counting skipped
        ones, and area ``n`` takes its origin from the ``AOI<n>`` child of
        ``AoiOrigin``.

        Raises
        ------
        ValueError
            If an ``ImageInfo`` has no ``AOIScanned``, if there is no ``AoiOrigin``
            block, if a scanned area has no ``AOI<n>`` origin, if an origin has no
            ``OriginX`` or ``OriginY``, if no area is scanned, or as
            `VentanaArea.from_image_info`.
        """
        root = ElementTree.fromstring(self._xmp.rstrip("\x00"))
        origins = self._aoi_origins(root)
        areas: list[VentanaArea] = []
        for index, image_info in enumerate(VentanaElement(root).iter("ImageInfo")):
            if image_info.required_int("AOIScanned") == 0:
                continue
            origin = origins.get(index)
            if origin is None:
                raise ValueError(
                    f"Ventana metadata has no AoiOrigin AOI{index} for scanned "
                    f"ImageInfo {index}."
                )
            areas.append(VentanaArea.from_image_info(image_info, origin))
        if not areas:
            raise MissingMetadataError("No scanned areas found in Ventana metadata.")
        return areas

    @staticmethod
    def _aoi_origins(root: Element) -> dict[int, PointF]:
        """Area origins in pixels, by area index, from the ``AOI<n>`` children of
        ``AoiOrigin``."""
        aoi_origin = root.find(".//AoiOrigin")
        if aoi_origin is None:
            raise ValueError("Ventana metadata has no AoiOrigin block.")
        return {
            int(child.tag[len("AOI") :]): PointF(
                child.required_float("OriginX"),
                child.required_float("OriginY"),
            )
            for child in VentanaElement(aoi_origin).children()
            if child.tag.startswith("AOI")
        }

    @staticmethod
    def _find_iscan(xmp: str) -> Element:
        """The ``iScan`` element of a Ventana XMP document: its root, or nested in an
        ``EncodeInfo`` or ``Metadata`` root."""
        root = ElementTree.fromstring(xmp.rstrip("\x00"))
        if root.tag == "iScan":
            return root
        iscan = root.find(".//iScan")
        if iscan is None:
            raise MissingMetadataError("No iScan element found in Ventana XMP.")
        return iscan
