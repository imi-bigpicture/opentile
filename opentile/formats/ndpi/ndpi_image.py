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

from abc import ABCMeta, abstractmethod
from functools import cached_property, lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from tifffile.tifffile import COMPRESSION, TIFF, TiffPage

from opentile.config import settings
from opentile.formats.ndpi.ndpi_tile import NdpiFrameJob, NdpiTile
from opentile.geometry import Point, Region, Size, SizeMm
from opentile.jpeg import Jpeg, JpegCropError
from opentile.tiff_image import LockableFileHandle
from opentile.tiler import TiffImage


class NdpiImage(TiffImage):
    _pyramid_index = 0

    def __init__(self, page: TiffPage, fh: LockableFileHandle, jpeg: Jpeg):
        """Ndpi image that should not be tiled (e.g. overview or label).
        Image data is assumed to be jpeg.

        Parameters
        ----------
        page: TiffPage
            TiffPage defining the page.
        fh: LockableFileHandle
            Filehandler to read data from.
        jpeg: Jpeg
            Jpeg instance to use.
        """
        super().__init__(page, fh)
        if self.compression != COMPRESSION.JPEG:
            raise NotImplementedError(
                f"{self.compression} is unsupported for ndpi "
                "(Only jpeg is supported)"
            )
        self._jpeg = jpeg
        try:
            # Defined in nm
            assert isinstance(page.ndpi_tags, dict)
            self._focal_plane = page.ndpi_tags["ZOffsetFromSlideCenter"] / 1000.0
        except (KeyError, AssertionError):
            self._focal_plane = 0.0

        self._mpp = self._get_mpp_from_page()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._page}, {self._fh}, {self._jpeg}"

    @property
    def focal_plane(self) -> float:
        """Return focal plane (in um)."""
        return self._focal_plane

    @property
    def pixel_spacing(self) -> SizeMm:
        """Return pixel spacing in mm per pixel."""
        return self.mpp / 1000.0

    @property
    def supported_compressions(self) -> Optional[List[COMPRESSION]]:
        return [COMPRESSION.JPEG]

    @property
    def mpp(self) -> SizeMm:
        """Return pixel spacing in um per pixel."""
        return self._mpp

    @cached_property
    def mcu(self) -> Size:
        """Return mcu size of image."""
        return self._jpeg.get_mcu(self._read_frame(0))

    def get_tile(self, tile_position: Tuple[int, int]) -> bytes:
        """Return tile for tile position.

        Parameters
        ----------
        tile_position: Tuple[int, int]
            Tile position to get.

        Returns
        ----------
        bytes
            Produced tile at position.
        """
        if tile_position != (0, 0):
            raise ValueError("Non-tiled image, expected tile_position (0, 0)")
        return self._read_frame(0)

    def get_decoded_tile(self, tile_position: Tuple[int, int]) -> np.ndarray:
        """Return decoded tile for tile position.

        Parameters
        ----------
        tile_position: Tuple[int, int]
            Tile position to get.

        Returns
        ----------
        bytes
            Produced tile at position.
        """
        tile = self.get_tile(tile_position)
        return self._jpeg.decode(tile)

    def _get_mpp_from_page(self) -> SizeMm:
        """Return pixel spacing in um/pixel."""
        x_resolution = self.page.tags["XResolution"].value[0]
        y_resolution = self.page.tags["YResolution"].value[0]
        resolution_unit = self.page.tags["ResolutionUnit"].value
        if resolution_unit != TIFF.RESUNIT.CENTIMETER:
            raise ValueError("Unknown resolution unit")
        # 10*1000 um per cm
        mpp_x = 10 * 1000 / x_resolution
        mpp_y = 10 * 1000 / y_resolution
        return SizeMm(mpp_x, mpp_y)


class NdpiCroppedImage(NdpiImage):
    def __init__(
        self,
        page: TiffPage,
        fh: LockableFileHandle,
        jpeg: Jpeg,
        crop: Tuple[float, float],
    ):
        """Ndpi image that should be cropped (e.g. overview or label).
        Image data is assumed to be jpeg.

        Parameters
        ----------
        page: TiffPage
            TiffPage defining the page.
        fh: LockableFileHandle
            Filehandler to read data from.
        jpeg: Jpeg
            Jpeg instance to use.
        crop: Tuple[float, float]
            Crop start and end in x-direction relative to image width.
        """
        super().__init__(page, fh, jpeg)
        crop_from = self._calculate_crop(crop[0])
        crop_to = self._calculate_crop(crop[1])

        self._image_size = Size(crop_to - crop_from, self._page.shape[0])
        self._crop_parameters = (
            crop_from,
            0,
            crop_to - crop_from,
            self.image_size.height,
        )

    def get_tile(self, tile_position: Tuple[int, int]) -> bytes:
        """Return tile for tile position.

        Parameters
        ----------
        tile_position: Tuple[int, int]
            Tile position to get.

        Returns
        ----------
        bytes
            Produced tile at position.
        """
        if tile_position != (0, 0):
            raise ValueError("Non-tiled image, expected tile_position (0, 0)")
        full_frame = super().get_tile(tile_position)
        return self._jpeg.crop_multiple(full_frame, [self._crop_parameters])[0]

    def _calculate_crop(self, crop: float) -> int:
        """Return pixel position for crop position rounded down to closest mcu
        boarder.

        Parameters
        ----------
        crop: float
            Crop parameter relative to image width.

        Returns
        ----------
        int
            Pixel position for crop.
        """
        width = self._page.shape[1]
        return int(width * crop / self.mcu.width) * self.mcu.width


class NdpiTiledImage(NdpiImage, metaclass=ABCMeta):
    def __init__(
        self,
        page: TiffPage,
        fh: LockableFileHandle,
        base_size: Size,
        tile_size: Size,
        jpeg: Jpeg,
    ):
        """Metaclass for a tiled ndpi image.

        Parameters
        ----------
        page: TiffPage
            TiffPage defining the page.
        fh: LockableFileHandle
            Filehandler to read data from.
        base_size: Size
            Size of base level in pyramid.
        tile_size: Size
            Requested tile size.
        jpeg: Jpeg
            Jpeg instance to use.
        """
        super().__init__(page, fh, jpeg)
        self._base_size = base_size
        self._tile_size = tile_size
        self._file_frame_size = self._get_file_frame_size()
        self._frame_size = Size.max(self.tile_size, self._file_frame_size)
        self._pyramid_index = self._calculate_pyramidal_index(self._base_size)
        self._headers: Dict[Size, bytes] = {}

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._page}, {self._fh}, "
            f"{self._base_size}, {self.tile_size}, {self._jpeg}"
        )

    @property
    def suggested_minimum_chunk_size(self) -> int:
        return max(self._frame_size.width // self._tile_size.width, 1)

    @property
    def tile_size(self) -> Size:
        """The size of the tiles to generate."""
        return self._tile_size

    @property
    def frame_size(self) -> Size:
        """The default read size used for reading frames."""
        return self._frame_size

    @abstractmethod
    def _read_extended_frame(self, position: Point, frame_size: Size) -> bytes:
        """Read a frame of size frame_size covering position."""
        raise NotImplementedError()

    @abstractmethod
    def _get_file_frame_size(self) -> Size:
        """Return size of single frame/stripe in file."""
        raise NotImplementedError()

    @abstractmethod
    def _get_frame_size_for_tile(self, tile_position: Point) -> Size:
        """Return frame size used for creating tile at tile position."""
        raise NotImplementedError()

    def get_tile(self, tile_position: Tuple[int, int]) -> bytes:
        """Return image bytes for tile at tile position.

        Parameters
        ----------
        tile_position: Tuple[int, int]
            Tile position to get.

        Returns
        ----------
        bytes
            Produced tile at position.
        """
        return self.get_tiles([tile_position])[0]

    def get_tiles(self, tile_positions: Sequence[Tuple[int, int]]) -> List[bytes]:
        """Return list of image bytes for tile positions.

        Parameters
        ----------
        tile_positions: Sequence[Tuple[int, int]]
            Tile positions to get.

        Returns
        ----------
        List[bytes]
            List of tile bytes.
        """
        frame_jobs = self._sort_into_frame_jobs(tile_positions)
        return [
            tile
            for frame_job in frame_jobs
            for tile in self._create_tiles(frame_job).values()
        ]

    def get_decoded_tiles(
        self, tile_positions: Sequence[Tuple[int, int]]
    ) -> List[np.ndarray]:
        """Return list of decoded tiles for tiles at tile positions.

        Parameters
        ----------
        tile_positions: Sequence[Tuple[int, int]]
            Tile positions to get.

        Returns
        ----------
        List[np.ndarray]
            List of decoded tiles.
        """
        frame_jobs = self._sort_into_frame_jobs(tile_positions)
        return [
            self._jpeg.decode(tile)
            for frame_job in frame_jobs
            for tile in self._create_tiles(frame_job).values()
        ]

    def _create_tiles(self, frame_job: NdpiFrameJob) -> Dict[Point, bytes]:
        """Return tiles defined by frame job. Read frames are cached by
        frame position.

        Parameters
        ----------
        frame_job: NdpiFrameJob
            Tile job containing tiles that should be created.

        Returns
        ----------
        Dict[Point, bytes]:
            Created tiles ordered by tile coordinate.
        """

        frame = self._read_extended_frame(frame_job.position, frame_job.frame_size)
        tiles = self._crop_to_tiles(frame_job, frame)
        return tiles

    def _crop_to_tiles(
        self, frame_job: NdpiFrameJob, frame: bytes
    ) -> Dict[Point, bytes]:
        """Crop jpeg data to tiles.

        Parameters
        ----------
        frame_job: NdpiFrameJob
            Frame job defining the tiles to produce by cropping jpeg data.
        frame: bytes
            Data to crop from.

        Returns
        ----------
        Dict[Point, bytes]:
            Created tiles ordered by tile coordinate.
        """
        try:
            tiles: List[bytes] = self._jpeg.crop_multiple(
                frame, frame_job.crop_parameters
            )
        except JpegCropError:
            raise ValueError(
                f"Failed to crop at position {frame_job.position} with "
                f"parameters {frame_job.crop_parameters}. "
                "This might be due using libjpeg-turbo < 2.1."
            )
        return {tile.position: tiles[i] for i, tile in enumerate(frame_job.tiles)}

    def _sort_into_frame_jobs(
        self, tile_positions: Sequence[Tuple[int, int]]
    ) -> List[NdpiFrameJob]:
        """Sorts tile positions into frame jobs (i.e. from the same frame.)

        Parameters
        ----------
        tile_positions: Sequence[Point]
            List of position to sort.

        Returns
        ----------
        List[NdpiFrameJob]
            List of created frame jobs.

        """
        frame_jobs: Dict[Point, NdpiFrameJob] = {}
        for tile_position in tile_positions:
            tile_point = Point.from_tuple(tile_position)
            if not self._check_if_tile_inside_image(tile_point):
                raise ValueError(
                    f"Tile {tile_point} is outside " f"tiled size {self.tiled_size}"
                )
            frame_size = self._get_frame_size_for_tile(tile_point)
            tile = NdpiTile(tile_point, self.tile_size, frame_size)
            if tile.frame_position in frame_jobs:
                frame_jobs[tile.frame_position].append(tile)
            else:
                frame_jobs[tile.frame_position] = NdpiFrameJob(tile)
        return list(frame_jobs.values())


class NdpiOneFrameImage(NdpiTiledImage):
    """Class for a ndpi image containing only one frame that should be tiled.
    The frame can be of any size (smaller or larger than the wanted tile size).
    The frame is padded to an even multiple of tile size.
    """

    def _get_file_frame_size(self) -> Size:
        """Return size of the single frame in file. For single framed image
        this is equal to the level size.

        Returns
        ----------
        Size
            The size of frame in the file.
        """
        return self.image_size

    def _get_frame_size_for_tile(self, tile_position: Point) -> Size:
        """Return read frame size for tile position. For single frame image
        the read frame size is the image size rounded up to the closest tile
        size.

        Returns
        ----------
        Size
            The read frame size.
        """
        return ((self.frame_size) // self.tile_size + 1) * self.tile_size

    @lru_cache(settings.ndpi_frame_cache)
    def _read_extended_frame(self, position: Point, frame_size: Size) -> bytes:
        """Return padded image covering tile coordinate as valid jpeg bytes.

        Parameters
        ----------
        frame_position: Point
            Upper left tile position that should be covered by the frame.
        frame_size: Size
            Size of the frame to read.

        Returns
        ----------
        bytes
            Frame
        """
        if position != Point(0, 0):
            raise ValueError("Frame position not (0, 0) for one frame level.")
        frame = self._read_frame(0)
        if (
            self.image_size.width % self.mcu.width != 0
            or self.image_size.height % self.mcu.height != 0
        ):
            # Extend to whole MCUs
            even_size = Size.ceil_div(self.image_size, self.mcu) * self.mcu
            frame = Jpeg.manipulate_header(frame, even_size)
        # Use crop_multiple as it allows extending frame
        tile = self._jpeg.crop_multiple(
            frame, [(0, 0, frame_size.width, frame_size.height)]
        )[0]
        return tile


class NdpiStripedImage(NdpiTiledImage):
    """Class for a ndpi image containing stripes. Frames are constructed by
    concatenating multiple stripes, and from the frame one or more tiles can be
    produced by lossless cropping.
    """

    def __init__(
        self,
        page: TiffPage,
        fh: LockableFileHandle,
        base_size: Size,
        tile_size: Size,
        jpeg: Jpeg,
    ):
        """Ndpi image with striped image data.

        Parameters
        ----------
        page: TiffPage
            TiffPage defining the page.
        fh: LockableFileHandle
            Filehandler to read data from.
        base_size: Size
            Size of base level in pyramid.
        tile_size: Size
            Requested tile size.
        jpeg: Jpeg
            Jpeg instance to use.
        """
        super().__init__(page, fh, base_size, tile_size, jpeg)
        self._striped_size = Size(self.page.chunked[1], self.page.chunked[0])
        jpeg_header = self.page.jpegheader
        assert isinstance(jpeg_header, bytes)
        self._jpeg_header = jpeg_header

    @property
    def stripe_size(self) -> Size:
        """Size of stripes."""
        return self._file_frame_size

    @property
    def striped_size(self) -> Size:
        """Number of stripes in columns and rows."""
        return self._striped_size

    @property
    def jpeg_header(self) -> bytes:
        """Jpeg header in image."""
        return self._jpeg_header

    def _get_file_frame_size(self) -> Size:
        """Return size of stripes in file. For striped levels this is parsed
        from the jpeg header.

        Returns
        ----------
        Size
            The size of stripes in the file.
        """
        stripe_height, stripe_width, _ = self.page.chunks
        return Size(stripe_width, stripe_height)

    def _is_partial_frame(self, tile_position: Point) -> Tuple[bool, bool]:
        """Return a tuple of bools, that are true if tile position is at the
        edge of the image in x or y.

        Parameters
        ----------
        tile_position: int
            Tile position (x or y) to check.

        Returns
        ----------
        Tuple[bool, bool]
            Tuple that is True if tile position x or y is at edge of image.
        """
        partial_x = (
            tile_position.x == (self.tiled_size.width - 1)
            and self.stripe_size.width < self.tile_size.width
        )
        partial_y = (
            tile_position.y == (self.tiled_size.height - 1)
            and self.stripe_size.height < self.tile_size.height
        )
        return partial_x, partial_y

    def _get_frame_size_for_tile(self, tile_position: Point) -> Size:
        """Return frame size used for creating tile at tile position.
        If tile is an edge tile, ensure that the frame does not extend beyond
        the image limits.

        Parameters
        ----------
        tile_position: Point
            Tile position for frame size calculation.

        Returns
        ----------
        Size
            Frame size to be used at tile position.
        """
        is_partial_x, is_partial_y = self._is_partial_frame(tile_position)
        if is_partial_x:
            width = (
                self.stripe_size.width * self.striped_size.width
                - tile_position.x * self.tile_size.width
            )
        else:
            width = self.frame_size.width

        if is_partial_y:
            height = (
                self.stripe_size.height * self.striped_size.height
                - tile_position.y * self.tile_size.height
            )
        else:
            height = self.frame_size.height
        return Size(width, height)

    @lru_cache(settings.ndpi_frame_cache)
    def _read_extended_frame(self, position: Point, frame_size: Size) -> bytes:
        """Return extended frame of frame size starting at frame position.
        Returned frame is jpeg bytes including header with correct image size.
        Original restart markers are updated to get the proper incrementation.
        End of image tag is appended.

        Parameters
        ----------
        position: Point
            Upper left tile position that should be covered by the frame.
        frame_size: Size
            Size of the frame to get.

        Returns
        ----------
        bytes
            Concatenated frame as jpeg bytes.
        """
        if frame_size in self._headers:
            header = self._headers[frame_size]
        else:
            header = self._jpeg.manipulate_header(self.jpeg_header, frame_size)
            self._headers[frame_size] = header

        stripe_region = Region(
            (position * self.tile_size) // self.stripe_size,
            Size.max(frame_size // self.stripe_size, Size(1, 1)),
        )
        indices = [
            self._get_stripe_position_to_index(stripe_coordinate)
            for stripe_coordinate in stripe_region.iterate_all()
        ]
        frame = self._jpeg.concatenate_fragments(
            (self._read_frame(index) for index in indices), header
        )
        return frame

    def _get_stripe_position_to_index(self, position: Point) -> int:
        """Return stripe index from position.

        Parameters
        ----------
        position: Point
            position of stripe to get index for.

        Returns
        ----------
        int
            Stripe index.
        """
        return position.x + position.y * self.striped_size.width
