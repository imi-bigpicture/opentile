#    Copyright 2021-2024 SECTRA AB
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

"""Image implementations for ndpi files."""

from abc import abstractmethod
from collections.abc import Iterator, Sequence
from functools import cached_property
from typing import Optional

import numpy as np
from imagecodecs import jpeg8_decode
from tifffile import COMPRESSION, RESUNIT, TiffPage

from opentile.cache import lru_cached_method
from opentile.config import get_settings
from opentile.file import OpenTileFile
from opentile.formats.ndpi.ndpi_tile import NdpiFrameJob, NdpiTile
from opentile.geometry import Point, Region, Size, SizeMm
from opentile.jpeg import Jpeg, JpegCropError
from opentile.tiff_image import AssociatedTiffImage, BaseTiffImage, LevelTiffImage


class NdpiImage(BaseTiffImage):
    def __init__(self, page: TiffPage, file: OpenTileFile, jpeg: Jpeg):
        """Ndpi image that should not be tiled (e.g. overview or label).
        Image data is assumed to be jpeg.

        Parameters
        ----------
        page: TiffPage
            TiffPage defining the page.
        file: OpenTileFile
            File to read data from.
        jpeg: Jpeg
            Jpeg instance to use.
        """
        super().__init__(page, file)
        if self.compression != COMPRESSION.JPEG:
            raise NotImplementedError(
                f"{self.compression} is unsupported for ndpi (Only jpeg is supported)"
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
        return f"{type(self).__name__}({self._page}, {self._file}, {self._jpeg}"

    @property
    def focal_plane(self) -> float:
        return self._focal_plane

    @property
    def pixel_spacing(self) -> SizeMm:
        return self.mpp / 1000

    @property
    def supported_compressions(self) -> Optional[list[COMPRESSION]]:
        return [COMPRESSION.JPEG]

    @property
    def mpp(self) -> SizeMm:
        return self._mpp

    @cached_property
    def mcu(self) -> Size:
        """Return mcu size of image."""
        return self._jpeg.get_mcu(self._read_frame(0))

    def get_tile(self, tile_position: tuple[int, int]) -> bytes:
        if tile_position != (0, 0):
            raise ValueError("Non-tiled image, expected tile_position (0, 0)")
        return self._read_frame(0)

    def get_decoded_tile(self, tile_position: tuple[int, int]) -> np.ndarray:
        tile = self.get_tile(tile_position)
        return jpeg8_decode(tile)

    def _get_mpp_from_page(self) -> SizeMm:
        """Return pixel spacing in um/pixel."""
        x_resolution = self._page.tags["XResolution"].value[0]
        y_resolution = self._page.tags["YResolution"].value[0]
        resolution_unit = self._page.tags["ResolutionUnit"].value
        if resolution_unit != RESUNIT.CENTIMETER:
            raise ValueError("Unknown resolution unit")
        # 10*1000 um per cm
        mpp_x = 10 * 1000 / x_resolution
        mpp_y = 10 * 1000 / y_resolution
        return SizeMm(mpp_x, mpp_y)


class NdpiOverviewImage(NdpiImage, AssociatedTiffImage):
    pass


class NdpiLabelImage(NdpiImage, AssociatedTiffImage):
    def __init__(
        self,
        page: TiffPage,
        file: OpenTileFile,
        jpeg: Jpeg,
        crop: tuple[float, float],
    ):
        """Ndpi image that should be cropped (e.g. overview or label).
        Image data is assumed to be jpeg.

        Parameters
        ----------
        page: TiffPage
            TiffPage defining the page.
        file: OpenTileFile
            File to read data from.
        jpeg: Jpeg
            Jpeg instance to use.
        crop: Tuple[float, float]
            Crop start and end in x-direction relative to image width.
        """
        super().__init__(page, file, jpeg)
        crop_from = self._calculate_crop(crop[0])
        crop_to = self._calculate_crop(crop[1])

        self._image_size = Size(crop_to - crop_from, self._page.shape[0])
        self._crop_parameters = (
            crop_from,
            0,
            crop_to - crop_from,
            self.image_size.height,
        )

    def get_tile(self, tile_position: tuple[int, int]) -> bytes:
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


class NdpiTiledImage(NdpiImage, LevelTiffImage):
    def __init__(
        self,
        page: TiffPage,
        file: OpenTileFile,
        base_size: Size,
        tile_size: Size,
        jpeg: Jpeg,
    ):
        """Metaclass for a tiled ndpi image.

        Parameters
        ----------
        page: TiffPage
            TiffPage defining the page.
        file: OpenTileFile
            File to read data from.
        base_size: Size
            Size of base level in pyramid.
        tile_size: Size
            Requested tile size.
        jpeg: Jpeg
            Jpeg instance to use.
        """
        super().__init__(page, file, jpeg)
        self._base_size = base_size
        self._tile_size = tile_size
        self._file_frame_size = self._get_file_frame_size()
        self._frame_size = Size.max(self.tile_size, self._file_frame_size)
        self._scale = self._calculate_scale(self._base_size)
        self._pyramid_index = self._calculate_pyramidal_index(self._scale)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._page}, {self._file}, "
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

    @property
    def scale(self) -> float:
        return self._scale

    @property
    def pyramid_index(self) -> int:
        return self._pyramid_index

    @abstractmethod
    def _read_job_frames(
        self, frame_jobs: Sequence[NdpiFrameJob]
    ) -> Iterator[tuple[NdpiFrameJob, bytes]]:
        """Yield each (frame job, concatenated jpeg frame) for the frame jobs."""
        raise NotImplementedError()

    @abstractmethod
    def _get_file_frame_size(self) -> Size:
        """Return size of single frame/stripe in file."""
        raise NotImplementedError()

    @abstractmethod
    def _get_frame_size_for_tile(self, tile_position: Point) -> Size:
        """Return frame size used for creating tile at tile position."""
        raise NotImplementedError()

    def get_tile(self, tile_position: tuple[int, int]) -> bytes:
        return next(self.get_tiles([tile_position]))

    def get_tiles(self, tile_positions: Sequence[tuple[int, int]]) -> Iterator[bytes]:
        frame_jobs = self._sort_into_frame_jobs(tile_positions)
        return (
            tile
            for frame_job, frame in self._read_job_frames(frame_jobs)
            for tile in self._crop_to_tiles(frame_job, frame).values()
        )

    def get_decoded_tiles(
        self, tile_positions: Sequence[tuple[int, int]]
    ) -> Iterator[np.ndarray]:
        frame_jobs = self._sort_into_frame_jobs(tile_positions)
        return (
            jpeg8_decode(tile)
            for frame_job, frame in self._read_job_frames(frame_jobs)
            for tile in self._crop_to_tiles(frame_job, frame).values()
        )

    def _crop_to_tiles(
        self, frame_job: NdpiFrameJob, frame: bytes
    ) -> dict[Point, bytes]:
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
            tiles = self._jpeg.crop_multiple(frame, frame_job.crop_parameters)
        except JpegCropError:
            raise ValueError(
                f"Failed to crop at position {frame_job.position} with "
                f"parameters {frame_job.crop_parameters}. "
                "This might be due using libjpeg-turbo < 2.1."
            ) from None
        return {tile.position: tiles[i] for i, tile in enumerate(frame_job.tiles)}

    def _sort_into_frame_jobs(
        self, tile_positions: Sequence[tuple[int, int]]
    ) -> list[NdpiFrameJob]:
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
        frame_jobs: dict[Point, NdpiFrameJob] = {}
        for tile_position in tile_positions:
            tile_point = Point.from_tuple(tile_position)
            if not self._check_if_tile_inside_image(tile_point):
                raise ValueError(
                    f"Tile {tile_point} is outside tiled size {self.tiled_size}"
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

    def _read_job_frames(
        self, frame_jobs: Sequence[NdpiFrameJob]
    ) -> Iterator[tuple[NdpiFrameJob, bytes]]:
        for frame_job in frame_jobs:
            yield (
                frame_job,
                self._read_extended_frame(frame_job.position, frame_job.frame_size),
            )

    @lru_cached_method(maxsize=lambda: get_settings().ndpi_frame_cache)
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
        file: OpenTileFile,
        base_size: Size,
        tile_size: Size,
        jpeg: Jpeg,
    ):
        """Ndpi image with striped image data.

        Parameters
        ----------
        page: TiffPage
            TiffPage defining the page.
        file: OpenTileFile
            File to read data from.
        base_size: Size
            Size of base level in pyramid.
        tile_size: Size
            Requested tile size.
        jpeg: Jpeg
            Jpeg instance to use.
        """
        super().__init__(page, file, base_size, tile_size, jpeg)
        self._striped_size = Size(self._page.chunked[1], self._page.chunked[0])
        if self._page.jpegheader is None:
            raise ValueError("Missing jpeg header for striped ndpi image")
        self._jpeg_header = self._page.jpegheader

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
        stripe_height, stripe_width, _ = self._page.chunks
        return Size(stripe_width, stripe_height)

    def _is_partial_frame(self, tile_position: Point) -> tuple[bool, bool]:
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

    def _read_job_frames(
        self, frame_jobs: Sequence[NdpiFrameJob]
    ) -> Iterator[tuple[NdpiFrameJob, bytes]]:
        """Yield each (frame job, concatenated jpeg frame), reading the stripes
        for *all* frame jobs in a single call. A single frame's stripes are
        scattered through the raster-ordered file, but the union over all jobs
        forms contiguous runs, so the read can be coalesced.
        """
        job_indices = [
            (frame_job, self._stripe_indices(frame_job.position, frame_job.frame_size))
            for frame_job in frame_jobs
        ]
        frame_indices = sorted(
            {index for _, indices in job_indices for index in indices}
        )
        stripes = dict(zip(frame_indices, self._read_frames(frame_indices)))
        for frame_job, indices in job_indices:
            frame_stripes = (stripes[index] for index in indices)
            header = self._header(frame_job.frame_size)
            frame = self._jpeg.concatenate_fragments(frame_stripes, header)
            yield frame_job, frame

    def _stripe_indices(self, position: Point, frame_size: Size) -> list[int]:
        """Return the stripe indices that make up the frame at position."""
        stripe_region = Region(
            (position * self.tile_size) // self.stripe_size,
            Size.max(frame_size // self.stripe_size, Size(1, 1)),
        )
        return [
            self._get_stripe_position_to_index(stripe_coordinate)
            for stripe_coordinate in stripe_region.iterate_all()
        ]

    @lru_cached_method()
    def _header(self, frame_size: Size) -> bytes:
        """Return the jpeg header manipulated for frame_size (cached)."""
        return self._jpeg.manipulate_header(self.jpeg_header, frame_size)

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
