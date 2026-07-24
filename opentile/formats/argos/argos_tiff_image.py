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

"""Image implementations for Argos avs files.

Argos base levels are a sparse JPEG-tiled pyramid, mechanically identical to Philips
(plus a per-plane focal_plane for stacked files); its Map/Macro associated images are
striped JPEG, identical to Svs. These subclasses reuse that behaviour.
"""

from typing import Optional

import numpy as np
from tifffile import TiffPage

from opentile.file import OpenTileFile
from opentile.geometry import Size, SizeMm
from opentile.jpeg import Jpeg
from opentile.tiff_image import (
    AssociatedTiffImage,
    SparseTiledLevelImage,
    StripedTiffImage,
)


class ArgosLevelTiffImage(SparseTiledLevelImage):
    """Sparse JPEG-tiled pyramid level, carrying its focal plane offset (um)."""

    def __init__(
        self,
        page: TiffPage,
        file: OpenTileFile,
        base_size: Size,
        base_mpp: SizeMm,
        jpeg: Jpeg,
        focal_plane: float = 0.0,
    ):
        super().__init__(page, file, base_size, base_mpp, jpeg)
        self._focal_plane = focal_plane

    @property
    def focal_plane(self) -> float:
        return self._focal_plane


class ArgosLabelImage(StripedTiffImage, AssociatedTiffImage):
    """Label cropped from the right side of the striped Macro image.

    Argos has no dedicated label IFD; the label sits at the high-X end of the macro
    (visible in the overview). The macro's strips are concatenated into one JPEG and the
    right-hand region is cropped losslessly, mirroring how `NdpiLabelImage` crops its
    macro (Ndpi's label is on the left instead).
    """

    def __init__(
        self,
        page: TiffPage,
        file: OpenTileFile,
        jpeg: Jpeg,
        crop_position: float,
    ):
        """
        Parameters
        ----------
        page: TiffPage
            The Macro page to crop the label from.
        file: OpenTileFile
            File to read data from.
        jpeg: Jpeg
            Jpeg instance to use.
        crop_position: float
            Left edge of the label as a fraction of the macro width; the label spans
            from here to the right edge.
        """
        super().__init__(page, file, jpeg)
        # Align the crop to a whole MCU so the lossless crop keeps the label intact.
        mcu_width = self._jpeg.get_mcu(super().get_tile((0, 0))).width
        width = self._page.imagewidth
        self._crop_from = int(width * crop_position / mcu_width) * mcu_width
        self._image_size = Size(width - self._crop_from, self._page.imagelength)

    @property
    def pixel_spacing(self) -> Optional[SizeMm]:
        return None

    def get_tile(self, tile_position: tuple[int, int]) -> bytes:
        if tile_position != (0, 0):
            raise ValueError("Non-tiled image, expected tile_position (0, 0)")
        full_frame = super().get_tile(tile_position)
        crop = (self._crop_from, 0, self.image_size.width, self.image_size.height)
        return self._jpeg.crop_multiple(full_frame, [crop])[0]

    def get_decoded_tile(self, tile_position: tuple[int, int]) -> np.ndarray:
        if tile_position != (0, 0):
            raise ValueError("Non-tiled image, expected tile_position (0, 0)")
        return self._page.asarray(squeeze=True)[:, self._crop_from :]
