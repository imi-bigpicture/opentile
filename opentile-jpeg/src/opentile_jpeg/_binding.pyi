from collections.abc import Sequence
from typing import Annotated

import numpy
from numpy.typing import NDArray


LIBJPEG_TURBO_VERSION: str = '3.2.0'

STRIP_ICC: int = 1

STRIP_EXIF: int = 2

STRIP_XMP: int = 4

STRIP_COMMENTS: int = 8

STRIP_OTHER: int = 16

def read_layout(jpeg: bytes) -> tuple:
    """
    ``((width, height), [((h_samp, v_samp), (height_in_blocks, width_in_blocks)), ...])`` from the frame header.
    """

def optimize(jpeg: bytes, progressive: bool = False, strip: int = 0) -> bytes:
    """See :func:`opentile_jpeg.optimize`."""

def fill(template: bytes, color: tuple[int, int, int], color_transform: bool = True, strip: int = 0) -> bytes:
    """See :func:`opentile_jpeg.fill`."""

def encode_coefficients(template: bytes, coefficients: Sequence[Annotated[NDArray[numpy.int16], dict(shape=(None, None, None, None), order='C', device='cpu', writable=False)]], optimize: bool = False, strip: int = 0) -> bytes:
    """See :func:`opentile_jpeg.encode_coefficients`."""

class Source:
    """
    A JPEG with its DCT coefficients read once, shared by crops, Croppers and Composers.
    """

    def __init__(self, jpeg: bytes) -> None: ...

    def layout(self) -> tuple:
        """Like :func:`read_layout`."""

    def crop(self, x: int, y: int, width: int, height: int, background: tuple[int, int, int] | None = None, color_transform: bool = True, strip: int = 0) -> bytes:
        """
        Losslessly crop one MCU-aligned region as a new JPEG; see :func:`opentile_jpeg.crop`.
        """

    def read_component(self, component: int) -> Annotated[NDArray[numpy.int16], dict(shape=(None, None, None, None), order='C', device='cpu')]:
        """
        The component's coefficients as a new int16 array of shape ``(height_in_blocks, width_in_blocks, 8, 8)``.
        """

class Cropper:
    """Repeated same-size crops from one Source, reusing a single compressor."""

    def __init__(self, source: Source, width: int, height: int, background: tuple[int, int, int] | None = None, color_transform: bool = True, strip: int = 0) -> None: ...

    def crop(self, x: int, y: int) -> bytes:
        """Losslessly crop the MCU-aligned region at ``(x, y)`` as a new JPEG."""

class Composer:
    """
    Re-tiles Sources placed on a grid into output tiles; see :func:`opentile_jpeg.compose`.
    """

    def __init__(self, sources: Sequence[Source], cells: Sequence[tuple[int, int]], tile_size: tuple[int, int], content_size: tuple[int, int], source_size: tuple[int, int], background: tuple[int, int, int] | None = None, color_transform: bool = True, strip: int = 0) -> None: ...

    def tile(self, x: int, y: int) -> bytes:
        """The output tile whose top-left is at plane pixel ``(x, y)``."""
