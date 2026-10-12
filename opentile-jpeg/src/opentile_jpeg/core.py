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

"""Implementation of the opentile_jpeg public API (see the package docstring)."""

from collections.abc import Iterable, Iterator, Sequence

import numpy as np

import opentile_jpeg._binding as lib
from opentile_jpeg.models import CoefficientInfo, Color, JpegHeader, Metadata

DEFAULT_MAX_PIXELS = 178_956_970
"""The default ``max_pixels``: larger images are refused before they are read.

Twice Pillow's ``Image.MAX_IMAGE_PIXELS``, the size at which Pillow refuses an
image as a likely decompression bomb. Reading an image's coefficients takes 3
bytes per pixel for 4:2:0 and 6 for 4:4:4, so about 0.5 to 1 GB at this size.
"""


def _pixel_limit(max_pixels: int | None) -> int:
    """``max_pixels`` for the extension, where 0 means no limit."""
    if max_pixels is None:
        return 0
    if max_pixels <= 0:
        raise ValueError(f"max_pixels must be positive or None, got {max_pixels}")
    return max_pixels


def optimize(
    jpeg: bytes,
    progressive: bool = False,
    strip: Metadata = Metadata.NONE,
    *,
    max_pixels: int | None = DEFAULT_MAX_PIXELS,
) -> bytes:
    """Losslessly shrink a JPEG by rebuilding it with optimal Huffman tables.

    Pixels are identical (the DCT coefficients are untouched); only the entropy
    coding is recomputed.

    Parameters
    ----------
    jpeg: bytes
        The source JPEG.
    progressive: bool = False
        If True, write a progressive JPEG; otherwise a sequential one, even when
        the source is progressive.
    strip: Metadata = Metadata.NONE
        Metadata to leave out of the output; by default all of it is kept.
    max_pixels: int | None = DEFAULT_MAX_PIXELS
        Refuse a source of more pixels than this before allocating for it; None
        for no limit.

    Returns
    ----------
    bytes
        The optimized JPEG.

    Raises
    ----------
    JpegTooLargeError
        If the source has more than ``max_pixels`` pixels.
    """
    return lib.optimize(jpeg, progressive, strip.value, _pixel_limit(max_pixels))


def header(jpeg: bytes) -> JpegHeader:
    """Read a JPEG's dimensions and luma sampling factors without decoding pixels.

    The sampling factors give the MCU grid used for alignment (see the module
    docstring): the MCU is ``8 * h_samp_factor`` by ``8 * v_samp_factor`` pixels.

    Parameters
    ----------
    jpeg: bytes
        The source JPEG.

    Returns
    ----------
    JpegHeader
        The image ``width``, ``height`` and luma ``sampling`` factors.
    """
    (width, height), components = lib.read_layout(jpeg)
    luma_sampling, _ = components[0]
    return JpegHeader(width, height, luma_sampling)


def _rgb(color: Color, name: str) -> tuple[int, int, int]:
    """The color as RGB; any sequence of three values is accepted at runtime."""
    if isinstance(color, (int, np.integer)):
        return int(color), int(color), int(color)
    if len(color) != 3:
        raise ValueError(
            f"{name} must be a gray level or an (r, g, b) triple, got {color!r}"
        )
    r, g, b = color
    return int(r), int(g), int(b)


def _optional_rgb(color: Color | None, name: str) -> tuple[int, int, int] | None:
    return None if color is None else _rgb(color, name)


def crop(
    jpeg: bytes,
    x: int,
    y: int,
    width: int,
    height: int,
    background: Color | None = None,
    color_transform: bool = True,
    strip: Metadata = Metadata.NONE,
    *,
    max_pixels: int | None = DEFAULT_MAX_PIXELS,
) -> bytes:
    """Losslessly crop one region ``(x, y, width, height)`` from a JPEG.

    The region is copied in the DCT domain (no re-encoding); the coordinates must
    be MCU-aligned (see the module docstring).

    With a ``background`` color the region may run past the image's right and
    bottom edges, growing the output: every block wholly outside the image is a
    flat block of that color, while blocks straddling the edge are copied as they
    are. ``color_transform`` applies to the background as for :func:`fill`.

    Parameters
    ----------
    jpeg: bytes
        The source JPEG.
    x: int
        Left edge of the region in pixels, from the source's left.
    y: int
        Top edge of the region in pixels, from the source's top.
    width: int
        Region width in pixels.
    height: int
        Region height in pixels.
    background: int | tuple[int, int, int] | None = None
        Color filling the region outside the image, as for :func:`fill`. When
        None, the region must lie inside the image.
    color_transform: bool = True
        How ``background`` reaches the components, as for :func:`fill`.
    strip: Metadata = Metadata.NONE
        Metadata to leave out of the output; by default all of it is kept.
    max_pixels: int | None = DEFAULT_MAX_PIXELS
        Refuse a source of more pixels than this before allocating for it; None
        for no limit.

    Returns
    ----------
    bytes
        The cropped region as a JPEG.

    Raises
    ----------
    ValueError
        If the region is not MCU-aligned, starts at a negative position, leaves
        the image without a ``background``, or ``background`` is outside 0..255.
    JpegTooLargeError
        If the source has more than ``max_pixels`` pixels.
    """
    return lib.Source(jpeg, _pixel_limit(max_pixels)).crop(
        x,
        y,
        width,
        height,
        _optional_rgb(background, "background"),
        color_transform,
        strip.value,
    )


def fill(
    template: bytes,
    color: Color,
    color_transform: bool = True,
    strip: Metadata = Metadata.NONE,
    *,
    max_pixels: int | None = DEFAULT_MAX_PIXELS,
) -> bytes:
    """A solid-color JPEG reusing ``template``'s tables, dimensions and subsampling.

    ``color`` is either a single 0..255 gray level or an ``(r, g, b)`` triple of
    0..255 values. Sharing the template's codestream layout makes it a cheap
    stand-in for blank/background tiles.

    ``color_transform`` controls how the color reaches the components of a
    multi-component frame: ``True`` (default) converts RGB to the frame's YCbCr
    (for the usual YCbCr JPEGs); ``False`` writes the channels straight to the
    components (for an RGB-colorspace frame). It has no effect on a single-
    component (grayscale) frame, which is always filled with the color's luma.

    The output keeps the template's metadata except what ``strip`` names.

    A template of more than ``max_pixels`` pixels is refused before anything is
    allocated for it; pass None for no limit.

    Raises
    ----------
    ValueError
        If a color component is outside 0..255, or ``color`` is not one or
        three values.
    JpegTooLargeError
        If the template has more than ``max_pixels`` pixels.
    """
    return lib.fill(
        template,
        _rgb(color, "color"),
        color_transform,
        strip.value,
        _pixel_limit(max_pixels),
    )


def crop_multiple(
    jpeg: bytes,
    regions: Sequence[tuple[int, int, int, int]],
    background: Color | None = None,
    color_transform: bool = True,
    strip: Metadata = Metadata.NONE,
    *,
    max_pixels: int | None = DEFAULT_MAX_PIXELS,
) -> Iterator[bytes]:
    """Losslessly crop many ``(x, y, width, height)`` regions, decoding once.

    Each crop is yielded as it is produced (one resident at a time); same-size
    regions use a faster compressor-reusing path.

    Parameters
    ----------
    jpeg: bytes
        The source JPEG.
    regions: Sequence[tuple[int, int, int, int]]
        The regions to crop as ``(x, y, width, height)``, each MCU-aligned (see
        the module docstring).
    background: int | tuple[int, int, int] | None = None
        Color filling regions past the image's right and bottom edges, as for
        :func:`crop`. When None, every region must lie inside the image.
    color_transform: bool = True
        How ``background`` reaches the components, as for :func:`fill`.
    strip: Metadata = Metadata.NONE
        Metadata to leave out of the output; by default all of it is kept.
    max_pixels: int | None = DEFAULT_MAX_PIXELS
        Refuse a source of more pixels than this before allocating for it; None
        for no limit.

    Returns
    ----------
    Iterator[bytes]
        One cropped JPEG per region, in the same order as ``regions``, produced
        lazily. Wrap in ``list()`` for all at once.

    Raises
    ----------
    ValueError
        If a region is not MCU-aligned, starts at a negative position, or leaves
        the image without a ``background``; raised when that region is reached.
        Also if ``background`` is outside 0..255.
    JpegTooLargeError
        If the source has more than ``max_pixels`` pixels.
    """
    if not regions:
        return iter(())
    rgb = _optional_rgb(background, "background")
    source = lib.Source(jpeg, _pixel_limit(max_pixels))
    sizes = {(width, height) for _, _, width, height in regions}
    if len(sizes) == 1:
        cropper = lib.Cropper(source, *sizes.pop(), rgb, color_transform, strip.value)
        return (cropper.crop(x, y) for x, y, _, _ in regions)
    return (
        source.crop(*region, rgb, color_transform, strip.value) for region in regions
    )


def compose(
    sources: Iterable[tuple[bytes, int, int]],
    content_size: tuple[int, int],
    tile_size: tuple[int, int],
    positions: Sequence[tuple[int, int]],
    source_size: tuple[int, int] | None = None,
    background: Color | None = None,
    color_transform: bool = True,
    strip: Metadata = Metadata.NONE,
    *,
    max_pixels: int | None = DEFAULT_MAX_PIXELS,
) -> Iterator[bytes]:
    """Re-tile several source JPEGs into a uniform grid of output tiles, losslessly.

    The sources are laid out on a virtual plane: a source at grid cell
    ``(gx, gy)`` occupies the plane rectangle starting at
    ``(gx * content_width, gy * content_height)``. Each requested output tile is a
    ``tile_size`` window of that plane, filled by copying DCT blocks verbatim from
    whichever source covers each position (no re-encoding). Every source is
    decoded once and reused across all positions, so output tiles that share a
    source tile do not re-decode it.

    All sizes and positions must be MCU-aligned (see the module docstring).

    Parameters
    ----------
    sources: Iterable[tuple[bytes, int, int]]
        The source tiles as ``(jpeg, gx, gy)``: the JPEG bytes and the source's
        integer grid cell on the plane. Must share tables and subsampling.
    content_size: tuple[int, int]
        ``(width, height)`` of the unique, kept region of each source tile — i.e.
        the stride at which sources are placed on the plane. Where sources
        overlap, this is the source tile size minus the overlap, so the duplicated
        margin is stepped over and dropped; e.g. 512x512 sources overlapping by
        64px use ``content_size=(448, 448)``. With no overlap it equals the source
        tile size and the result is a plain grid stitch.
    tile_size: tuple[int, int]
        ``(width, height)`` in pixels of every output tile.
    positions: Sequence[tuple[int, int]]
        Top-left ``(plane_x, plane_y)`` pixel of each output tile on the plane;
        one output tile is produced per position, in the same order.
    source_size: tuple[int, int] | None = None
        ``(width, height)`` of the full source tile. When sources overlap this is
        larger than ``content_size``, and the trailing source in each row/column
        keeps its tail (the extra ``source - content`` region no later source
        overwrites); interior tails are still dropped. Defaults to
        ``content_size`` (no tail kept — a plain grid stitch).
    background: int | tuple[int, int, int] | None = None
        Color, as for :func:`fill`, of everything no source covers, including
        the padding past each source's image edges as in :func:`crop`. When
        None, uncovered areas are neutral gray.
    color_transform: bool = True
        How ``background`` reaches the components, as for :func:`fill`.
    strip: Metadata = Metadata.NONE
        Metadata to leave out of the output, which carries the first source's;
        by default all of it is kept.
    max_pixels: int | None = DEFAULT_MAX_PIXELS
        Refuse a source of more pixels than this before allocating for it; None
        for no limit.

    Returns
    ----------
    Iterator[bytes]
        One output JPEG per position, in order, sharing the sources' tables.
        Tiles are produced lazily (one in memory at a time); wrap in ``list()``
        for all at once.

    Raises
    ----------
    ValueError
        If there are no sources, their sampling differs, a size is not
        MCU-aligned, or ``background`` is outside 0..255 (raised by this call);
        or if a position is not MCU-aligned (raised when it is reached).
    JpegTooLargeError
        If a source has more than ``max_pixels`` pixels.
    """
    limit = _pixel_limit(max_pixels)
    handles, cells = [], []
    for jpeg, gx, gy in sources:  # a loop keeps warnings pointing at the caller
        handles.append(lib.Source(jpeg, limit))
        cells.append((gx, gy))
    composer = lib.Composer(
        handles,
        cells,
        tile_size,
        content_size,
        source_size or content_size,
        _optional_rgb(background, "background"),
        color_transform,
        strip.value,
    )
    return (composer.tile(plane_x, plane_y) for plane_x, plane_y in positions)


def decode_coefficients(
    jpeg: bytes, *, max_pixels: int | None = DEFAULT_MAX_PIXELS
) -> tuple[list[np.ndarray], CoefficientInfo]:
    """Read a JPEG's quantized DCT coefficients, without decoding to pixels.

    Parameters
    ----------
    jpeg: bytes
        The source JPEG.
    max_pixels: int | None = DEFAULT_MAX_PIXELS
        Refuse a source of more pixels than this before allocating for it; None
        for no limit.

    Returns
    ----------
    tuple[list[numpy.ndarray], CoefficientInfo]
        One ``int16`` array per component, shape
        ``(height_in_blocks, width_in_blocks, 8, 8)`` in natural (row-major 8x8)
        DCT order, plus the :class:`CoefficientInfo` needed to re-encode. Modify
        the arrays in place and pass them to :func:`encode_coefficients`.

    Raises
    ----------
    JpegTooLargeError
        If the source has more than ``max_pixels`` pixels.
    """
    source = lib.Source(jpeg, _pixel_limit(max_pixels))
    size, components = source.layout()
    coeffs = [source.read_component(index) for index in range(len(components))]
    sampling = tuple(factors for factors, _ in components)
    block_shapes = tuple(shape for _, shape in components)
    return coeffs, CoefficientInfo(size, sampling, block_shapes, bytes(jpeg))


def encode_coefficients(
    coeffs: Sequence[np.ndarray],
    info: CoefficientInfo,
    optimize: bool = False,
    strip: Metadata = Metadata.NONE,
) -> bytes:
    """Encode a JPEG from (possibly modified) DCT coefficients.

    The tables, sampling and dimensions are taken from ``info.template`` so only
    the coefficient values you changed take effect; the arrays must keep the
    per-component block shapes from the decode.

    Parameters
    ----------
    coeffs: Sequence[numpy.ndarray]
        One coefficient array per component, as from :func:`decode_coefficients`.
    info: CoefficientInfo
        The info returned alongside those coefficients.
    optimize: bool = False
        If True, rebuild optimal Huffman tables (smaller file, same pixels).
    strip: Metadata = Metadata.NONE
        Metadata of ``info.template`` to leave out; by default all of it is kept.

    Returns
    ----------
    bytes
        The encoded JPEG.

    Raises
    ----------
    ValueError
        If ``coeffs`` does not match the template's per-component block shapes.
    """
    arrays = [np.ascontiguousarray(array, dtype=np.int16) for array in coeffs]
    return lib.encode_coefficients(info.template, arrays, optimize, strip.value)
