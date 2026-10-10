# opentile-jpeg

Lossless JPEG operations that work directly on the compressed DCT coefficients:
crop, compose (re-tile), optimize and fill, plus coefficient access as numpy
arrays. Pixels are never decoded or re-encoded, so there is no quality loss.

Built on libjpeg-turbo, which is bundled in the wheels: no system install needed.

## Install

```sh
pip install opentile-jpeg
```

## Use

```python
import opentile_jpeg

# Crop a region (x, y, width, height).
tile = opentile_jpeg.crop(jpeg, 512, 0, 512, 512)

# Crop many regions from one source, decoded once; crops are produced lazily.
for tile in opentile_jpeg.crop_multiple(jpeg, [(0, 0, 512, 512), (512, 0, 512, 512)]):
    ...

# Crop past the right/bottom edge, filling outside the image with a color.
edge_tile = opentile_jpeg.crop(jpeg, 1536, 0, 512, 512, background=255)

# Shrink with optimal Huffman tables.
smaller = opentile_jpeg.optimize(jpeg)

# A solid-color tile with the template's size and encoding.
blank = opentile_jpeg.fill(template, (255, 255, 255))

# Re-tile sources placed on a grid at (gx, gy), e.g. overlapping scanner tiles,
# into output tiles at the given positions.
for tile in opentile_jpeg.compose(
    [(left, 0, 0), (right, 1, 0)],
    content_size=(448, 448),  # source size minus overlap
    tile_size=(512, 512),
    positions=[(0, 0), (512, 0)],
):
    ...

# Edit DCT coefficients and encode them back.
coeffs, info = opentile_jpeg.decode_coefficients(jpeg)
coeffs[0][:, :, 1:, :] = 0  # one (rows, cols, 8, 8) int16 array per component
edited = opentile_jpeg.encode_coefficients(coeffs, info)
```

Positions and sizes must follow the source's MCU grid, given by its subsampling
(see `opentile_jpeg.header`):

| Subsampling | MCU (width × height) |
|-------------|----------------------|
| 4:4:4       | 8 × 8                |
| 4:2:2       | 16 × 8               |
| 4:4:0       | 8 × 16               |
| 4:2:0       | 16 × 16              |
| 4:1:1       | 32 × 8               |

## Metadata

Outputs keep the source's metadata (ICC profile, EXIF, XMP, comments) by
default. Remove any of it with `strip`:

```python
opentile_jpeg.crop(jpeg, 0, 0, 512, 512, strip=opentile_jpeg.Metadata.EXIF)
```

EXIF is copied as is, so after a crop or compose its dimensions and thumbnail
still describe the source.

## Errors and warnings

- Invalid arguments, such as a region off the MCU grid, raise `ValueError`.
- JPEG data libjpeg-turbo cannot read raises `opentile_jpeg.JpegError`.
- Problems libjpeg-turbo recovers from, such as a truncated file, issue a
  `opentile_jpeg.JpegWarning`.

## License

opentile-jpeg is licensed under Apache-2.0. The wheels include libjpeg-turbo, under
the IJG and BSD-3-Clause licenses, whose texts are in `licenses/`.

This software is based in part on the work of the Independent JPEG Group.
