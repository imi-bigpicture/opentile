# *opentile*

*opentile* is a Python library for reading tiles from wsi tiff files. The aims of the project are:

- Allow compressed tiles to be losslessly read from WSI TIFFS using 2D coordinates (tile position x, y),
- Provide a unified interface for relevant metadata,
- Support file formats supported by `tifffile` that have a non-overlapping tile structure, as well as formats whose tiles overlap their neighbours (Trestle, Ventana), for which the de-overlapped tile placement is also exposed.

*opentile* does _not_ provide methods for reading regions from images (e.g. `get_region()`). See [openslide-python](https://github.com/openslide/openslide-python), [tiffslide](https://github.com/bayer-science-for-a-better-life/tiffslide), or [wsidicomizer](https://github.com/imi-bigpicture/wsidicomizer) for such use.

Currently implemented file formats are listed and described under [Implemented file formats](implemented-file-formats).

## Installing *opentile*

*opentile* is available on PyPI:

```console
pip install opentile
```

Alternatively, it can be installed via conda:

```console
conda install -c conda-forge opentile
```

## Important note

Please note that this is an early release and the API is not frozen yet. Function names and functionality are prone to change.

## Requirements

*opentile* requires python >=3.12 and uses numpy, Pillow, TiffFile, and PyTurboJPEG (with lib-turbojpeg >= 2.1 ), imagecodecs, defusedxml, and ome-types.

The minimum supported Python and dependency versions follow [SPEC 0](https://scientific-python.org/specs/spec-0000/).

## Limitations

Files with z-stacks are currently not fully supported for all formats.

## Implemented file formats

The following description of the workings of the implemented file formats does not include the additional specifics for each format that is handled by tifffile. Additional formats supported by tifffile are likely to be added in future releases.

*opentile* presents each level as a regular, non-overlapping grid of `tile_size` tiles. When `TiffImage.overlap` is `None` (the common case) the stored tiles already form that regular grid and are served directly. When it instead returns a `TileOverlap` — because the stored tiles overlap their neighbors (Trestle, Ventana) or use a different native tiling (JPEG XR Ndpi) — the raw stored tiles are still served by grid position, and the `TileOverlap` gives the composed level size and the placement of each stored tile, so a consumer can compose (de-overlap and/or stitch) them into the regular grid.

***Hamamatsu Ndpi***
The Ndpi-format uses non-rectangular tile size typically 8 pixels high, i.e. stripes. To form tiles, first multiple stripes are concatenated to form a frame covering the tile region. Second, if the stripes are longer than the tile width, the tile is cropped out of the frame. The concatenation and crop transformations are performed losslessly.

A ndpi-file can also contain non-tiled images. If these are part of a pyramidal series, *opentile* tiles the image.

Ndpi levels can also be compressed with JPEG XR instead of jpeg. These native tiles cannot be re-tiled losslessly, so they are served by their native grid position with a zero-overlap `TileOverlap` (see above) for a consumer to decode and stitch into the regular grid; a coarsest single-frame level is served as one tile. The macro (overview) is served as native JPEG XR, while the label, which is cropped out of the macro, is decoded and served as uncompressed pixels (there is no lossless crop for JPEG XR, and re-encoding would add loss).

The macro page in ndpi-files images the whole slide including label. A label and overview is created by cropping the macro image.

***Philips tiff***
The Philips tiff-format allows tiles to be sparse, i.e. missing. For such tiles, *opentile* instead provides a blank (currently white) tile image using the same jpeg header as the rest of the image.

***Aperio svs***
Some Aperio svs-files have corrupt tile data at edges of non-base pyramidal levels. This is observed as tiles with 0-byte length and tiles with incorrect pixel data. *opentile* detects such corruption and instead returns downscaled image data from lower levels.

***3DHistech tiff***
Only the pyramidal levels are supported (not overviews or labels).

***Huron tiff***
Huron (MACROscan) tiff-files are identified by an `Image Dimensions =` field in the description, which is Aperio-like but does not start with `Aperio `. The natively tiled levels (JPEG 2000 or jpeg) are served as-is and the pixel spacing is read from the `Resolution` field. The thumbnail, label, and macro associated images are stored uncompressed and are decoded and served as raw pixels.

***Mikroscan tiff***
Mikroscan (SL5) tiff-files use the Aperio pipe-separated description format but with a `Mikroscan Image Structure` header instead of the `Aperio ` prefix. The natively tiled jpeg levels are served as-is, and pixel spacing, magnification, scanner model/serial, and acquisition datetime are read from the description. The thumbnail, label, and macro associated images are stored uncompressed; since their series are unnamed, they are identified by the second description line (`label`, `macro`, or a `-> WxH` downscale) and are decoded and served as raw pixels.

***Motic tiff***
Motic tiff-files use the Aperio pipe-separated description format but with a `Motic <version>` header instead of the `Aperio ` prefix, and are otherwise structured like svs. The natively tiled jpeg levels are served as-is; magnification, pixel spacing, and barcode are read from the description (there is no serial number or acquisition datetime). The jpeg thumbnail and lzw label/macro associated images reuse the svs associated-image handling; since their series are unnamed, they are identified by the second description line (`label`, `macro`, or a `-> WxH` downscale).

***OME tiff***
Both tiled and strip-stored (e.g. uncompressed) levels are supported. Each level's focal plane and optical path are read from the OME metadata, so multi-plane z-stacks are exposed as separate focal planes. General (`metadata`-property) parsing is not yet implemented.

***Trestle tiff***
Trestle tiff-files (identified by a `Software` tag starting with `MedScan`) store tiles that overlap their neighbors by a fixed per-level amount. The recorded overlap is used to place each tile on the de-overlapped level (see the note on overlapping formats above).

***Ventana bif***
Ventana bif-files store a single-file pyramidal tiled BigTIFF whose tiles overlap. The per-boundary overlaps and each scanned area's origin are parsed from the `EncodeInfo` XMP (serpentine-indexed), and multi-area slides are supported. Already-stitched (non-overlapping) Ventana tiff files are also read, as a plain tiled pyramid.

***Argos avs***
Argos avs-files (identified by TIFF tag 65000 holding `Argos.Scan.Metadata` XML) store a single-file pyramidal tiled BigTIFF with sparse JPEG tiles. Missing tiles (zero offset and byte count) are served as blank (white) tiles, as for Philips. The last two directories are the thumbnail (`Map`) and overview (`Macro`) images; Argos has no dedicated label image, so the label is cropped from the right side of the overview. Stacked (z-stack) files store the focal planes on the base series' Z axis, and each plane is exposed as a level image with its `focal_plane` set from the `MinZ`/`ZRange` metadata.

## Metadata

File metadata can be accessed through the `metadata`-property of a tiler. Depending on file format and content, the following metadata is available:

- Magnification
- Scanner manufacturer
- Scanner model
- Scanner software versions
- Scanner serial number
- Acquisition datetime

## Basic usage

***Load a Ndpi-file using tile size (1024, 1024) pixels.***

```python
from opentile import OpenTile
tile_size = (1024, 1024)
tiler = OpenTile.open(path_to_ndpi_file, tile_size)
```

***Load a file using fsspec and with some file options.***

```python
from opentile import OpenTile
tiler = OpenTile.open("s3://bucket/key", file_options={"s3": "anon": True})
```

***If turbo jpeg library path is not found.***

```python
from opentile import OpenTile
tile_size = (1024, 1024)
turbo_path = 'C:/libjpeg-turbo64/bin/turbojpeg.dll'
tiler = OpenTile.open(path_to_ndpi_file, tile_size, turbo_path)
```

***Get rectangular tile at level 0 and position x=0, y=0.***

```python
level = tiler.get_evel(0)
tile = level.get_tile((0, 0))
```

***Close the tiler object.***

```python
tiler.close()
```

***Usage as context manager***

The tiler can also be used as a context manager:

```python
from opentile import OpenTile
tile_size = (1024, 1024)
with OpenTile.open(path_to_ndpi_file, tile_size) as tiler:
    level = tiler.get_evel(0)
    tile = level.get_tile((0, 0))
```

## Setup environment for development

Requires uv installed.

```console
git clone https://github.com/imi-bigpicture/opentile.git
uv sync --all-extras
```

By default the tests looks for slides in `tests/testdata`. This can be overridden by setting the `OPENTILE_TESTDIR` environment variable. The script 'tests/download_test_images.py' can be used to download publicly available [openslide testdata](https://openslide.cs.cmu.edu/download/openslide-testdata/) into the set testdata folder:

```console
python tests/download_test_images.py
```

The test data used for philips tiff is currently not publicly available as we don't have permission to share them. If you have slides in philips tiff format that can be freely shared, we would be happy to use them instead.

To watch unit tests use:

```console
uv run pytest-watch -- -m unittest
```

## Other TIFF python tools

- [tifffile](https://github.com/cgohlke/tifffile)
- [tiffslide](https://github.com/bayer-science-for-a-better-life/tiffslide)

## Contributing

We welcome any contributions to help improve this tool for the WSI community!

We recommend first creating an issue before creating potential contributions to check that the contribution is in line with the goals of the project. To submit your contribution, please issue a pull request on the imi-bigpicture/opentile repository with your changes for review.

Our aim is to provide constructive and positive code reviews for all submissions. The project relies on gradual typing and roughly follows PEP8. However, we are not dogmatic. Most important is that the code is easy to read and understand.

## Acknowledgement

*opentile*: Copyright 2021-2024 Sectra AB, licensed under Apache 2.0.

This project is part of a project that has received funding from the Innovative Medicines Initiative 2 Joint Undertaking under grant agreement No 945358. This Joint Undertaking receives support from the European Union’s Horizon 2020 research and innovation programme and EFPIA. IMI website: www.imi.europa.eu
