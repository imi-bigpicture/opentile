# *opentile*

*opentile* is a Python library for reading tiles from wsi tiff files. The aims of the proect are:

- Allow compressed tiles to be losslessly read from wsi tiffs using 2D coordinates (tile position x, y).
- Provide unified interface for relevant metadata.
- Support all file formats supported by tifffile that has a non-overlapping tile structure.

Crrently supported file formats are listed and described under *Supported file formats*.

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

Please note that this is an early release and the API is not frozen yet. Function names and functionality is prone to change.

## Requirements

*opentile* requires python >=3.8 and uses numpy, Pillow, TiffFile, and PyTurboJPEG (with lib-turbojpeg >= 2.1 ), imagecodecs, defusedxml, and ome-types.

## Limitations

Files with z-stacks are currently not fully supported for all formats.

## Supported file formats

The following description of the workings of the supported file formats does not include the additional specifics for each format that is handled by tifffile. Additional formats supported by tifffile and that have non-overlapping tile layout are likely to be added in future release.

***Hamamatsu Ndpi***
The Ndpi-format uses non-rectangular tile size typically 8 pixels high, i.e. stripes. To form tiles, first multiple stripes are concatenated to form a frame covering the tile region. Second, if the stripes are longer than the tile width, the tile is croped out of the frame. The concatenation and crop transformations are performed losslessly.

A ndpi-file can also contain non-tiled images. If these are part of a pyramidal series, *opentile* tiles the image.

The macro page in ndpi-files images the whole slide including label. A label and overview is created by cropping the macro image.

***Philips tiff***
The Philips tiff-format allows tiles to be sparse, i.e. missing. For such tiles, *opentile* instead provides a blank (currently white) tile image using the same jpeg header as the rest of the image.

***Aperio svs***
Some Asperio svs-files have corrupt tile data at edges of non-base pyramidal levels. This is observed as tiles with 0-byte length and tiles with incorrect pixel data. *opentile* detects such corruption and instead returns downscaled image data from lower levels. Associated images (label, overview) are currently not handled correctly.

***3DHistech tiff***
Only the pyramidal levels are supported (not overviews or labels).

## Metadata

File metadata can be accessed through the `metadata`-property of a tiler. Depending on file format and content, the following metadata is avaiable:

- Magnification
- Scanner manufacturer
- Scanner model
- Scanner software versions
- Scanner serial number
- Aquisition datetime

## Basic usage

***Load a Ndpi-file using tile size (1024, 1024) pixels.***

```python
from opentile import OpenTile
tile_size = (1024, 1024)
tiler = OpenTile.open(path_to_ndpi_file, tile_size)
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

The tiler can also be used as context manager:

```python
from opentile import OpenTile
tile_size = (1024, 1024)
with OpenTile.open(path_to_ndpi_file, tile_size) as tiler:
    level = tiler.get_evel(0)
    tile = level.get_tile((0, 0))
```

## Setup environment for development

Requires poetry and pytest and pytest-watch installed in the virtual environment.

```console
git clone https://github.com/imi-bigpicture/opentile.git
poetry install
```

By default the tests looks for slides in 'tests/testdata'. This can be overriden by setting the OPENTILE_TESTDIR environment variable. The script 'tests/download_test_images.py' can be used to download publically available [openslide testdata](https://openslide.cs.cmu.edu/download/openslide-testdata/) into the set testdata folder:

```console
python tests/download_test_images.py
```

The test data used for philips tiff is currently not publically available as we dont have permission to share them. If you have slides in philips tiff format that can be freely shared we would be happy to use them instead.

To watch unit tests use:

```console
poetry run pytest-watch -- -m unittest
```

## Other TIFF python tools

- [tifffile](https://github.com/cgohlke/tifffile)
- [tiffslide](https://github.com/bayer-science-for-a-better-life/tiffslide)

## Contributing

We welcome any contributions to help improve this tool for the WSI community!

We recommend first creating an issue before creating potential contributions to check that the contribution is in line with the goals of the project. To submit your contribution, please issue a pull request on the imi-bigpicture/opentile repository with your changes for review.

Our aim is to provide constructive and positive code reviews for all submissions. The project relies on gradual typing and roughly follows PEP8. However, we are not dogmatic. Most important is that the code is easy to read and understand.

## Acknowledgement

*opentile*: Copyright 2021-2023 Sectra AB, licensed under Apache 2.0.

This project is part of a project that has received funding from the Innovative Medicines Initiative 2 Joint Undertaking under grant agreement No 945358. This Joint Undertaking receives support from the European Union’s Horizon 2020 research and innovation programme and EFPIA. IMI website: www.imi.europa.eu
