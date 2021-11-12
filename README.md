# *opentile*
*opentile* is a Python library for reading tiles from wsi tiff-files. Specifically, it allows tiles to be read using 2d coordinates (tile position x, y) and returns complete image data (e.g. including header for jpeg). Supported file formats are listed and descriped under *Supported file formats*.

## Important note
Please note that this is an early release and the API is not frozen yet. Function names and functionality is prone to change.

## Requirements
*opentile* requires python >=3.7 and uses numpy, Pillow, TiffFile and PyTurboJPEG (with lib-turbojpeg >= 2.1 ).

## Limitations
Files with z-stacks are currently not fully supported.
Striped pages with stripes divided in frames are not supported for other file except for Ndpi. This is common for overview and label images.

## File formats
The following description of the workings of the supported file formats does not include the additional specifics for each format that is handled by tifffile.

***Hamamatsu Ndpi***
The Ndpi-format uses non-rectangular tile size typically 8 pixels high, i.e. stripes. To form tiles, first multiple stripes are concatenated to form a frame covering the tile region. Second, if the stripes are longer than the tile width, the tile is croped out of the frame. The concatenation and crop transformations are performed losslessly.

A ndpi-file can also contain non-tiled images. If these are part of a pyramidal series, *opentile* tiles the image.

***Philips tiff***
The Philips tiff-format allows tiles to be sparse, i.e. missing. For such tiles, *opentile* instead provides a blank (currently white) tile image using the same jpeg header as the rest of the image.

***Aperio svs***
Some Asperio svs-files have corrupt tile data at edges of non-base pyramidal levels. This is observed as tiles with 0-byte length and tiles with incorrect pixel data. *opentile* detects such corruption and instead returns downscaled image data from lower levels.

## Basic usage
***Load a Ndpi-file using tile size (1024, 1024) pixels.***
```python
from opentile import OpenTile
tile_size = (1024, 1024)
turbo_path = 'C:/libjpeg-turbo64/bin/turbojpeg.dll'
tiler = OpenTile.open(path_to_ndpi_file, tile_size, turbo_path)
```

***Get rectangular tile at level 0 and position x=0, y=0.***
```python
tile = tiler.get_tile(0, (0, 0))
```

***Close the tiler object.***
```python
tiler.close()
```

## Acknowledgement
This project is part of a project that has received funding from the Innovative Medicines Initiative 2 Joint Undertaking under grant agreement No 945358. This Joint Undertaking receives support from the European Union’s Horizon 2020 research and innovation programme and EFPIA. IMI website: www.imi.europa.eu