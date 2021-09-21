# *opentile*
*opentile* is a Python library for reading tiles from wsi tiff-files. Specifically, it allows tiles to be read using 2d coordinates (tile position x, y) and returns complete image data (e.g. including header for jpeg). Supported file formats are listed and descriped under *Supported file formats*.

## Important note
Please note that this is an early release and the API is not frozen yet. Function names and functionality is prone to change.

## Requirements
*opentile* uses tifffile and PyTurboJPEG.

## Limitations
Files with z-stacks are currently not fully supported.

## File formats
***Hamamatsu Ndpi***

***Phillips tiff***

***Aperio svs***

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
