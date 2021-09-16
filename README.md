# *opentile*
*opentile* is a Python library for reading tiles from Tifffile-compatible wsi-files. Specifically image data in Ndpi-files are losslessly converted to tiles.

## Important note
Please note that this is an early release and the API is not frozen yet. Function names and functionality is prone to change.

## Requirements
*opentile* uses tifffile and PyTurboJPEG.

## Limitations
Files with z-stacks are currently not supported.

## Basic usage
***Load a Ndpi-file using tile size (1024, 1024) pixels.***
```python
from open_tiler import NdpiTiler
tile_size = (1024, 1024)
turbo_path = 'C:/libjpeg-turbo64/bin/turbojpeg.dll'
ndpi_tiler = NdpiTiler(path_to_ndpi_file, tile_size, turbo_path)
```

***Get rectangular tile at level 0 and position x=0, y=0.***
```python
tile = ndpi_tiler.get_tile(0, (0, 0))
```

***Close the ndpi tiler object.***
```python
ndpi_tiler.close()
```
