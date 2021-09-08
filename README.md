# *opentile*
Python library for reading tiles from Tifffile-compatible wsi-files. Image data
in Ndpi-files are losslessly converted to tiles.

## Important note
Please note that this is an early release and the API is not frozen yet. Function names and functionality is prone to change.

## Requirements
*opentile* uses tifffile, PyTurboJPEG, and wsidicom.

## Basic usage
***Load a Ndpi-file using tile size (1024, 1024) pixels. Optionally specify series index for levels, labels, and overview if not default.***
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

## Convert Ndpi-file into Dicom using WsiDicom
***Create a base dataset that will be common for all created Dicom instances.***
```python
from wsidicom import WsiDataset
base_dataset = WsiDataset.create_test_base_dataset()
```

***Create a FileImporter using the ndpi tiler base dataset. Optional specify function for creating uids, levels to include, and if label(s) and overview(s) should be inclued, and transfer syntax.***
```python
from wsidicom import FileImporter
file_importer = NdpiFileImporter(
    ndpi_tiler,
    base_dataset
)
```

***Import the Ndpi-file as a wsidicom object with similar functionality.***
```python
wsi = WsiDicom.import_wsi(file_importer)
region = wsi.read_region((1000, 1000), 6, (200, 200))
```

***Convert the Ndpi-file into Dicom files in folder.***
```python
WsiDicom.convert(path_to_export_folder, file_importer)
```