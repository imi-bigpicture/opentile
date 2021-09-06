# *ndpi-tiler*
Python library for producing rectangular tiles from Ndpi-files, to be used
together with WsiDicom to convert Ndpi-files to Dicom.

## Important note
Please note that this is an early release and the API is not frozen yet. Function names and functionality is prone to change.

## Requirements
*ndpi-tiler* uses tifffile, PyTurboJPEG, wsidicom, and pydicom.

## Limitations

## Basic usage
***Load a Ndpi-file using tile size (1024, 1024) pixels.***
```python
from ndpi_tiler import NdpiTiler
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
***Specify series and levels to extract from Ndpi-file, and define instance uids.***
```python
import pydicom
uids = [pydicom.uid.generate_uid() for i in range(10)]
include_series = {
    'VOLUME': (0, {
        0: uids[0],
        1: uids[1],
        2: uids[2],
        3: uids[3],
        4: uids[4],
        5: uids[5],
        6: uids[6],
    }),
    'LABEL': (2, {0: uids[7]}),
    'OVERVIEW': (3, {0: uids[8]})
```

***Create a base dataset that will be common for all created Dicom instances.***
```python
from wsidicom import WsiDataset
base_dataset = WsiDataset.create_test_base_dataset()
```

***Create a NdpiFileImporter, specifying input file path, base dataset, included series, transfer syntax, tile size, and path to turbojpeg library.***
```python
from ndpi_tiler import NdpiFileImporter
file_importer = NdpiFileImporter(
    path_to_ndpi_file,
    base_dataset,
    include_series,
    (1024, 1024),
    'C:/libjpeg-turbo64/bin/turbojpeg.dll'
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