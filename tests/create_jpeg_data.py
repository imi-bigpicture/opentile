import os

from tifffile import TiffFile, TiffPage

tif_test_data_dir = os.environ.get("TIF_TESTDIR", "C:/temp/tif")
tif_test_file_name = "test.ndpi"
tif_test_file_path = tif_test_data_dir + '/' + tif_test_file_name


def open_tif() -> TiffFile:
    return TiffFile(tif_test_file_path)


def get_page(tif: TiffFile) -> TiffPage:
    return tif.series[0].levels[0].pages[0]


def create_large_scan_data(tif: TiffFile) -> bytes:
    page = get_page(tif)
    fh = tif.filehandle
    offset = page.dataoffsets[0]
    length = page.databytecounts[0]
    fh.seek(offset)
    return fh.read(length)


def save_scan_as_jpeg(jpeg_header: bytes, scan: bytes):
    f = open("scan.jpeg", "wb")
    f.write(jpeg_header)
    f.write(scan)
    f.write(bytes([0xFF, 0xD9]))  # End of Image Tag
    f.close()
