#    Copyright 2024 SECTRA AB
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


from pathlib import Path
from typing import Any, BinaryIO, Dict, Optional, Union

from tifffile import TiffFileError, TiffPageSeries, TiffPages, TiffFile
from upath import UPath
from fsspec.core import open

from typing import List, Sequence, Tuple

"""Wrapper around a TiffFile to provide thread safe access to the file handle."""


class OpenTileFile:

    def __init__(
        self,
        file: Union[str, Path, UPath],
        options: Optional[Dict[str, Any]] = None,
    ):
        """Open a file as TiffFIle and provide thread safe access to the file handle.

        Parameters
        ----------
        file: Union[str, Path, UPath]
            Path to file.
        options: Optional[Dict[str, Any]]
            Options to pass to open function.
        """
        opened_file: BinaryIO = open(str(file), **options or {})  # type: ignore
        try:
            self._tiff_file = TiffFile(opened_file)
        except (FileNotFoundError, TiffFileError):
            opened_file.close()
            raise
        except Exception as exception:
            opened_file.close()
            raise Exception(f"Failed to open file {file}") from exception

    @property
    def tiff(self) -> TiffFile:
        """Return the TiffFile object."""
        return self._tiff_file

    @property
    def pages(self) -> TiffPages:
        """Return the pages in the TiffFile."""
        return self._tiff_file.pages

    @property
    def series(self) -> List[TiffPageSeries]:
        """Return the series in the TiffFile."""
        return self._tiff_file.series

    @property
    def filepath(self) -> Path:
        """Return the path to the file."""
        return Path(self._tiff_file.filehandle.path)

    @property
    def lock(self):
        """Return the lock for the file handle."""
        return self._tiff_file.filehandle.lock

    def read(self, offset: int, bytecount: int) -> bytes:
        """Return bytes from single location from file handle. Is thread safe.

        Parameters
        ----------
        offset: int
            Offset in bytes.
        bytecount: int
            Length in bytes.

        Returns
        ----------
        bytes
            Requested bytes.
        """
        with self.lock:
            return self._read(offset, bytecount)

    def read_multiple(
        self, offsets_bytecounts: Sequence[Tuple[int, int]]
    ) -> List[bytes]:
        """Return bytes from multiple locations from file handle. Is thread
        safe.

        Parameters
        ----------
        offsets_bytecounts: Sequence[Tuple[int, int]]
            List of tuples with offset and lengths to read.

        Returns
        ----------
        List[bytes]
            List of requested bytes.
        """
        with self.lock:
            return [
                self._read(offset, bytecount)
                for (offset, bytecount) in offsets_bytecounts
            ]

    def _read(self, offset: int, bytecount: int):
        """Read bytes from file handle. Is not thread safe.

        Parameters
        ----------
        offset: int
            Offset in bytes.
        bytecount: int
            Length in bytes.

        Returns
        ----------
        bytes
            Requested bytes.
        """
        self._tiff_file.filehandle.seek(offset)
        return self._tiff_file.filehandle.read(bytecount)

    def close(self):
        """Close the TiffFile."""
        self._tiff_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
