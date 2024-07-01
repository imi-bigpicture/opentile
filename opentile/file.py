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

from tifffile import TiffPageSeries, TiffPages, TiffFile, FileHandle
from upath import UPath
from fsspec.core import open

import threading
from typing import List, Sequence, Tuple


class LockableFileHandle:
    """A lockable file handle for reading frames."""

    def __init__(self, fh: FileHandle):
        self._fh = fh
        self._lock = threading.Lock()

    def __str__(self) -> str:
        return f"{type(self).__name__} for FileHandle {self._fh}"

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._fh})"

    @property
    def filepath(self) -> Path:
        return Path(self._fh.path)

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
        with self._lock:
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
        with self._lock:
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
        self._fh.seek(offset)
        return self._fh.read(bytecount)

    def close(self) -> None:
        """Close the file handle"""
        self._fh.close()


class OpenTileFile:
    def __init__(
        self,
        file: Union[str, Path, UPath],
        options: Optional[Dict[str, Any]] = None,
    ):
        opened_file: BinaryIO = open(str(file), **options or {})  # type: ignore
        try:
            self._tiff_file = TiffFile(opened_file)
        except Exception as exception:
            opened_file.close()
            raise Exception(f"Failed to open file {file}") from exception
        self._fh = LockableFileHandle(self._tiff_file.filehandle)

    @property
    def tiff(self) -> TiffFile:
        return self._tiff_file

    @property
    def pages(self) -> TiffPages:
        return self._tiff_file.pages

    @property
    def series(self) -> List[TiffPageSeries]:
        return self._tiff_file.series

    @property
    def fh(self) -> LockableFileHandle:
        return self._fh

    def close(self):
        self._tiff_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
