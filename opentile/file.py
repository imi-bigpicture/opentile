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


import mmap
import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any, BinaryIO, Optional, Union, cast

from fsspec import AbstractFileSystem
from fsspec.core import url_to_fs
from fsspec.implementations.local import LocalFileSystem
from tifffile import TiffFile, TiffFileError, TiffPages, TiffPageSeries
from upath import UPath

"""Wrapper around a TiffFile to provide thread safe access to the file handle."""


class FrameReader(ABC):
    """Reads frame bytes at absolute offsets."""

    @abstractmethod
    def read(self, offset: int, bytecount: int) -> bytes:
        """Return `bytecount` bytes at `offset`."""

    def read_multiple(
        self, offsets_bytecounts: Sequence[tuple[int, int]]
    ) -> list[bytes]:
        """Return bytes for each (offset, bytecount)."""
        return [
            self.read(offset, bytecount) for offset, bytecount in offsets_bytecounts
        ]

    @abstractmethod
    def close(self) -> None:
        """Release any resources held by the reader."""


class MmapReader(FrameReader):
    """Positioned reads via slicing a read-only memory map of a local file.
    Thread safe with no lock -- slicing uses explicit offsets."""

    def __init__(self, path: str) -> None:
        self._fd = os.open(path, os.O_RDONLY)
        try:
            self._mmap = mmap.mmap(self._fd, 0, access=mmap.ACCESS_READ)
        except (ValueError, OSError):
            os.close(self._fd)
            raise

    def read(self, offset: int, bytecount: int) -> bytes:
        return self._mmap[offset : offset + bytecount]

    def close(self) -> None:
        self._mmap.close()
        os.close(self._fd)


class FsspecReader(FrameReader):
    """Positioned reads via the fsspec filesystem's ranged reads. Thread safe
    with no lock -- each read is an independent range request -- and a batch is
    fetched with `cat_ranges`, which is concurrent for remote backends."""

    def __init__(self, fs: AbstractFileSystem, path: str) -> None:
        self._fs = fs
        self._path = path

    def read(self, offset: int, bytecount: int) -> bytes:
        return cast(bytes, self._fs.cat_file(self._path, offset, offset + bytecount))

    def read_multiple(
        self, offsets_bytecounts: Sequence[tuple[int, int]]
    ) -> list[bytes]:
        starts = [offset for offset, _ in offsets_bytecounts]
        ends = [offset + bytecount for offset, bytecount in offsets_bytecounts]
        return self._fs.cat_ranges([self._path] * len(starts), starts, ends)

    def close(self) -> None:
        pass


class OpenTileFile:
    def __init__(
        self,
        file: Union[str, Path, UPath],
        options: Optional[dict[str, Any]] = None,
    ):
        """Open a file as TiffFile, parsing it once, and serve frame reads
        through a memory map (local) or fsspec ranged reads (remote).

        Parameters
        ----------
        file: Union[str, Path, UPath]
            Path to file.
        options: Optional[Dict[str, Any]]
            Storage options to pass to the filesystem.
        """
        self._file = file
        self._options = options or {}
        fs, path = cast(
            "tuple[AbstractFileSystem, str]", url_to_fs(str(file), **self._options)
        )
        self._tiff_file = self._open_tiff_file(fs, path)
        self._reader = self._create_reader(fs, path)

    def _open_tiff_file(self, fs: AbstractFileSystem, path: str) -> TiffFile:
        """Open and parse the file as a TiffFile, closing the handle on error."""
        opened_file = cast(BinaryIO, fs.open(path))
        try:
            return TiffFile(opened_file)
        except (FileNotFoundError, TiffFileError):
            opened_file.close()
            raise
        except Exception as exception:
            opened_file.close()
            raise Exception(f"Failed to open file {self._file}") from exception

    def _create_reader(self, fs: AbstractFileSystem, path: str) -> FrameReader:
        """Memory-map a local file (fast, lock-free); read a remote file through
        the fsspec filesystem's ranged reads (concurrent, lock-free)."""
        if isinstance(fs, LocalFileSystem):
            try:
                return MmapReader(path)
            except OSError:
                pass  # fall back to ranged fsspec reads
        return FsspecReader(fs, path)

    @property
    def tiff(self) -> TiffFile:
        """Return the TiffFile object."""
        return self._tiff_file

    @property
    def pages(self) -> TiffPages:
        """Return the pages in the TiffFile."""
        return self._tiff_file.pages

    @property
    def series(self) -> list[TiffPageSeries]:
        """Return the series in the TiffFile."""
        return self._tiff_file.series

    @property
    def filepath(self) -> UPath:
        """Return the path to the file."""
        return UPath(self._file, **self._options)

    def read(self, offset: int, bytecount: int) -> bytes:
        """Return bytes from single location from file. Is thread safe."""
        return self._reader.read(offset, bytecount)

    def read_multiple(
        self, offsets_bytecounts: Sequence[tuple[int, int]]
    ) -> list[bytes]:
        """Return bytes from multiple locations from file. Is thread safe."""
        return self._reader.read_multiple(offsets_bytecounts)

    def close(self):
        """Close the reader and the TiffFile."""
        self._reader.close()
        self._tiff_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
