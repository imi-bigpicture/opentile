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

"""Wrapper around a TiffFile to provide thread safe access to the file handle."""

import os
import queue
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import Any, BinaryIO, Optional, Union, cast

from fsspec import AbstractFileSystem
from fsspec.core import url_to_fs
from fsspec.implementations.local import LocalFileSystem
from tifffile import TiffFile, TiffFileError, TiffPages, TiffPageSeries
from upath import UPath

# Open local files for sequential read-ahead via FILE_FLAG_SEQUENTIAL_SCAN on
# Windows (O_SEQUENTIAL); other platforms rely on the kernel's default readahead.
_READ_FLAGS = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_SEQUENTIAL", 0)


class FrameReader(ABC):
    """Reads frame bytes at absolute offsets."""

    @abstractmethod
    def read(self, offset: int, bytecount: int) -> bytes:
        """Return `bytecount` bytes at `offset`."""

    @abstractmethod
    def read_multiple(
        self, offsets_bytecounts: Sequence[tuple[int, int]]
    ) -> list[bytes]:
        """Return bytes for each (offset, bytecount)."""

    @abstractmethod
    def close(self) -> None:
        """Release any resources held by the reader."""


class FileHandlePool:
    """A pool of reusable raw file handles for a single path. `acquire()` hands
    out a free handle, opening a new one only when none is free, so the number of
    open handles tracks peak concurrency rather than total callers."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._available: queue.SimpleQueue[BinaryIO] = queue.SimpleQueue()
        self._handles: list[BinaryIO] = []

    @contextmanager
    def acquire(self) -> Iterator[BinaryIO]:
        """Yield a handle for the duration of the block, returning it to the pool
        on exit."""
        handle = self._get_handle()
        try:
            yield handle
        finally:
            self._available.put(handle)

    def close(self) -> None:
        for handle in self._handles:
            handle.close()

    def _open(self) -> BinaryIO:
        fd = os.open(self._path, _READ_FLAGS)
        return cast(BinaryIO, os.fdopen(fd, "rb", buffering=0))

    def _get_handle(self) -> BinaryIO:
        """Get a handle from the pool, opening a new one if necessary."""
        try:
            return self._available.get_nowait()
        except queue.Empty:
            return self._new_handle()

    def _new_handle(self) -> BinaryIO:
        """Create a new handle and add it to the pool."""
        handle = self._open()
        self._handles.append(handle)
        return handle


class LocalFileReader(FrameReader):
    """Reads a local file through a pool of raw handles with OS read-ahead."""

    def __init__(self, path: str) -> None:
        self._pool = FileHandlePool(path)

    @staticmethod
    def _read_at(handle: BinaryIO, offset: int, bytecount: int) -> bytes:
        handle.seek(offset)
        return handle.read(bytecount)

    def read(self, offset: int, bytecount: int) -> bytes:
        with self._pool.acquire() as handle:
            return self._read_at(handle, offset, bytecount)

    def read_multiple(
        self, offsets_bytecounts: Sequence[tuple[int, int]]
    ) -> list[bytes]:
        """Coalesce contiguous reads into a single read and slice them apart."""
        if len(offsets_bytecounts) == 1:
            offset, bytecount = offsets_bytecounts[0]
            return [self.read(offset, bytecount)]
        with self._pool.acquire() as handle:
            result = [b""] * len(offsets_bytecounts)
            for run in self._contiguous_runs(offsets_bytecounts):
                start = offsets_bytecounts[run[0]][0]
                last_offset, last_bytecount = offsets_bytecounts[run[-1]]
                blob = self._read_at(
                    handle, start, last_offset + last_bytecount - start
                )
                if len(run) == 1:
                    # The blob is exactly the frame; return it without a copy.
                    result[run[0]] = blob
                    continue
                for index in run:
                    offset, bytecount = offsets_bytecounts[index]
                    result[index] = blob[offset - start : offset - start + bytecount]
            return result

    @staticmethod
    def _contiguous_runs(
        offsets_bytecounts: Sequence[tuple[int, int]],
    ) -> list[list[int]]:
        """Group contiguous frames into runs of original indices, in offset order."""
        in_offset_order = sorted(
            range(len(offsets_bytecounts)),
            key=lambda index: offsets_bytecounts[index][0],
        )
        runs: list[list[int]] = []
        previous_end = -1
        for index in in_offset_order:
            offset, bytecount = offsets_bytecounts[index]
            if runs and offset == previous_end:
                runs[-1].append(index)
            else:
                runs.append([index])
            previous_end = offset + bytecount
        return runs

    def close(self) -> None:
        self._pool.close()


class FsspecReader(FrameReader):
    """Reads a remote file via the filesystem's ranged reads."""

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
        through a frame reader.

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
        self._frame_reader = self._create_frame_reader(fs, path)

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

    def _create_frame_reader(self, fs: AbstractFileSystem, path: str) -> FrameReader:
        """Create the FrameReader for the file: a `LocalFileReader` for a plain
        local file, or an `FsspecReader` for a remote file (or a local file
        opened with storage options)."""
        if isinstance(fs, LocalFileSystem) and not self._options:
            return LocalFileReader(path)
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
        return list(self._tiff_file.series)

    @property
    def filepath(self) -> UPath:
        """Return the path to the file."""
        return UPath(self._file, **self._options)

    def read(self, offset: int, bytecount: int) -> bytes:
        """Return bytes from single location from file. Is thread safe."""
        return self._frame_reader.read(offset, bytecount)

    def read_multiple(
        self, offsets_bytecounts: Sequence[tuple[int, int]]
    ) -> list[bytes]:
        """Return bytes from multiple locations from file. Is thread safe."""
        return self._frame_reader.read_multiple(offsets_bytecounts)

    def close(self) -> None:
        """Close the reader and the TiffFile."""
        self._frame_reader.close()
        self._tiff_file.close()

    def __enter__(self) -> "OpenTileFile":
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.close()
