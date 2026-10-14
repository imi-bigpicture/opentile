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
import threading
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

from opentile.config import get_settings

# Local files are opened without an access hint (such as O_SEQUENTIAL or O_RANDOM
# on Windows), leaving read-ahead to the operating system, which detects the short
# sequential runs of tiles read by a viewer as well as the full rows read by a
# converter. Read cold, O_SEQUENTIAL read all tiles in order no faster and the
# tiles of a view slower, and O_RANDOM was slower for both.
_READ_FLAGS = os.O_RDONLY | getattr(os, "O_BINARY", 0)


class FrameReader(ABC):
    """Reads frames at absolute offsets of a file, ready to serve.

    A reader can prepend a prefix to every frame it reads, in place of the start of
    the frame as stored (see `with_prefix()`), e.g. to make a complete jpeg of an
    abbreviated one stored in a tiff page with jpeg tables. Is thread safe."""

    @abstractmethod
    def read(self, offset: int, bytecount: int) -> bytes:
        """Return the frame of `bytecount` bytes stored at `offset`."""

    @abstractmethod
    def read_multiple(
        self, offsets_bytecounts: Sequence[tuple[int, int]]
    ) -> list[bytes]:
        """Return the frame for each (offset, bytecount)."""

    @abstractmethod
    def with_prefix(self, prefix: bytes, skip: int) -> "FrameReader":
        """Return a reader of the same file, sharing its resources, that reads a
        frame without its first `skip` bytes and with `prefix` before it.

        Parameters
        ----------
        prefix: bytes
            Bytes to put before every frame read.
        skip: int
            Number of bytes at the start of every frame not to read.

        Returns
        -------
        FrameReader
            Reader of the same file. Closing it does not close this reader.
        """

    @abstractmethod
    def close(self) -> None:
        """Release any resources held by the reader."""


class FileHandlePool:
    """A pool of reusable raw file handles for a single path. `acquire()` hands
    out a free handle, opening a new one only when none is free, so the number of
    open handles tracks peak concurrency rather than total callers, up to
    `max_handles`. With that many open, a caller waits for a handle to be returned,
    so that many threads reading a slide of many files do not open a handle each
    for every file.

    A reader reading many small frames takes and returns handles through
    `available` and `open_or_wait()` itself, rather than through `acquire()`, as the
    generator of a context manager costs a noticeable part of such a read."""

    def __init__(self, path: str, max_handles: int) -> None:
        """
        Parameters
        ----------
        path: str
            Path of the local file to read.
        max_handles: int
            Most handles to have open at once.
        """
        if max_handles < 1:
            raise ValueError(f"Max handles must be at least 1, got {max_handles}.")
        self._path = path
        self._max_handles = max_handles
        self._available: queue.SimpleQueue[BinaryIO] = queue.SimpleQueue()
        self._handles: list[BinaryIO] = []
        self._lock = threading.Lock()
        self._closed = False

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._path})"

    @property
    def available(self) -> "queue.SimpleQueue[BinaryIO]":
        """Handles free to take. A handle taken is to be put back when done with."""
        return self._available

    @property
    def handle_count(self) -> int:
        """Number of handles open, at most `max_handles`."""
        with self._lock:
            return len(self._handles)

    @contextmanager
    def acquire(self) -> Iterator[BinaryIO]:
        """Yield a handle for the duration of the block, returning it to the pool
        on exit."""
        try:
            handle = self._available.get_nowait()
        except queue.Empty:
            handle = self.open_or_wait()
        try:
            yield handle
        finally:
            self._available.put(handle)

    def open_or_wait(self) -> BinaryIO:
        """Open a new handle, to be closed with the pool, or, if `max_handles` are
        open, wait for one of them to be returned. For use when no handle is
        available; the handle is to be put in `available` when done with.

        Raises ValueError if the pool is closed, rather than opening a handle that
        nothing would close. A caller waiting when the pool is closed gets a closed
        handle, and so raises when reading with it."""
        with self._lock:
            if self._closed:
                raise ValueError(f"{self} is closed.")
            if len(self._handles) < self._max_handles:
                fd = os.open(self._path, _READ_FLAGS)
                handle = cast(BinaryIO, os.fdopen(fd, "rb", buffering=0))
                self._handles.append(handle)
                return handle
        return self._available.get()

    def close(self) -> None:
        """Close the handles of the pool. A handle taken before is closed too, so a
        read with it after raises."""
        with self._lock:
            self._closed = True
            handles, self._handles = self._handles, []
        for handle in handles:
            handle.close()


class LocalFileReader(FrameReader):
    """Reads a local file through a pool of raw handles with OS read-ahead."""

    def __init__(
        self,
        pool: FileHandlePool,
        prefix: bytes = b"",
        skip: int = 0,
        owns_pool: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        pool: FileHandlePool
            Pool of handles of the file to read.
        prefix: bytes = b""
            Bytes to put before every frame read.
        skip: int = 0
            Number of bytes at the start of every frame not to read.
        owns_pool: bool = True
            If closing the reader closes the pool.
        """
        self._pool = pool
        self._available = pool.available
        self._prefix = prefix
        self._skip = skip
        self._owns_pool = owns_pool

    @staticmethod
    def _read_at(handle: BinaryIO, offset: int, bytecount: int) -> bytes:
        """Return `bytecount` bytes at `offset`, or fewer if the file ends before."""
        handle.seek(offset)
        data = handle.read(bytecount)
        if len(data) == bytecount:
            return data
        # An unbuffered read returns what one system call gives, which may be less
        # than asked for without the file having ended (e.g. on a network drive).
        parts = [data]
        remaining = bytecount - len(data)
        while remaining > 0:
            part = handle.read(remaining)
            if not part:
                break
            parts.append(part)
            remaining -= len(part)
        return b"".join(parts)

    def read(self, offset: int, bytecount: int) -> bytes:
        # Takes and returns the handle itself rather than with `acquire()`, see
        # `FileHandlePool`.
        try:
            handle = self._available.get_nowait()
        except queue.Empty:
            handle = self._pool.open_or_wait()
        try:
            data = self._read_at(handle, offset + self._skip, bytecount - self._skip)
        finally:
            self._available.put(handle)
        if self._prefix:
            return self._prefix + data
        return data

    def with_prefix(self, prefix: bytes, skip: int) -> "LocalFileReader":
        return LocalFileReader(self._pool, prefix, skip, owns_pool=False)

    def read_multiple(
        self, offsets_bytecounts: Sequence[tuple[int, int]]
    ) -> list[bytes]:
        """Coalesce contiguous frames into a single read and slice them apart."""
        if len(offsets_bytecounts) == 1:
            offset, bytecount = offsets_bytecounts[0]
            return [self.read(offset, bytecount)]
        frames = self._read_stored_frames(offsets_bytecounts)
        if not self._skip and not self._prefix:
            return frames
        return [self._prefix + frame[self._skip :] for frame in frames]

    def _read_stored_frames(
        self, offsets_bytecounts: Sequence[tuple[int, int]]
    ) -> list[bytes]:
        """Read frames as stored, coalescing contiguous frames into single reads."""
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
        if self._owns_pool:
            self._pool.close()


class FsspecReader(FrameReader):
    """Reads a remote file via the filesystem's ranged reads."""

    def __init__(
        self, fs: AbstractFileSystem, path: str, prefix: bytes = b"", skip: int = 0
    ) -> None:
        """
        Parameters
        ----------
        fs: AbstractFileSystem
            File system of the file.
        path: str
            Path of the file in the file system.
        prefix: bytes = b""
            Bytes to put before every frame read.
        skip: int = 0
            Number of bytes at the start of every frame not to read.
        """
        self._fs = fs
        self._path = path
        self._prefix = prefix
        self._skip = skip

    def read(self, offset: int, bytecount: int) -> bytes:
        data = cast(
            bytes,
            self._fs.cat_file(self._path, offset + self._skip, offset + bytecount),
        )
        if self._prefix:
            return self._prefix + data
        return data

    def read_multiple(
        self, offsets_bytecounts: Sequence[tuple[int, int]]
    ) -> list[bytes]:
        starts = [offset + self._skip for offset, _ in offsets_bytecounts]
        ends = [offset + bytecount for offset, bytecount in offsets_bytecounts]
        frames = self._fs.cat_ranges([self._path] * len(starts), starts, ends)
        if not self._prefix:
            return frames
        return [self._prefix + frame for frame in frames]

    def with_prefix(self, prefix: bytes, skip: int) -> "FsspecReader":
        return FsspecReader(self._fs, self._path, prefix, skip)

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
            return LocalFileReader(
                FileHandlePool(path, get_settings().max_file_handles_per_file)
            )
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

    @property
    def frame_reader(self) -> FrameReader:
        """Return the reader frames are read from the file with. Is thread safe."""
        return self._frame_reader

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
