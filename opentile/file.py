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


import threading
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

    @abstractmethod
    def read_multiple(
        self, offsets_bytecounts: Sequence[tuple[int, int]]
    ) -> list[bytes]:
        """Return bytes for each (offset, bytecount)."""

    @abstractmethod
    def close(self) -> None:
        """Release any resources held by the reader."""


class ThreadLocalHandle(threading.local):
    """Per-thread file handle for `LocalFileReader`."""

    handle: Optional[BinaryIO] = None


class LocalFileReader(FrameReader):
    """Reads a local file through a per-thread handle."""

    def __init__(self, fs: AbstractFileSystem, path: str) -> None:
        self._fs = fs
        self._path = path
        self._local = ThreadLocalHandle()
        self._handles: list[BinaryIO] = []

    def _handle(self) -> BinaryIO:
        handle = self._local.handle
        if handle is not None:
            return handle
        handle = cast(BinaryIO, self._fs.open(self._path))
        self._local.handle = handle
        self._handles.append(handle)
        return handle

    def read(self, offset: int, bytecount: int) -> bytes:
        handle = self._handle()
        handle.seek(offset)
        return handle.read(bytecount)

    def read_multiple(
        self, offsets_bytecounts: Sequence[tuple[int, int]]
    ) -> list[bytes]:
        """Coalesce contiguous reads into a single read and slice them apart."""
        if len(offsets_bytecounts) == 1:
            offset, bytecount = offsets_bytecounts[0]
            return [self.read(offset, bytecount)]
        result = [b""] * len(offsets_bytecounts)
        for run in self._contiguous_runs(offsets_bytecounts):
            start = offsets_bytecounts[run[0]][0]
            last_offset, last_bytecount = offsets_bytecounts[run[-1]]
            blob = self.read(start, last_offset + last_bytecount - start)
            if len(run) == 1:
                # The blob is exactly the frame; return it without a slice copy.
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
        for handle in self._handles:
            handle.close()


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
        """Read a local file through per-thread handles (fast, lock-free); read a
        remote file through the fsspec filesystem's ranged reads."""
        if isinstance(fs, LocalFileSystem):
            return LocalFileReader(fs, path)
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
        return self._frame_reader.read(offset, bytecount)

    def read_multiple(
        self, offsets_bytecounts: Sequence[tuple[int, int]]
    ) -> list[bytes]:
        """Return bytes from multiple locations from file. Is thread safe."""
        return self._frame_reader.read_multiple(offsets_bytecounts)

    def close(self):
        """Close the reader and the TiffFile."""
        self._frame_reader.close()
        self._tiff_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
