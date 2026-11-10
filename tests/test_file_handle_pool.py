#    Copyright 2026 SECTRA AB
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


import io
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from typing import BinaryIO

import pytest

from opentile.file import FileHandlePool, LocalFileReader

CONTENT = bytes(range(256)) * 16
MAX_HANDLES = 8
"""As many as the most threads that hold a handle at once in the tests of the pool."""
WAIT_SECONDS = 10
"""The longest a test waits for another thread. A fault then fails the test instead of
hanging it."""
NOT_RETURNED_SECONDS = 0.2
"""How long a caller that should be waiting is given to return. A caller that is slow to
start also passes, as it has not returned either."""


@pytest.fixture
def path(tmp_path: Path) -> Path:
    file = tmp_path / "data.bin"
    file.write_bytes(CONTENT)
    return file


@pytest.fixture
def pool(path: Path) -> Iterator[FileHandlePool]:
    pool = FileHandlePool(str(path), MAX_HANDLES)
    yield pool
    pool.close()


@pytest.mark.unittest
class TestFileHandlePool:
    def test_sequential_acquires_reuse_one_handle(self, pool: FileHandlePool) -> None:
        # Arrange
        handles: list[BinaryIO] = []

        # Act
        for _ in range(10):
            with pool.acquire() as handle:
                handles.append(handle)

        # Assert
        assert len({id(handle) for handle in handles}) == 1

    def test_concurrent_acquires_get_distinct_handles(
        self, pool: FileHandlePool
    ) -> None:
        # Arrange
        count = 8
        all_acquired = threading.Barrier(count)
        lock = threading.Lock()
        held: list[BinaryIO] = []

        def worker() -> None:
            with pool.acquire() as handle:
                with lock:
                    held.append(handle)
                all_acquired.wait()  # hold until all `count` are out at once

        threads = [threading.Thread(target=worker) for _ in range(count)]

        # Act
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Assert -- no handle was handed to two threads at once
        assert len({id(handle) for handle in held}) == count

    def test_acquired_handle_reads_the_file(self, pool: FileHandlePool) -> None:
        # Arrange
        expected = bytes(range(4))

        # Act
        with pool.acquire() as handle:
            handle.seek(0)
            data = handle.read(4)

        # Assert
        assert data == expected

    def test_close_closes_all_opened_handles(self, pool: FileHandlePool) -> None:
        # Arrange
        with pool.acquire() as first, pool.acquire() as second:
            pass

        # Act
        pool.close()

        # Assert
        assert first.closed
        assert second.closed

    def test_open_or_wait_after_close_raises(self, pool: FileHandlePool) -> None:
        # Arrange
        pool.close()

        # Act & Assert
        with pytest.raises(ValueError):
            pool.open_or_wait()

    def test_acquire_beyond_max_handles_waits_for_a_handle_returned(
        self, path: Path
    ) -> None:
        # Arrange
        single_handle_pool = FileHandlePool(str(path), 1)

        def acquire_other() -> BinaryIO:
            with single_handle_pool.acquire() as handle:
                return handle

        try:
            with (
                ThreadPoolExecutor(1) as executor,
                single_handle_pool.acquire() as held,
            ):
                # Act
                other = executor.submit(acquire_other)
                wait([other], NOT_RETURNED_SECONDS)
                done_while_held = other.done()

            # Assert
            assert not done_while_held
            assert other.result(WAIT_SECONDS) is held
            assert single_handle_pool.handle_count == 1
        finally:
            single_handle_pool.close()

    def test_reads_from_more_threads_than_max_handles_open_max_handles(
        self, path: Path
    ) -> None:
        """When all handles are taken, a caller waits for one instead of opening
        another. Each caller still reads its own bytes."""
        # Arrange
        capped_pool = FileHandlePool(str(path), 2)
        reader = LocalFileReader(capped_pool)
        offsets = [offset for _ in range(20) for offset in range(0, 4000, 37)]

        # Act
        with ThreadPoolExecutor(16) as executor:
            data = list(executor.map(lambda offset: reader.read(offset, 50), offsets))

        # Assert
        assert data == [CONTENT[offset : offset + 50] for offset in offsets]
        assert 1 <= capped_pool.handle_count <= 2
        reader.close()

    def test_create_with_no_handles_raises(self, path: Path) -> None:
        # Arrange

        # Act & Assert
        with pytest.raises(ValueError):
            FileHandlePool(str(path), 0)


@pytest.mark.unittest
class TestLocalFileReader:
    def test_read_returns_bytes_at_offset(self, pool: FileHandlePool) -> None:
        # Arrange
        reader = LocalFileReader(pool)

        # Act
        data = reader.read(100, 10)

        # Assert
        assert data == CONTENT[100:110]

    def test_reads_from_threads_at_once_return_bytes_at_their_offsets(
        self, pool: FileHandlePool
    ) -> None:
        # Arrange
        reader = LocalFileReader(pool)
        offsets = list(range(0, 4000, 37))

        # Act
        with ThreadPoolExecutor(8) as executor:
            data = list(executor.map(lambda offset: reader.read(offset, 50), offsets))

        # Assert
        assert data == [CONTENT[offset : offset + 50] for offset in offsets]

    def test_read_with_prefix_puts_prefix_in_place_of_skipped_bytes(
        self, pool: FileHandlePool
    ) -> None:
        # Arrange
        reader = LocalFileReader(pool).with_prefix(b"prefix", 4)

        # Act
        data = reader.read(100, 10)

        # Assert
        assert data == b"prefix" + CONTENT[104:110]

    def test_read_multiple_with_prefix_puts_prefix_in_place_of_skipped_bytes(
        self, pool: FileHandlePool
    ) -> None:
        # Arrange
        reader = LocalFileReader(pool).with_prefix(b"prefix", 4)

        # Act
        data = reader.read_multiple([(100, 10), (110, 10), (300, 10)])

        # Assert
        assert data == [
            b"prefix" + CONTENT[104:110],
            b"prefix" + CONTENT[114:120],
            b"prefix" + CONTENT[304:310],
        ]

    def test_close_of_reader_with_prefix_leaves_reader_of_file_open(
        self, pool: FileHandlePool
    ) -> None:
        # Arrange
        reader = LocalFileReader(pool)
        reader.read(0, 4)

        # Act
        reader.with_prefix(b"prefix", 4).close()

        # Assert
        assert reader.read(0, 4) == CONTENT[0:4]

    @pytest.mark.parametrize("prefix", [b"", b"prefix"])
    def test_read_reads_on_after_a_short_read(
        self, pool: FileHandlePool, prefix: bytes
    ) -> None:
        # Arrange
        pool.available.put(ShortReadingHandle(CONTENT, 3))
        reader = LocalFileReader(pool).with_prefix(prefix, 0)

        # Act
        data = reader.read(100, 10)

        # Assert
        assert data == prefix + CONTENT[100:110]

    def test_read_multiple_reads_on_after_a_short_read(
        self, pool: FileHandlePool
    ) -> None:
        # Arrange
        pool.available.put(ShortReadingHandle(CONTENT, 3))
        reader = LocalFileReader(pool)

        # Act
        data = reader.read_multiple([(100, 10), (110, 10), (300, 10)])

        # Assert
        assert data == [CONTENT[100:110], CONTENT[110:120], CONTENT[300:310]]

    def test_read_stops_at_end_of_file(self, pool: FileHandlePool) -> None:
        # Arrange
        reader = LocalFileReader(pool)

        # Act
        data = reader.read(len(CONTENT) - 4, 10)

        # Assert
        assert data == CONTENT[-4:]

    def test_read_after_close_raises(self, pool: FileHandlePool) -> None:
        # Arrange
        reader = LocalFileReader(pool)
        reader.read(0, 4)
        reader.close()

        # Act & Assert
        with pytest.raises(ValueError):
            reader.read(0, 4)


class ShortReadingHandle(io.BytesIO):
    """In-memory handle that returns at most `max_read` bytes per read, like an
    unbuffered handle on a network drive."""

    def __init__(self, content: bytes, max_read: int) -> None:
        super().__init__(content)
        self._max_read = max_read

    def read(self, size: int | None = -1, /) -> bytes:
        if size is None or size < 0:
            size = self._max_read
        return super().read(min(size, self._max_read))
