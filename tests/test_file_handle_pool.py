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


import threading
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO

import pytest

from opentile.file import FileHandlePool


@pytest.fixture
def pool(tmp_path: Path) -> Iterator[FileHandlePool]:
    file = tmp_path / "data.bin"
    file.write_bytes(bytes(range(256)) * 16)
    pool = FileHandlePool(str(file))
    yield pool
    pool.close()


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
