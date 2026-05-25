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

import gc
import weakref

import pytest

from opentile.cache import lru_cached_method


@pytest.mark.unittest
class TestLruCachedMethod:
    def test_caches_within_instance(self):
        # Arrange
        calls: list[int] = []

        class Example:
            @lru_cached_method()
            def value(self, x: int):
                calls.append(x)
                return object()

        example = Example()

        # Act
        first = example.value(1)
        second = example.value(1)

        # Assert
        assert first is second
        assert calls == [1]

    def test_separate_instances_have_separate_caches(self):
        # Arrange
        class Example:
            @lru_cached_method()
            def value(self, x: int):
                return object()

        first_instance = Example()
        second_instance = Example()

        # Act
        first = first_instance.value(1)
        second = second_instance.value(1)

        # Assert
        assert first is not second

    def test_distinguishes_arguments(self):
        # Arrange
        class Example:
            @lru_cached_method()
            def value(self, x: int):
                return object()

        example = Example()

        # Act
        first = example.value(1)
        second = example.value(2)

        # Assert
        assert first is not second

    def test_distinct_methods_use_distinct_caches(self):
        # Arrange
        class Example:
            @lru_cached_method()
            def one(self, x: int):
                return object()

            @lru_cached_method()
            def two(self, x: int):
                return object()

        example = Example()

        # Act: call both methods with the same argument
        one = example.one(1)
        two = example.two(1)

        # Assert: distinct results -> each method has its own cache (a shared
        # cache would return one's value for two's call on the same key)
        assert one is not two

    def test_bounded_evicts_when_full(self):
        # Arrange
        class Example:
            @lru_cached_method(maxsize=1)
            def value(self, x: int):
                return object()

        example = Example()

        # Act
        one = example.value(1)
        two = example.value(2)  # cache full (maxsize 1) -> 1 is evicted
        two_again = example.value(2)  # 2 is still cached -> not recomputed
        one_again = example.value(1)  # 1 was evicted -> recomputed

        # Assert
        assert two_again is two  # newest entry is cached
        assert one_again is not one  # older entry was evicted, recomputed

    def test_maxsize_callable_is_evaluated_lazily(self):
        # Arrange
        size = {"value": 2}

        class Example:
            @lru_cached_method(maxsize=lambda: size["value"])
            def value(self, x: int):
                return object()

        # Act: raise the size before the first call; with lazy evaluation the
        # cache uses 3 (read at first call), not the 2 it had at decoration time.
        size["value"] = 3
        example = Example()
        first_zero = example.value(0)
        example.value(1)
        example.value(2)
        second_zero = example.value(0)

        # Assert: entry 0 survived three inserts -> size is 3 (size 2 would have
        # evicted it)
        assert second_zero is first_zero

    def test_instance_is_reclaimed_after_drop(self):
        # Arrange
        class Example:
            @lru_cached_method()
            def value(self, x: int):
                return [x] * 1000

        example = Example()
        example.value(1)  # populate the cache before dropping the instance
        reference = weakref.ref(example)

        # Act
        del example
        gc.collect()

        # Assert
        assert reference() is None
