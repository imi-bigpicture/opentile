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

"""Per-instance method caching.

:func:`lru_cached_method` is the per-instance counterpart of
``functools.lru_cache``: the cache (and its condition) live on the instance, so
they are released together with the instance instead of being pinned on the
class for the lifetime of the process. The caching itself is delegated to
``cachetools.cachedmethod``; this only lazily creates the instance's cache and
condition.
"""

import threading
from typing import Callable, Optional, Union

from cachetools import LRUCache, cachedmethod


def lru_cached_method(
    maxsize: Optional[Union[int, Callable[[], Optional[int]]]] = 128,
):
    """Cache a method's return values in a per-instance LRU cache.

    Like ``functools.lru_cache`` but bound to the instance rather than the class.
    A per-instance ``threading.Condition`` makes concurrent calls for the same
    key wait for the first to finish instead of recomputing, so the wrapped
    method runs at most once per key while calls for different keys still run
    concurrently.

    Parameters
    ----------
    maxsize:
        Maximum number of entries before least-recently-used eviction, or
        ``None`` for an unbounded cache. May be a callable returning the size,
        evaluated when an instance's cache is first created, so a size taken from
        runtime configuration is honoured.
    """

    def decorator(method: Callable):
        cache_attr = f"_{method.__name__}_cache"
        condition_attr = f"_{method.__name__}_condition"

        def get_cache(self):
            cache = self.__dict__.get(cache_attr)
            if cache is None:
                size = maxsize() if callable(maxsize) else maxsize
                default = {} if size is None else LRUCache(maxsize=size)
                # setdefault is atomic, so racing first-callers share one cache.
                cache = self.__dict__.setdefault(cache_attr, default)
            return cache

        def get_condition(self):
            condition = self.__dict__.get(condition_attr)
            if condition is None:
                condition = self.__dict__.setdefault(
                    condition_attr, threading.Condition()
                )
            return condition

        return cachedmethod(get_cache, condition=get_condition)(method)

    return decorator
