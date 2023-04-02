#    Copyright 2021, 2022, 2023 SECTRA AB
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


from typing import Dict, List

from opentile.geometry import Point


class NdpiCache:
    """Cache for bytes ordered by tile position. Oldest entry is removed when
    size of content is above set size."""

    def __init__(self, size: int):
        """Create cache for size items.

        Parameters
        ----------
        size: int
            Size of the cache.

        """
        self._size = size
        self._content: Dict[Point, bytes] = {}
        self._history: List[Point] = []

    def __len__(self) -> int:
        return len(self._history)

    def __str__(self) -> str:
        return (
            f"{type(self).__name__} of size {len(self)} " f"and max size {self._size}"
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._size})"

    def __setitem__(self, key: Point, value: bytes) -> None:
        """Set item in cache. Remove old items if needed.

        Parameters
        ----------
        key: Point
            Key for item to set.
        value: bytes:
            Value for item to set.

        """
        self._content[key] = value
        self._history.append(key)
        self._remove_old()

    def __getitem__(self, key: Point) -> bytes:
        """Get item from cache.

        Parameters
        ----------
        key: Point
            Key for item to get.

        Returns
        ----------
        bytes
            Value for key.
        """
        return self._content[key]

    def __contains__(self, key: Point) -> bool:
        """Return true if key in cache.

        Parameters
        ----------
        key: Point
            Key to check for.

        Returns
        ----------
        bool
            True if key is in cache.
        """
        return key in self._content

    @property
    def size(self) -> int:
        return self._size

    def update(self, items: Dict[Point, bytes]) -> None:
        """Update items in cache. Remove old items if needed.

        Parameters
        ----------
        items: Dict[Point, bytes]
            Items to update.

        """
        self._content.update(items)
        self._history += list(items.keys())
        self._remove_old()

    def _remove_old(self) -> None:
        """Remove old items in cache if needed."""
        while len(self._history) > self._size:
            key_to_remove = self._history.pop(0)
            self._content.pop(key_to_remove)
