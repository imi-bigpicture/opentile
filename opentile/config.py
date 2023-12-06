#    Copyright 2023 SECTRA AB
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

"""General settings."""


class Settings:
    """Class containing settings. Settings are to be accessed through the
    global variable settings."""

    def __init__(self) -> None:
        self._ndpi_frame_cache = 128

    @property
    def ndpi_frame_cache(self) -> int:
        """Number of frames to cache for ndpi."""
        return self._ndpi_frame_cache

    @ndpi_frame_cache.setter
    def ndpi_frame_cache(self, value: int) -> None:
        self._ndpi_frame_cache = value


settings = Settings()
"""Global settings variable."""
