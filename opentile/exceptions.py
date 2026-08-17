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

"""Exceptions raised by opentile.

All inherit `OpenTileError`, so a caller can catch everything opentile raises with a
single ``except OpenTileError``. The "not supported / not implemented" errors also
inherit the built-in `NotImplementedError`, so code that catches that keeps working.
"""


class OpenTileError(Exception):
    """Base class for all opentile errors."""


class UnsupportedFileError(OpenTileError, NotImplementedError):
    """Raised when no tiler supports the opened file."""


class NonSupportedCompressionError(OpenTileError, NotImplementedError):
    """Raised when an image's compression is not supported for the requested
    operation."""


class NonDyadicPyramidLevelError(OpenTileError, NotImplementedError):
    """Raised when a level's downsample is not a clean power of two, so it cannot be
    placed in the pyramid."""


class MissingAssociatedImageError(OpenTileError):
    """Raised when a requested associated image (label, overview, or thumbnail) is not
    present in the file."""
