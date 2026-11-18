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
single ``except OpenTileError``. `UnsupportedFileError`,
`NonSupportedCompressionError` and `NonDyadicPyramidLevelError` also inherit the
built-in `NotImplementedError`, and the metadata errors (`MissingMetadataError`,
`InvalidMetadataError`, `UnsupportedMetadataError`) the built-in `ValueError`, so
code that catches those keeps working.

A file opentile does not read raises `UnsupportedFileError` or one of the metadata
errors, so a caller that tries several readers can move on to the next on an
`OpenTileError` and still see other errors, such as a failing read.
"""


class OpenTileError(Exception):
    """Base class for all opentile errors."""


class UnsupportedFileError(OpenTileError, NotImplementedError):
    """Raised when no tiler supports the opened file, or the file is of a supported
    format in a variant opentile does not read."""


class NonSupportedCompressionError(OpenTileError, NotImplementedError):
    """Raised when an image's compression is not supported for the requested
    operation."""


class NonDyadicPyramidLevelError(OpenTileError, NotImplementedError):
    """Raised when a level's downsample is not a clean power of two, so it cannot be
    placed in the pyramid."""


class MissingMetadataError(OpenTileError, ValueError):
    """Raised when a file lacks metadata opentile needs to read it, such as the
    pixel size."""


class InvalidMetadataError(OpenTileError, ValueError):
    """Raised when a file has metadata opentile needs to read it, but with values
    that are unlikely to be valid for the format, such as a malformed number or a
    negative size."""


class UnsupportedMetadataError(OpenTileError, ValueError):
    """Raised when a file has metadata opentile needs to read it, with values that
    may be valid for the format but that opentile does not handle, such as a layout
    it can not place."""


class MissingAssociatedImageError(OpenTileError):
    """Raised when a requested associated image (label, overview, or thumbnail) is not
    present in the file."""
