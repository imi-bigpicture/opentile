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

"""Tests of metadata handling: kept by default, removed with ``strip``."""

import pytest

import opentile_jpeg
from conftest import MakeJpeg

ICC = (0xE2, b"ICC_PROFILE\0\x01\x01" + bytes(64))
EXIF = (0xE1, b"Exif\0\0MM\0*" + bytes(16))
XMP = (0xE1, b"http://ns.adobe.com/xap/1.0/\0<x:xmpmeta/>")
COMMENT = (0xFE, b"scanned slide")
IPTC = (0xED, b"Photoshop 3.0\0" + bytes(8))
ALL_METADATA = [ICC, EXIF, XMP, COMMENT, IPTC]


def _with_metadata(jpeg: bytes, segments: list[tuple[int, bytes]]) -> bytes:
    """Insert marker segments after the JFIF APP0 segment."""
    assert jpeg[2:4] == b"\xff\xe0"
    end_of_app0 = 4 + int.from_bytes(jpeg[4:6], "big")
    inserted = b"".join(
        bytes([0xFF, marker]) + (len(data) + 2).to_bytes(2, "big") + data
        for marker, data in segments
    )
    return jpeg[:end_of_app0] + inserted + jpeg[end_of_app0:]


def _metadata(jpeg: bytes) -> list[tuple[int, bytes]]:
    """The APPn and COM segments before the first scan, minus JFIF and Adobe."""
    segments, i = [], 2
    while jpeg[i + 1] != 0xDA:
        marker, length = jpeg[i + 1], int.from_bytes(jpeg[i + 2 : i + 4], "big")
        data = jpeg[i + 4 : i + 2 + length]
        is_app = 0xE0 <= marker <= 0xEF
        if (is_app or marker == 0xFE) and not data.startswith((b"JFIF\0", b"Adobe")):
            segments.append((marker, data))
        i += 2 + length
    return segments


@pytest.fixture
def tagged(make_jpeg: MakeJpeg) -> bytes:
    """A JPEG carrying one segment of every metadata kind."""
    return _with_metadata(make_jpeg(64, 64, seed=4), ALL_METADATA)


class TestMetadata:
    @pytest.mark.parametrize(
        "operation",
        [
            lambda jpeg: opentile_jpeg.optimize(jpeg),
            lambda jpeg: opentile_jpeg.crop(jpeg, 16, 16, 32, 32),
            lambda jpeg: next(
                opentile_jpeg.crop_multiple(jpeg, [(0, 0, 32, 32), (32, 0, 32, 32)])
            ),
            lambda jpeg: next(
                opentile_jpeg.crop_multiple(jpeg, [(0, 0, 32, 32), (32, 0, 16, 16)])
            ),
            lambda jpeg: opentile_jpeg.fill(jpeg, 255),
            lambda jpeg: next(
                opentile_jpeg.compose(
                    [(jpeg, 0, 0)],
                    content_size=(64, 64),
                    tile_size=(64, 64),
                    positions=[(0, 0)],
                )
            ),
            lambda jpeg: opentile_jpeg.encode_coefficients(
                *opentile_jpeg.decode_coefficients(jpeg)
            ),
        ],
        ids=[
            "optimize",
            "crop",
            "cropper",
            "crop_varying",
            "fill",
            "compose",
            "encode",
        ],
    )
    def test_kept_by_default(self, tagged: bytes, operation) -> None:
        # Act
        out = operation(tagged)

        # Assert
        assert _metadata(out) == ALL_METADATA

    @pytest.mark.parametrize(
        ("strip", "kept"),
        [
            (opentile_jpeg.Metadata.EXIF, [ICC, XMP, COMMENT, IPTC]),
            (
                opentile_jpeg.Metadata.ICC | opentile_jpeg.Metadata.XMP,
                [EXIF, COMMENT, IPTC],
            ),
            (
                opentile_jpeg.Metadata.COMMENTS | opentile_jpeg.Metadata.OTHER,
                [ICC, EXIF, XMP],
            ),
            (opentile_jpeg.Metadata.ALL, []),
        ],
    )
    def test_strip_removes_only_the_named_kinds(
        self, tagged: bytes, strip: opentile_jpeg.Metadata, kept: list
    ) -> None:
        # Act
        out = opentile_jpeg.optimize(tagged, strip=strip)

        # Assert
        assert _metadata(out) == kept
        assert out[6:11] == b"JFIF\0"  # libjpeg-turbo still writes its own JFIF

    def test_strip_applies_to_every_tile(self, tagged: bytes) -> None:
        # Act
        tiles = opentile_jpeg.crop_multiple(
            tagged, [(0, 0, 32, 32), (32, 32, 32, 32)], strip=opentile_jpeg.Metadata.ALL
        )

        # Assert
        assert all(_metadata(tile) == [] for tile in tiles)
