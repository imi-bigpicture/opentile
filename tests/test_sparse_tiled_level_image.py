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

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from decoy import Decoy, matchers
from imagecodecs import jpeg8_decode
from tifffile import COMPRESSION, TiffPage

from opentile.file import FileHandlePool, LocalFileReader, OpenTileFile
from opentile.geometry import Size, SizeMm
from opentile.jpeg import Jpeg
from opentile.jpeg.jpeg import find_turbojpeg_path
from opentile.tiff_image_bases import SparseTiledLevelImage

test_file_path = "tests/testdata/turbojpeg/frame_2048x512.jpg"


def _split_tables_and_frame(frame: bytes) -> tuple[bytes, bytes]:
    """Split a complete jpeg into a tiff JPEGTables blob and an abbreviated frame,
    the way a tiled tiff page stores them."""
    tables = bytearray(b"\xff\xd8")
    abbreviated = bytearray(b"\xff\xd8")
    index = 2
    while index + 4 <= len(frame):
        marker = frame[index + 1]
        length = int.from_bytes(frame[index + 2 : index + 4], "big")
        segment = frame[index : index + 2 + length]
        if marker in (Jpeg.TAGS["quantization table"], Jpeg.TAGS["huffman table"]):
            tables += segment
        else:
            abbreviated += segment
        index += 2 + length
        if marker == Jpeg.TAGS["start of scan"]:
            abbreviated += frame[index:]
            break
    tables += b"\xff\xd9"
    return bytes(tables), bytes(abbreviated)


@pytest.fixture()
def frame():
    with open(test_file_path, "rb") as file:
        yield file.read()


@pytest.fixture()
def level(
    decoy: Decoy, frame: bytes, tmp_path: Path
) -> Iterator[SparseTiledLevelImage]:
    """A level of two tiles, where (0, 0) is stored and (1, 0) is sparse."""
    tables, abbreviated = _split_tables_and_frame(frame)
    path = tmp_path.joinpath("frames.bin")
    path.write_bytes(abbreviated)
    reader = LocalFileReader(FileHandlePool(str(path), 1))
    page = decoy.mock(cls=TiffPage)
    decoy.when(page.compression).then_return(COMPRESSION.JPEG)
    decoy.when(page.imagewidth).then_return(4096)
    decoy.when(page.imagelength).then_return(512)
    decoy.when(page.is_tiled).then_return(True)
    decoy.when(page.tilewidth).then_return(2048)
    decoy.when(page.tilelength).then_return(512)
    decoy.when(page.jpegtables).then_return(tables)
    decoy.when(page.dataoffsets).then_return((0, 0))
    decoy.when(page.databytecounts).then_return((len(abbreviated), 0))
    # The file holds the stored tile.
    file = decoy.mock(cls=OpenTileFile)
    decoy.when(file.frame_reader).then_return(reader)
    decoy.when(file.read(matchers.Anything(), matchers.Anything())).then_do(reader.read)
    decoy.when(file.read_multiple(matchers.Anything())).then_do(reader.read_multiple)
    yield SparseTiledLevelImage(
        page,
        file,
        Size(4096, 512),
        SizeMm(0.25, 0.25),
        Jpeg(find_turbojpeg_path()),
    )
    reader.close()


@pytest.mark.unittest
class TestSparseTiledLevelImage:
    def test_sparse_tile_is_blank(self, level: SparseTiledLevelImage):
        # Arrange

        # Act
        tile = level.get_tile((1, 0))

        # Assert
        assert jpeg8_decode(tile).mean() == 255.0

    def test_get_populated_tile(self, level: SparseTiledLevelImage, frame: bytes):
        # Arrange

        # Act
        tile = level.get_tile((0, 0))

        # Assert
        assert np.array_equal(jpeg8_decode(tile), jpeg8_decode(frame))

    def test_sparse_tile_is_blank_after_populated_tile(
        self, level: SparseTiledLevelImage
    ):
        # Arrange
        # The stored tile is abbreviated and the blank tile is a complete jpeg, so a
        # prefix cached from the one must not be applied to the other.
        level.get_tile((0, 0))

        # Act
        tile = level.get_tile((1, 0))

        # Assert
        assert jpeg8_decode(tile).mean() == 255.0

    def test_get_populated_tile_after_sparse_tile(
        self, level: SparseTiledLevelImage, frame: bytes
    ):
        # Arrange
        level.get_tile((1, 0))

        # Act
        tile = level.get_tile((0, 0))

        # Assert
        assert np.array_equal(jpeg8_decode(tile), jpeg8_decode(frame))

    def test_get_tiles_serves_sparse_and_populated_tiles(
        self, level: SparseTiledLevelImage, frame: bytes
    ):
        # Arrange

        # Act
        tiles = list(level.get_tiles([(0, 0), (1, 0)]))

        # Assert
        assert np.array_equal(jpeg8_decode(tiles[0]), jpeg8_decode(frame))
        assert jpeg8_decode(tiles[1]).mean() == 255.0

    def test_get_tiles_serves_sparse_tile_past_stored_frames(
        self, level: SparseTiledLevelImage
    ):
        # Arrange
        # Only one frame is stored, so tile (1, 0) has no entry at all in the page.

        # Act
        tiles = list(level.get_tiles([(1, 0)]))

        # Assert
        assert jpeg8_decode(tiles[0]).mean() == 255.0
