import math
from typing import List, Type

from tifffile.tifffile import TiffTag

from opentile.geometry import Size, SizeMm


def split_and_cast_text(string: str, type: Type) -> List[any]:
    return [type(element) for element in string.replace('"', '').split()]


def get_value_from_tiff_tags(
    tiff_tags: List[TiffTag],
    value_name: str
) -> str:
    for tag in tiff_tags:
        if tag.name == value_name:
            return tag.value


def calculate_pyramidal_index(
    base_shape: Size,
    level_shape: Size
) -> int:
    return int(
        math.log2(base_shape.width/level_shape.width)
    )


def calculate_mpp(
    base_mpp: SizeMm,
    pyramid_index: int
) -> SizeMm:
    return base_mpp * pow(2, pyramid_index)


class Jpeg:
    TAGS = {
        'tag marker': 0xFF,
        'start of image': 0xD8,
        'application default header': 0xE0,
        'quantization table': 0xDB,
        'start of frame': 0xC0,
        'huffman table': 0xC4,
        'start of scan': 0xDA,
        'end of image': 0xD9,
        'restart interval': 0xDD,
        'restart mark': 0xD0
    }

    @classmethod
    def start_of_frame(cls) -> bytes:
        """Return bytes representing a start of frame tag."""
        return bytes([cls.TAGS['tag marker'], cls.TAGS['start of frame']])

    @classmethod
    def start_of_scan(cls) -> bytes:
        """Return bytes representing a start of scan tag."""
        return bytes([cls.TAGS['tag marker'], cls.TAGS['start of scan']])

    @classmethod
    def end_of_image(cls) -> bytes:
        """Return bytes representing a end of image tag."""
        return bytes([cls.TAGS['tag marker'], cls.TAGS['end of image']])

    @classmethod
    def restart_mark(cls, index: int) -> bytes:
        """Return bytes representing a restart marker of index (0-7), without
        the prefixing tag (0xFF)."""
        return bytes([cls.TAGS['restart mark'] + index % 8])
