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
