from typing import Tuple


def split_byte_into_nibbles(value: int) -> Tuple[int, int]:
    """Split a byte into two nibbles"""
    first = value >> 4
    second = value & 0x0F
    return first, second
