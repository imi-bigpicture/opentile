from typing import Tuple

def split_byte_into_nibbles(value: int) -> Tuple[int, int]:
    first = value >> 4
    second = value & 0x0F
    return first, second