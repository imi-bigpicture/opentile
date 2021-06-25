import io
from dataclasses import dataclass
from struct import unpack
from typing import Callable, Dict, List, Optional, Tuple, Union

from bitstring import ConstBitStream, Bits

from ndpi_tiler.stream import Stream
from ndpi_tiler.utils import split_byte_into_nibbles


@dataclass
class HuffmanTableIdentifier:
    """Identifies for Huffman table. Mode is either 'DC' or 'AC', selection
    either 0 or 1."""
    mode: str
    selection: int

    @classmethod
    def from_byte(cls, data: int) -> 'HuffmanTableIdentifier':
        mode, selection = split_byte_into_nibbles(data)
        if mode == 0:
            str_mode = 'DC'
        else:
            str_mode = 'AC'
        return cls(str_mode, selection)

    def __hash__(self) -> int:
        return hash((self.mode, self.selection))


@dataclass
class HuffmanLeaf:
    """Huffman leaf, only contains a value"""
    value: int


@dataclass
class HuffmanNode:
    """Huffman node, contains up to two nodes, that are either other
    Huffman nodes or leaves."""

    def __init__(self, length: int) -> None:
        """Create a Huffman node at tree depth (symbol length).

        Parameters
        ----------
        length: int
            The symbol length at this node.

        """

        self._length: int = length
        self._nodes: List[Optional[Union['HuffmanNode', HuffmanLeaf]]] = []

    def __len__(self) -> int:
        return len(self._nodes)

    @property
    def full(self) -> bool:
        """Return True if node is full."""
        return len(self._nodes) > 1

    def shift(self, length) -> int:
        """Shift the symbol 'bit' from this node to the correct bit position.
        """
        return 2**(length - self._length)

    def _append(self, item: Union['HuffmanNode', HuffmanLeaf]) -> int:
        """Append a node or leaf to this node and return the insertion index.
        """
        self._nodes.append(item)
        return len(self._nodes) - 1

    def _insert_into_self(
        self,
        leaf: HuffmanLeaf,
        length: int
    ) -> Optional[int]:
        """Return (partial) symbol for leaf if leaf could be inserted as child
        to this node. Returns None if not inserted."""
        if length == self._length and not self.full:
            return self._append(leaf)
        return None

    def _insert_into_child(
        self,
        leaf: HuffmanLeaf,
        length: int
    ) -> Optional[int]:
        """Return (partial) symbol for leaf if leaf could be inserted in a
        child (or a child of a child, recursively) to this node. Returns None
        if not inserted."""
        for index, node in enumerate(self._nodes):
            if isinstance(node, HuffmanNode):
                # Try to insert leaf into child node
                symbol = node.insert(leaf, length)
                if symbol is not None:
                    return symbol + self.shift(length) * index
        return None

    def _insert_into_new_child(
        self,
        leaf: HuffmanLeaf,
        length: int
    ) -> Optional[int]:
        """Return (partial) symbol for leaf if leaf could be inserted as a new
        child to this node. Returns None if not inserted."""
        if self.full:
            return None
        node = HuffmanNode(self._length + 1)
        index = self._append(node)
        symbol = node.insert(leaf, length)
        return symbol + self.shift(length) * index

    def insert(
        self,
        leaf: HuffmanLeaf,
        length: int
    ) -> Optional[int]:
        """Returns symbol for leaf if leaf could be fit inside this node
        or this node's children, recursivley). Returns None if not inserted.
        """
        # Insertion order:
        # 1. Try to insert leaf directly into this node
        # 2. If there is a child node, try to insert into that
        # 3. Otherwise try to create a new child node
        insertion_order: List[Callable([HuffmanLeaf, int], Optional[int])] = [
            self._insert_into_self,
            self._insert_into_child,
            self._insert_into_new_child
        ]
        for insertion_function in insertion_order:
            symbol = insertion_function(leaf, length)
            if symbol is not None:
                return symbol

        # No space for a new child node, insert leaf somewhere else
        return None

    def get(self, key: int) -> Union[None, HuffmanLeaf, 'HuffmanNode']:
        """Return node child from this node"""
        try:
            return self._nodes[key]
        except IndexError:
            return None


class HuffmanTable:
    """Huffman table that can be used to decode bytes"""
    def __init__(
        self,
        identifer: HuffmanTableIdentifier,
        values_in_levels: List[List[int]]
    ) -> None:
        """Create a Huffman table from specifed table with symbols per level.
        Only the first Huffman table in the data is parsed. The number of bytes
        read for creating the table is avaiable in property byte_length, that
        can be used to read multiple tables.

        Parameters
        ----------
        identifer: HuffmanTableIdentifier
            Identifies the use of the table (DC or AC, selection)
        symbols_in_levels: List[List[int]]
            Symbols in the table, listed per level

        """
        self._root = HuffmanNode(0)
        self._identifier = identifer
        self.encode_dict: Dict[int, Tuple[int, int]] = {}
        self.decode_dict: Dict[Tuple[int, int], int] = {}

        for length, level in enumerate(values_in_levels):
            for value in level:
                leaf = HuffmanLeaf(value)
                # Return symbol if leaf inserted
                symbol = self._root.insert(leaf, length)
                if symbol is None:
                    raise ValueError(
                        f"Huffman table not correct "
                        f"identifier {identifer}, symbol {symbol}, "
                        f"depth {length}"
                    )
                if value in self.encode_dict.keys():
                    raise ValueError(
                        f"Duplicate value/symbol in Huffman table "
                        f"identifier {identifer}, symbol {symbol}, "
                        f"value {value}"
                    )
                self.encode_dict[value] = (symbol, length+1)
                self.decode_dict[(symbol, length+1)] = value
        print(self.decode_dict)

    @property
    def identifier(self) -> HuffmanTableIdentifier:
        """Header of the Huffman table"""
        return self._identifier

    @classmethod
    def from_data(cls, data: bytes) -> Tuple['HuffmanTable', int]:
        """Create a Huffman table using data from Quantization Table payload.
        Only the first Huffman table in the data is parsed. The number of bytes
        read for creating the table is avaiable in property byte_length, that
        can be used to read multiple tables.

        Parameters
        ----------
        data: bytes
            Quantization Table payload

        Returns
        ----------
        Tuple[HuffmanTable, int]
            Created Huffman table and length read from data

        """
        with io.BytesIO(data) as buffer:
            header: int = unpack('B', buffer.read(1))[0]
            identifier = HuffmanTableIdentifier.from_byte(header)

            symbols_per_level: Tuple[int] = unpack('B'*16, buffer.read(16))
            symbols_in_levels: List[List[int]] = [
                list(unpack(
                    'B'*number_of_symbols,
                    buffer.read(number_of_symbols)
                ))
                for number_of_symbols in symbols_per_level
            ]
            return (
                cls(identifier, symbols_in_levels),
                buffer.tell()
            )

    def encode(self, value) -> Tuple[int, int]:
        """Encode value into symbol."""
        return self.encode_dict[value]

    def encode_into_bits(self, value) -> Bits:
        symobl, length = self.encode(value)
        return Bits(uint=symobl, length=length)

    def decode(self, stream: Stream) -> int:
        """Decode stream using Huffman table.

        Parameters
        ----------
        stream: Stream
            Byte stream to decode.

        Returns
        ----------
        int
            Decoded value from stream.

        """
        symbol = stream.read()
        length = 1
        while (symbol, length) not in self.decode_dict.keys():
            symbol = 2*symbol + stream.read()
            length += 1
        print(f"found {symbol} {self.decode_dict[(symbol, length)]}")
        return self.decode_dict[(symbol, length)]
        node = self._root
        # Search table until leaf is found
        while not isinstance(node, HuffmanLeaf):
            bit = stream.read()
            try:
                node = node.get(bit)
            except IndexError:
                raise ValueError(
                    f"error when reading bit {bit} at"
                    f"position{stream.pos}"
                )
        return node.value

    def decode_from_bits(self, bits: Bits) -> int:
        """Decode bits using Huffman table.

        Parameters
        ----------
        bits: Bits
            Bits to decode.

        Returns
        ----------
        int
            Decoded value from bits.
        """
        node = self._root
        stream = ConstBitStream(bits)
        # Search table until leaf is found
        while not isinstance(node, HuffmanLeaf):
            bit = stream.read('uint:1')
            try:
                node = node._nodes[bit]
            except IndexError:
                raise ValueError(
                    f"error when reading bit {bit} at"
                    f"position{stream.pos}"
                )
        return node.value
