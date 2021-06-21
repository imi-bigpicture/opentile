MARER_MAPPINGS = {
    0xFFD8: "Start of Image",
    0xFFE0: "Application Default Header",
    0xFFDB: "Quantization Table",
    0xFFC0: "Start of Frame",
    0xFFC4: "Huffman Table",
    0xFFDA: "Start of Scan",
    0xFFD9: "End of Image",
    0xFFFE: "Comment",
    0xFFDD: "Define Restart Interval"
}

TAGS = {
    'start of image': 0xFFD8,
    'application default header': 0xFFE0,
    'quantization table': 0xFFDB,
    'start of frame': 0xFFC0,
    'huffman table': 0xFFC4,
    'start of scan': 0xFFDA,
    'end of image': 0xFFD9,
    'restart interval': 0xFFDD
}

BYTE_TAG = bytes([0xFF])
BYTE_STUFFING = bytes([0x00])