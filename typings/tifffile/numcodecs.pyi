from numcodecs.abc import Codec
from typing import Any

class Tiff(Codec):
    codec_id: str
    key: Any
    series: Any
    level: Any
    maxworkers: Any
    bigtiff: Any
    byteorder: Any
    imagej: Any
    ome: Any
    photometric: Any
    planarconfig: Any
    volumetric: Any
    tile: Any
    truncate: Any
    rowsperstrip: Any
    compression: Any
    predictor: Any
    subsampling: Any
    metadata: Any
    extratags: Any
    def __init__(self, key: Any | None = ..., series: Any | None = ..., level: Any | None = ..., maxworkers: Any | None = ..., bigtiff: Any | None = ..., byteorder: Any | None = ..., imagej: bool = ..., ome: Any | None = ..., photometric: Any | None = ..., planarconfig: Any | None = ..., volumetric: Any | None = ..., tile: Any | None = ..., truncate: bool = ..., rowsperstrip: Any | None = ..., compression: Any | None = ..., predictor: Any | None = ..., subsampling: Any | None = ..., metadata=..., extratags=...) -> None: ...
    def encode(self, buf): ...
    def decode(self, buf, out: Any | None = ...): ...

def register_codec(cls=..., codec_id: Any | None = ...) -> None: ...
