"""AEF embeddings on British National Grid."""

from aef_bng.config import AEFBNGConfig
from aef_bng.dequantise import dequantise, dequantise_dataframe, dequantise_table
from aef_bng.grid import BNGOutputGrid, ChunkSpec
from aef_bng.index import AEFBNGIndex
from aef_bng.types import BoundingBox

__all__ = [
    "AEFBNGConfig",
    "AEFBNGIndex",
    "BNGOutputGrid",
    "BoundingBox",
    "ChunkSpec",
    "dequantise",
    "dequantise_dataframe",
    "dequantise_table",
]

__version__ = "0.1.0"
