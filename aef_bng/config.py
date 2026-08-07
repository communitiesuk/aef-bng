"""Configuration for AEF-BNG processing."""

from __future__ import annotations

import math
from dataclasses import dataclass

from aef_bng.constants import BNG_BOUNDS, CHUNK_SIZE


@dataclass
class AEFBNGConfig:
    """Configuration for AEF to BNG processing pipeline.

    Attributes:
        years: List of years to process (e.g. [2020, 2021, 2022]).
        bounds: BNG bounding box (minx, miny, maxx, maxy) in EPSG:27700 metres.
        chunk_size: Size of processing chunks in metres (must align with BNG grid).
        output_path: Path for GeoParquet output directory.
        resampling: Resampling method for reprojection.
        max_workers: Maximum concurrent workers for local processing.
        table_name: Unity Catalog table name (e.g. "my_catalog.schema.aef_bng").
            When set, Spark writes to this table instead of output_path.
        boundary_path: Optional OGR-readable vector file (GeoJSON, GPKG, ...) used to
            spatially filter ingestion: only pixels whose 10m cell intersects the
            boundary are kept. None (default) disables filtering entirely.
        boundary_query: Optional pandas-style attribute filter applied to the boundary
            file, e.g. ``"CTRY25NM in ['England', 'Scotland', 'Wales']"``.
        boundary_buffer_m: Optional outward buffer in metres applied to the dissolved
            boundary geometry.
    """

    years: list[int]
    bounds: tuple[int, int, int, int] = BNG_BOUNDS
    chunk_size: int = CHUNK_SIZE
    output_path: str = "./aef_bng_output"
    resampling: str = "nearest"
    max_workers: int = 4
    table_name: str | None = None
    boundary_path: str | None = None
    boundary_query: str | None = None
    boundary_buffer_m: float = 0.0

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not self.years:
            raise ValueError("At least one year must be specified")
        if self.chunk_size <= 0 or self.chunk_size % 1000 != 0:
            raise ValueError("chunk_size must be a positive multiple of 1000")
        if len(self.bounds) != 4:
            raise ValueError("bounds must be a 4-tuple (minx, miny, maxx, maxy)")
        minx, miny, maxx, maxy = self.bounds
        if minx >= maxx or miny >= maxy:
            raise ValueError("Invalid bounds: min must be less than max")
        if not math.isfinite(self.boundary_buffer_m) or self.boundary_buffer_m < 0:
            raise ValueError("boundary_buffer_m must be finite and >= 0")
        if self.boundary_path is None and (self.boundary_query or self.boundary_buffer_m > 0):
            raise ValueError("boundary_query/boundary_buffer_m require boundary_path")
