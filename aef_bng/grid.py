"""BNG output grid definition and chunk enumeration.

Enumerates 10km BNG chunks covering the requested bounds, computing the affine transform, pixel
shape, and WGS84 bounds for each chunk.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from affine import Affine
from osbng import BNGReference, bbox_to_bng
from osbng.indexing import bng_to_bbox
from pyproj import Transformer

from aef_bng.constants import BNG_BOUNDS, BNG_CRS, BNG_RESOLUTION, CHUNK_SIZE


@dataclass(frozen=True)
class ChunkSpec:
    """Specification for a single 10km BNG processing chunk.

    Attributes:
        bng_10km_ref: 10km BNG grid reference string (e.g. "SU14").
        bounds_bng: BNG bounding box (minx, miny, maxx, maxy) in EPSG:27700.
        bounds_wgs84: Same bounds transformed to WGS84 (EPSG:4326).
        shape: Pixel dimensions (rows, cols) — always (1000, 1000) for 10km at 10m.
        transform: Affine transform for this chunk's raster grid.
    """

    bng_10km_ref: str
    bounds_bng: tuple[int, int, int, int]
    bounds_wgs84: tuple[float, float, float, float]
    shape: tuple[int, int] = (1000, 1000)
    transform: Affine | None = None

    def __post_init__(self) -> None:
        """Compute the affine transform from bounds if not explicitly provided."""
        if self.transform is None:
            minx, _miny, _maxx, maxy = self.bounds_bng
            computed = Affine(BNG_RESOLUTION, 0, minx, 0, -BNG_RESOLUTION, maxy)
            object.__setattr__(self, "transform", computed)


class BNGOutputGrid:
    """Enumerates 10km BNG grid chunks within given bounds.

    Args:
        bounds: BNG bounding box (minx, miny, maxx, maxy) in EPSG:27700.
        chunk_size: Chunk size in metres (default 10,000 for 10km).
    """

    def __init__(
        self,
        bounds: tuple[int, int, int, int] = BNG_BOUNDS,
        chunk_size: int = CHUNK_SIZE,
    ) -> None:
        self.bounds = bounds
        self.chunk_size = chunk_size
        self._transformer = Transformer.from_crs(BNG_CRS, "EPSG:4326", always_xy=True)

    def _bng_bounds_to_wgs84(
        self, bounds_bng: tuple[int, int, int, int]
    ) -> tuple[float, float, float, float]:
        """Transform BNG bounds to WGS84 bounds.

        Transforms all four corners and takes the envelope to handle
        the non-axis-aligned nature of the BNG→WGS84 transformation.

        Args:
            bounds_bng: (minx, miny, maxx, maxy) in EPSG:27700.

        Returns:
            (minx, miny, maxx, maxy) in EPSG:4326.
        """
        minx, miny, maxx, maxy = bounds_bng
        corners_e = np.array([minx, maxx, maxx, minx])
        corners_n = np.array([miny, miny, maxy, maxy])
        lons, lats = self._transformer.transform(corners_e, corners_n)
        return (float(lons.min()), float(lats.min()), float(lons.max()), float(lats.max()))

    def enumerate_chunks(self) -> list[ChunkSpec]:
        """Enumerate all 10km BNG chunks within the configured bounds.

        Uses ``osbng.bbox_to_bng`` to generate valid 10km grid references,
        then builds a ``ChunkSpec`` for each with aligned affine transforms.

        Returns:
            List of ChunkSpec objects covering the bounds.
        """
        resolution = self.chunk_size
        resolution_label = f"{resolution // 1000}km"

        bng_refs: list[BNGReference] = bbox_to_bng(*self.bounds, resolution_label)

        chunks: list[ChunkSpec] = []
        pixels_per_side = resolution // BNG_RESOLUTION

        for ref in bng_refs:
            ref_str = ref.bng_ref_compact
            bounds_bng = bng_to_bbox(ref)
            bounds_wgs84 = self._bng_bounds_to_wgs84(bounds_bng)

            minx, _miny, _maxx, maxy = bounds_bng
            transform = Affine(BNG_RESOLUTION, 0, minx, 0, -BNG_RESOLUTION, maxy)

            chunks.append(
                ChunkSpec(
                    bng_10km_ref=ref_str,
                    bounds_bng=bounds_bng,
                    bounds_wgs84=bounds_wgs84,
                    shape=(pixels_per_side, pixels_per_side),
                    transform=transform,
                )
            )

        return chunks
