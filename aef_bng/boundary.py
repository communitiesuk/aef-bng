"""Boundary loading and chunk classification for spatial ingest filtering.

Restricts ingestion to pixels intersecting a boundary geometry (e.g. the ONS BFE
"Extent of the Realm" countries for Great Britain) using a two-level filter:

1. Chunk level (driver): every 10km chunk is classified against the boundary as
   OUTSIDE (dropped before any S3 read), FULL (every pixel is inside - no per-pixel
   work needed), or PARTIAL (carries the boundary clipped to the chunk as WKB).
2. Pixel level (executor): PARTIAL chunks rasterise their clipped geometry onto the
   chunk grid and AND it with the nodata mask (see ``extract._compute_pixel_data``).

A 10m cell is kept if it intersects the boundary at all ("any overlap" semantics,
via ``all_touched`` rasterisation in ``extract``).
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING

import shapely
from shapely.geometry import box

from aef_bng.constants import BNG_CRS

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry

logger = logging.getLogger(__name__)


class Coverage(Enum):
    """How much of a chunk lies within the boundary geometry."""

    OUTSIDE = "outside"
    PARTIAL = "partial"
    FULL = "full"


def load_boundary(
    path: str,
    query: str | None = None,
    buffer_m: float = 0.0,
) -> BaseGeometry:
    """Load a boundary geometry from a vector file for ingest filtering.

    Any boundary file works; GeoParquet is recommended (compact, fast, typed CRS).
    ``.parquet``/``.geoparquet`` paths are read natively rather than through GDAL,
    whose wheels often lack the Parquet driver.

    Args:
        path: Vector source - GeoParquet (recommended) or any OGR-readable format
            (GeoJSON, GPKG, ...). Must declare a CRS.
        query: Optional pandas-style attribute filter applied before dissolving,
            e.g. ``"CTRY25NM in ['England', 'Scotland', 'Wales']"``.
        buffer_m: Optional outward buffer in metres applied after dissolving.

    Returns:
        A single valid (Multi)Polygon in EPSG:27700.

    Raises:
        ValueError: If the file has no CRS, the query matches nothing, or the
            resulting geometry is empty.
    """
    import geopandas as gpd

    if path.endswith((".parquet", ".geoparquet")):
        gdf = gpd.read_parquet(path)
    else:
        gdf = gpd.read_file(path)
    if gdf.crs is None:
        msg = f"Boundary file has no CRS: {path}"
        raise ValueError(msg)
    if query:
        gdf = gdf.query(query)
        if gdf.empty:
            msg = f"Boundary query matched no features: {query!r}"
            raise ValueError(msg)
    gdf = gdf.to_crs(BNG_CRS)

    geometry = shapely.union_all(shapely.make_valid(gdf.geometry.values))
    if buffer_m > 0:
        geometry = geometry.buffer(buffer_m)
    if geometry.is_empty:
        msg = f"Boundary geometry is empty after processing: {path}"
        raise ValueError(msg)

    logger.info(
        "Boundary loaded: %d features, %.0f km2 (query=%r, buffer=%.0fm)",
        len(gdf),
        geometry.area / 1e6,
        query,
        buffer_m,
    )
    return geometry


def classify_chunk(
    bounds_bng: tuple[int, int, int, int],
    boundary: BaseGeometry,
) -> tuple[Coverage, bytes | None]:
    """Classify one chunk's bounds against the boundary geometry.

    Call ``shapely.prepare(boundary)`` once before classifying many chunks - the
    prepared predicates make the 9,100-chunk sweep take seconds.

    Args:
        bounds_bng: Chunk bounding box (minx, miny, maxx, maxy) in EPSG:27700.
        boundary: Boundary geometry from :func:`load_boundary`.

    Returns:
        ``(Coverage.FULL, None)``, ``(Coverage.OUTSIDE, None)``, or
        ``(Coverage.PARTIAL, wkb)`` where ``wkb`` is the boundary clipped to the
        chunk. Only polygonal parts of the intersection are kept: edge-touch cases
        produce line/point parts with no interior for any pixel to overlap, and
        downstream rasterisation requires polygons.
    """
    cell = box(*bounds_bng)
    if not shapely.intersects(boundary, cell):
        return Coverage.OUTSIDE, None
    if shapely.contains_properly(boundary, cell):
        return Coverage.FULL, None

    clipped = _polygonal(shapely.intersection(boundary, cell))
    if clipped is None:
        return Coverage.OUTSIDE, None
    return Coverage.PARTIAL, shapely.to_wkb(clipped)


def _polygonal(geometry: BaseGeometry) -> BaseGeometry | None:
    """Extract the polygonal component of a geometry, or None if it has none.

    ``shapely.intersection`` returns a GeometryCollection when two polygons overlap
    in an area AND touch along an edge elsewhere; line/point members must be dropped
    before rasterisation.
    """
    if geometry.geom_type in ("Polygon", "MultiPolygon"):
        return None if geometry.is_empty else geometry
    parts: list[BaseGeometry] = []
    for part in shapely.get_parts(geometry):
        if part.geom_type == "Polygon" and not part.is_empty:
            parts.append(part)
        elif part.geom_type == "MultiPolygon":
            parts.extend(sub for sub in shapely.get_parts(part) if not sub.is_empty)
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else shapely.union_all(parts)
