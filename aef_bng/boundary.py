"""Boundary loading and chunk classification for spatial ingest filtering.

Restricts ingestion to pixels intersecting a boundary geometry (e.g. the ONS BFC
coastline-clipped countries for Great Britain) using a two-level filter:

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
from typing import TYPE_CHECKING, cast

import shapely
from shapely.geometry import MultiPolygon, box

from aef_bng.constants import BNG_CRS

if TYPE_CHECKING:
    from shapely.geometry import Polygon
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
        query: Optional pandas-style attribute filter applied before combining,
            e.g. ``"CTRY25NM in ['England', 'Scotland', 'Wales']"``.
        buffer_m: Optional outward buffer in metres applied after combining
            (buffering dissolves - GEOS buffer output is a valid union).

    Returns:
        A (Multi)Polygon in EPSG:27700 holding the valid polygonal parts of
        every matching feature. Features are combined WITHOUT dissolving: a
        unary union nodes the entire boundary linework, which is
        memory-prohibitive for full-resolution coastlines and unnecessary -
        :func:`classify_chunk` treats the collection exactly as its union.

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

    geoms = list(gdf.geometry.values)
    n_features = len(geoms)
    del gdf  # only the geometries are needed from here; release the rest

    # Validate feature-by-feature, releasing each source geometry as it is
    # consumed, so peak memory is the parts plus one in-flight feature.
    parts: list[Polygon] = []
    while geoms:
        parts.extend(_polygon_parts(shapely.make_valid(geoms.pop())))
    if not parts:
        msg = f"Boundary geometry is empty after processing: {path}"
        raise ValueError(msg)

    geometry: BaseGeometry = parts[0] if len(parts) == 1 else MultiPolygon(parts)
    if buffer_m > 0:
        geometry = geometry.buffer(buffer_m)

    logger.info(
        "Boundary loaded: %d features, %.0f km2 (query=%r, buffer=%.0fm)",
        n_features,
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
        chunk. Only polygonal parts of the clip are kept: edge-touch cases
        produce line/point parts with no interior for any pixel to overlap, and
        downstream rasterisation requires polygons.

    Note:
        With an undissolved multi-feature boundary (see :func:`load_boundary`),
        a chunk lying across two touching features classifies PARTIAL rather
        than FULL - its clipped mask covers the whole chunk, so the output is
        identical at the cost of rasterising a few internal-border chunks.
    """
    cell = box(*bounds_bng)
    if not shapely.intersects(boundary, cell):
        return Coverage.OUTSIDE, None
    if shapely.contains_properly(boundary, cell):
        return Coverage.FULL, None

    # clip_by_rect is a rectangle clip, far cheaper than a general overlay
    # against the full boundary; it may emit invalid output, so make_valid
    # runs on the (small) clipped result rather than the whole boundary.
    clipped_raw = shapely.clip_by_rect(boundary, *bounds_bng)
    clipped = _polygonal(shapely.make_valid(clipped_raw))
    if clipped is None:
        return Coverage.OUTSIDE, None
    return Coverage.PARTIAL, shapely.to_wkb(clipped)


def _polygonal(geometry: BaseGeometry) -> BaseGeometry | None:
    """Extract the polygonal component of a geometry, or None if it has none.

    Rectangle clips return a GeometryCollection when a polygon overlaps the
    rectangle in an area AND touches along an edge elsewhere; line/point
    members must be dropped before rasterisation.
    """
    if geometry.geom_type in ("Polygon", "MultiPolygon"):
        return None if geometry.is_empty else geometry
    parts = _polygon_parts(geometry)
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else shapely.union_all(parts)


def _polygon_parts(geometry: BaseGeometry) -> list[Polygon]:
    """Recursively collect the non-empty Polygon parts of a geometry."""
    if geometry.geom_type == "Polygon":
        return [] if geometry.is_empty else [cast("Polygon", geometry)]
    if geometry.geom_type in ("MultiPolygon", "GeometryCollection"):
        parts: list[Polygon] = []
        for part in shapely.get_parts(geometry):
            parts.extend(_polygon_parts(part))
        return parts
    return []
