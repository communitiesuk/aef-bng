"""Vectorised BNG reference generation and Arrow table extraction.

Converts reprojected raster arrays into tabular rows with BNG references.

Performance-critical: ~1M pixels per 10km chunk.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa

from aef_bng.constants import AEF_BAND_NAMES, AEF_NODATA, AEF_NUM_BANDS, BNG_RESOLUTION

if TYPE_CHECKING:
    from aef_bng.grid import ChunkSpec

# WKB little-endian Polygon header: byte_order=1, type=3 (Polygon), num_rings=1, num_points=5
_WKB_BOX_HEADER = np.frombuffer(
    b"\x01\x03\x00\x00\x00\x01\x00\x00\x00\x05\x00\x00\x00",
    dtype=np.uint8,
)

PREFIXES: list[list[str]] = [
    ["SV", "SW", "SX", "SY", "SZ", "TV", "TW"],
    ["SQ", "SR", "SS", "ST", "SU", "TQ", "TR"],
    ["SL", "SM", "SN", "SO", "SP", "TL", "TM"],
    ["SF", "SG", "SH", "SJ", "SK", "TF", "TG"],
    ["SA", "SB", "SC", "SD", "SE", "TA", "TB"],
    ["NV", "NW", "NX", "NY", "NZ", "OV", "OW"],
    ["NQ", "NR", "NS", "NT", "NU", "OQ", "OR"],
    ["NL", "NM", "NN", "NO", "NP", "OL", "OM"],
    ["NF", "NG", "NH", "NJ", "NK", "OF", "OG"],
    ["NA", "NB", "NC", "ND", "NE", "OA", "OB"],
    ["HV", "HW", "HX", "HY", "HZ", "JV", "JW"],
    ["HQ", "HR", "HS", "HT", "HU", "JQ", "JR"],
    ["HL", "HM", "HN", "HO", "HP", "JL", "JM"],
]


def _get_prefix(easting: int, northing: int) -> str:
    """Get the 100km BNG prefix for a coordinate.

    Args:
        easting: Easting coordinate in metres.
        northing: Northing coordinate in metres.

    Returns:
        Two-letter 100km grid square prefix.
    """
    return PREFIXES[northing // 100_000][easting // 100_000]


def _parent_refs(bng_refs: list[str]) -> tuple[list[str], list[str]]:
    """Derive 10km and 1km parent grid references from 10m BNG references.

    A 10m ref lays out its axes as ``<letters><eeee><nnnn>`` (e.g.
    ``TQ12345678``), so each parent takes the leading digit(s) of both axes:
    ``TQ15`` (10km) and ``TQ1256`` (1km). Parents are therefore digit-picks,
    not string prefixes, of the child ref.

    Args:
        bng_refs: 10-character 10m BNG references.

    Returns:
        Tuple of (grid_10km_ref, grid_1km_ref) lists.
    """
    grid_10km = [ref[:3] + ref[6] for ref in bng_refs]
    grid_1km = [ref[:4] + ref[6:8] for ref in bng_refs]
    return grid_10km, grid_1km


def _boundary_mask(mask_wkb: bytes, chunk: ChunkSpec) -> np.ndarray:
    """Rasterise a clipped boundary geometry onto the chunk's pixel grid.

    ``all_touched=True`` keeps every 10m cell the boundary overlaps at all
    ("any overlap" semantics, matching ``ST_Intersects`` filtering downstream).

    Args:
        mask_wkb: WKB of the boundary clipped to this chunk (EPSG:27700 polygons).
        chunk: The BNG chunk specification.

    Returns:
        Boolean array of ``chunk.shape``; True where the pixel is inside/touching.
    """
    import shapely.wkb
    from rasterio.features import geometry_mask

    geometry = shapely.wkb.loads(mask_wkb)
    return geometry_mask(
        [geometry],
        out_shape=chunk.shape,
        transform=chunk.transform,
        invert=True,
        all_touched=True,
    )


def _compute_pixel_data(
    data: np.ndarray,
    chunk: ChunkSpec,
    mask_wkb: bytes | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray] | None:
    """Compute valid pixel coordinates, BNG references, and embeddings.

    Args:
        data: Reprojected array of shape (bands, rows, cols) int8.
        chunk: The BNG chunk specification.
        mask_wkb: Optional boundary geometry (WKB, EPSG:27700) restricting output
            to intersecting pixels. None keeps every non-nodata pixel.

    Returns:
        Tuple of (valid_eastings, valid_northings, valid_rows, valid_cols,
        bng_refs, embeddings) or None if no valid pixels.
    """
    rows, cols = chunk.shape

    valid_mask = ~np.all(data == AEF_NODATA, axis=0)
    if mask_wkb is not None:
        valid_mask &= _boundary_mask(mask_wkb, chunk)
    n_valid = int(valid_mask.sum())

    if n_valid == 0:
        return None

    origin_e = chunk.bounds_bng[0]
    origin_n_top = chunk.bounds_bng[3]

    col_idx, row_idx = np.meshgrid(np.arange(cols), np.arange(rows))
    eastings = origin_e + col_idx * BNG_RESOLUTION
    northings = origin_n_top - (row_idx + 1) * BNG_RESOLUTION

    valid_rows, valid_cols = np.where(valid_mask)
    valid_eastings = eastings[valid_rows, valid_cols]
    valid_northings = northings[valid_rows, valid_cols]

    e_bins = (valid_eastings % 100_000) // BNG_RESOLUTION
    n_bins = (valid_northings % 100_000) // BNG_RESOLUTION

    prefix_x = valid_eastings // 100_000
    prefix_y = valid_northings // 100_000

    unique_px = np.unique(prefix_x)
    unique_py = np.unique(prefix_y)

    if len(unique_px) == 1 and len(unique_py) == 1:
        prefix = PREFIXES[int(unique_py[0])][int(unique_px[0])]
        bng_refs = [
            f"{prefix}{int(e):04d}{int(n):04d}" for e, n in zip(e_bins, n_bins, strict=True)
        ]
    else:
        bng_refs = [
            f"{PREFIXES[int(py)][int(px)]}{int(e):04d}{int(n):04d}"
            for px, py, e, n in zip(prefix_x, prefix_y, e_bins, n_bins, strict=True)
        ]

    embeddings = data[:, valid_rows, valid_cols].T

    return valid_eastings, valid_northings, valid_rows, valid_cols, bng_refs, embeddings


def _wkb_boxes(eastings: np.ndarray, northings: np.ndarray) -> pa.Array:
    """Build WKB Polygon bytes for n 10m x 10m BNG cell boxes.

    Constructed entirely in numpy - no shapely, no per-row Python loops.
    Each polygon is a fixed 93-byte little-endian WKB:
    - 13-byte header (byte order, geometry type, ring count, point count)
    - 5 x 2 x float64 coordinate pairs closing the rectangle

    Args:
        eastings: Lower-left easting of each cell.
        northings: Lower-left northing of each cell.

    Returns:
        PyArrow binary array of n WKB polygons.
    """
    n = len(eastings)
    e = eastings.astype(np.float64)
    n_ = northings.astype(np.float64)
    e1 = e + float(BNG_RESOLUTION)
    n1 = n_ + float(BNG_RESOLUTION)

    coords = np.empty((n, 10), dtype=np.float64)
    coords[:, 0] = e
    coords[:, 1] = n_
    coords[:, 2] = e1
    coords[:, 3] = n_
    coords[:, 4] = e1
    coords[:, 5] = n1
    coords[:, 6] = e
    coords[:, 7] = n1
    coords[:, 8] = e
    coords[:, 9] = n_

    wkb = np.empty((n, 93), dtype=np.uint8)
    wkb[:, :13] = _WKB_BOX_HEADER
    wkb[:, 13:] = coords.view(np.uint8).reshape(n, 80)

    flat = wkb.tobytes()
    offsets = np.arange(0, (n + 1) * 93, 93, dtype=np.int32)
    return pa.Array.from_buffers(
        pa.binary(),
        n,
        [None, pa.py_buffer(offsets.tobytes()), pa.py_buffer(flat)],
    )


def extract_pixels(
    data: np.ndarray,
    chunk: ChunkSpec,
    year: int,
    mask_wkb: bytes | None = None,
) -> pa.Table:
    """Extract valid pixels from a reprojected chunk as an Arrow table.

    Each embedding band is a separate int8 column (A00..A63).
    Includes easting/northing for GeoParquet geometry construction.

    Args:
        data: Reprojected array of shape (64, rows, cols) int8.
        chunk: The BNG chunk specification.
        year: Year of the AEF embeddings.
        mask_wkb: Optional boundary geometry (WKB) restricting output pixels.

    Returns:
        Arrow table with columns: bng_ref, year, grid_10km_ref, grid_1km_ref,
        A00..A63, easting, northing.
    """
    result = _compute_pixel_data(data, chunk, mask_wkb)
    if result is None:
        return _empty_table()

    valid_eastings, valid_northings, _, _, bng_refs, embeddings = result
    n_valid = len(bng_refs)

    grid_10km_refs, grid_1km_refs = _parent_refs(bng_refs)

    columns: dict[str, pa.Array] = {
        "bng_ref": pa.array(bng_refs, type=pa.string()),
        "year": pa.array([year] * n_valid, type=pa.int16()),
        "grid_10km_ref": pa.array(grid_10km_refs, type=pa.string()),
        "grid_1km_ref": pa.array(grid_1km_refs, type=pa.string()),
    }

    for i in range(AEF_NUM_BANDS):
        columns[AEF_BAND_NAMES[i]] = pa.array(embeddings[:, i], type=pa.int8())

    columns["easting"] = pa.array(valid_eastings, type=pa.int32())
    columns["northing"] = pa.array(valid_northings, type=pa.int32())

    return pa.table(columns)


def extract_pixels_spark(
    data: np.ndarray,
    chunk: ChunkSpec,
    year: int,
    mask_wkb: bytes | None = None,
) -> pa.Table:
    """Extract valid pixels for the Spark/Unity Catalog path.

    Each embedding band is a separate int8 column (A00..A63). A
    ``geometry_wkb`` binary column is always included, containing standard
    WKB polygons for each 10m cell (EPSG:27700).

    Args:
        data: Reprojected array of shape (64, rows, cols) int8.
        chunk: The BNG chunk specification.
        year: Year of the AEF embeddings.
        mask_wkb: Optional boundary geometry (WKB) restricting output pixels.

    Returns:
        Arrow table with columns: bng_ref, year, grid_10km_ref, grid_1km_ref,
        A00..A63, geometry_wkb.
    """
    result = _compute_pixel_data(data, chunk, mask_wkb)
    if result is None:
        return _empty_table_spark()

    valid_eastings, valid_northings, _, _, bng_refs, embeddings = result
    n_valid = len(bng_refs)

    grid_10km_refs, grid_1km_refs = _parent_refs(bng_refs)

    columns: dict[str, pa.Array] = {
        "bng_ref": pa.array(bng_refs, type=pa.string()),
        "year": pa.array([year] * n_valid, type=pa.int16()),
        "grid_10km_ref": pa.array(grid_10km_refs, type=pa.string()),
        "grid_1km_ref": pa.array(grid_1km_refs, type=pa.string()),
    }

    for i in range(AEF_NUM_BANDS):
        columns[AEF_BAND_NAMES[i]] = pa.array(embeddings[:, i], type=pa.int8())

    columns["geometry_wkb"] = _wkb_boxes(valid_eastings, valid_northings)

    return pa.table(columns)


def _empty_table() -> pa.Table:
    """Return an empty Arrow table with the local output schema."""
    columns: dict[str, pa.Array] = {
        "bng_ref": pa.array([], type=pa.string()),
        "year": pa.array([], type=pa.int16()),
        "grid_10km_ref": pa.array([], type=pa.string()),
        "grid_1km_ref": pa.array([], type=pa.string()),
    }
    for name in AEF_BAND_NAMES:
        columns[name] = pa.array([], type=pa.int8())
    columns["easting"] = pa.array([], type=pa.int32())
    columns["northing"] = pa.array([], type=pa.int32())
    return pa.table(columns)


def _empty_table_spark() -> pa.Table:
    """Return an empty Arrow table with the Spark output schema."""
    columns: dict[str, pa.Array] = {
        "bng_ref": pa.array([], type=pa.string()),
        "year": pa.array([], type=pa.int16()),
        "grid_10km_ref": pa.array([], type=pa.string()),
        "grid_1km_ref": pa.array([], type=pa.string()),
    }
    for name in AEF_BAND_NAMES:
        columns[name] = pa.array([], type=pa.int8())
    columns["geometry_wkb"] = pa.array([], type=pa.binary())
    return pa.table(columns)
