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


def _compute_pixel_data(
    data: np.ndarray,
    chunk: ChunkSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray] | None:
    """Compute valid pixel coordinates, BNG references, and embeddings.

    Args:
        data: Reprojected array of shape (bands, rows, cols) int8.
        chunk: The BNG chunk specification.

    Returns:
        Tuple of (valid_eastings, valid_northings, valid_rows, valid_cols,
        bng_refs, embeddings) or None if no valid pixels.
    """
    rows, cols = chunk.shape

    valid_mask = ~np.all(data == AEF_NODATA, axis=0)
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

    Constructed entirely in numpy — no shapely, no per-row Python loops.
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


def extract_pixels(data: np.ndarray, chunk: ChunkSpec, year: int) -> pa.Table:
    """Extract valid pixels from a reprojected chunk as an Arrow table.

    Each embedding band is a separate int8 column (A00..A63).
    Includes easting/northing for GeoParquet geometry construction.

    Args:
        data: Reprojected array of shape (64, rows, cols) int8.
        chunk: The BNG chunk specification.
        year: Year of the AEF embeddings.

    Returns:
        Arrow table with columns: bng_ref, year, A00..A63, easting, northing.
    """
    result = _compute_pixel_data(data, chunk)
    if result is None:
        return _empty_table()

    valid_eastings, valid_northings, _, _, bng_refs, embeddings = result
    n_valid = len(bng_refs)

    columns: dict[str, pa.Array] = {
        "bng_ref": pa.array(bng_refs, type=pa.string()),
        "year": pa.array([year] * n_valid, type=pa.int16()),
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
) -> pa.Table:
    """Extract valid pixels for the Spark/Unity Catalog path.

    Each embedding band is a separate int8 column (A00..A63). A
    ``geometry_wkb`` binary column is always included, containing standard
    WKB polygons for each 10m cell (EPSG:27700).

    Args:
        data: Reprojected array of shape (64, rows, cols) int8.
        chunk: The BNG chunk specification.
        year: Year of the AEF embeddings.

    Returns:
        Arrow table with columns: bng_ref, year, A00..A63, geometry_wkb.
    """
    result = _compute_pixel_data(data, chunk)
    if result is None:
        return _empty_table_spark()

    valid_eastings, valid_northings, _, _, bng_refs, embeddings = result
    n_valid = len(bng_refs)

    columns: dict[str, pa.Array] = {
        "bng_ref": pa.array(bng_refs, type=pa.string()),
        "year": pa.array([year] * n_valid, type=pa.int16()),
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
    }
    for name in AEF_BAND_NAMES:
        columns[name] = pa.array([], type=pa.int8())
    columns["geometry_wkb"] = pa.array([], type=pa.binary())
    return pa.table(columns)
