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

# PREFIXES as ASCII byte pairs, indexable by (100km_row, 100km_col) arrays.
_PREFIX_BYTES = np.array(PREFIXES, dtype="S2").view(np.uint8).reshape(13, 7, 2)
_ASCII_ZERO = 48


def _get_prefix(easting: int, northing: int) -> str:
    """Get the 100km BNG prefix for a coordinate.

    Args:
        easting: Easting coordinate in metres.
        northing: Northing coordinate in metres.

    Returns:
        Two-letter 100km grid square prefix.
    """
    return PREFIXES[northing // 100_000][easting // 100_000]


def _string_array(codes: np.ndarray) -> pa.Array:
    """Zero-copy Arrow string array from an (n, width) ASCII byte matrix."""
    n, width = codes.shape
    offsets = np.arange(0, (n + 1) * width, width, dtype=np.int32)
    return pa.Array.from_buffers(
        pa.string(),
        n,
        [None, pa.py_buffer(offsets), pa.py_buffer(codes)],
    )


def _parent_ref_arrays(ref_codes: np.ndarray) -> tuple[pa.Array, pa.Array]:
    """10km and 1km parent grid references from the 10m ref byte matrix.

    A 10m ref lays out its axes as ``<letters><eeee><nnnn>`` (e.g.
    ``TQ12345678``), so each parent takes the leading digit(s) of both axes:
    ``TQ15`` (10km) and ``TQ1256`` (1km). Parents are therefore digit-picks,
    not string prefixes, of the child ref.

    Args:
        ref_codes: (n, 10) uint8 ASCII matrix of 10m BNG references.

    Returns:
        Tuple of (grid_10km_ref, grid_1km_ref) Arrow string arrays.
    """
    grid_10km = np.ascontiguousarray(ref_codes[:, [0, 1, 2, 6]])
    grid_1km = np.ascontiguousarray(ref_codes[:, [0, 1, 2, 3, 6, 7]])
    return _string_array(grid_10km), _string_array(grid_1km)


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Compute valid pixel coordinates and BNG references.

    BNG references are built as an (n, 10) ASCII byte matrix rather than
    Python strings - a 1M-pixel chunk needs ~10MB this way versus ~70MB of
    Python string objects, and the matrix converts to Arrow without copying.

    Args:
        data: Reprojected array of shape (bands, rows, cols) int8.
        chunk: The BNG chunk specification.
        mask_wkb: Optional boundary geometry (WKB, EPSG:27700) restricting output
            to intersecting pixels. None keeps every non-nodata pixel.

    Returns:
        Tuple of (valid_eastings, valid_northings, valid_rows, valid_cols,
        ref_codes) where ``ref_codes`` is the (n, 10) uint8 ASCII matrix of
        10m refs, or None if no valid pixels.
    """
    valid_mask = ~np.all(data == AEF_NODATA, axis=0)
    if mask_wkb is not None:
        valid_mask &= _boundary_mask(mask_wkb, chunk)
    n_valid = int(valid_mask.sum())

    if n_valid == 0:
        return None

    origin_e = chunk.bounds_bng[0]
    origin_n_top = chunk.bounds_bng[3]

    # Coordinates only for valid pixels - no full-grid meshgrid intermediates.
    valid_rows, valid_cols = np.where(valid_mask)
    valid_eastings = origin_e + valid_cols * BNG_RESOLUTION
    valid_northings = origin_n_top - (valid_rows + 1) * BNG_RESOLUTION

    e_bins = (valid_eastings % 100_000) // BNG_RESOLUTION
    n_bins = (valid_northings % 100_000) // BNG_RESOLUTION

    ref_codes = np.empty((n_valid, 10), dtype=np.uint8)
    ref_codes[:, 0:2] = _PREFIX_BYTES[valid_northings // 100_000, valid_eastings // 100_000]
    ref_codes[:, 2] = _ASCII_ZERO + e_bins // 1000
    ref_codes[:, 3] = _ASCII_ZERO + (e_bins // 100) % 10
    ref_codes[:, 4] = _ASCII_ZERO + (e_bins // 10) % 10
    ref_codes[:, 5] = _ASCII_ZERO + e_bins % 10
    ref_codes[:, 6] = _ASCII_ZERO + n_bins // 1000
    ref_codes[:, 7] = _ASCII_ZERO + (n_bins // 100) % 10
    ref_codes[:, 8] = _ASCII_ZERO + (n_bins // 10) % 10
    ref_codes[:, 9] = _ASCII_ZERO + n_bins % 10

    return valid_eastings, valid_northings, valid_rows, valid_cols, ref_codes


def _key_columns(ref_codes: np.ndarray, year: int, n_valid: int) -> dict[str, pa.Array]:
    """Build the leading key columns (bng_ref, year, parent refs)."""
    grid_10km_refs, grid_1km_refs = _parent_ref_arrays(ref_codes)
    return {
        "bng_ref": _string_array(ref_codes),
        "year": pa.array(np.full(n_valid, year, dtype=np.int16)),
        "grid_10km_ref": grid_10km_refs,
        "grid_1km_ref": grid_1km_refs,
    }


def _add_band_columns(
    columns: dict[str, pa.Array],
    data: np.ndarray,
    valid_rows: np.ndarray,
    valid_cols: np.ndarray,
    n_valid: int,
) -> None:
    """Add the 64 band columns; each gather feeds Arrow without copying."""
    for i in range(AEF_NUM_BANDS):
        band = data[i, valid_rows, valid_cols]
        columns[AEF_BAND_NAMES[i]] = pa.Array.from_buffers(
            pa.int8(), n_valid, [None, pa.py_buffer(band)]
        )


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

    offsets = np.arange(0, (n + 1) * 93, 93, dtype=np.int32)
    return pa.Array.from_buffers(
        pa.binary(),
        n,
        [None, pa.py_buffer(offsets), pa.py_buffer(wkb)],
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

    valid_eastings, valid_northings, valid_rows, valid_cols, ref_codes = result
    n_valid = len(ref_codes)

    columns = _key_columns(ref_codes, year, n_valid)
    _add_band_columns(columns, data, valid_rows, valid_cols, n_valid)

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

    valid_eastings, valid_northings, valid_rows, valid_cols, ref_codes = result
    n_valid = len(ref_codes)

    columns = _key_columns(ref_codes, year, n_valid)
    _add_band_columns(columns, data, valid_rows, valid_cols, n_valid)

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
