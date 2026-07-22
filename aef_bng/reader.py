"""COG reading via obstore + async-geotiff.

Reads AEF Cloud Optimised GeoTIFF tiles from Source Cooperative, optionally with windowed reads to
limit data transfer.

Note on AEF COG orientation
----------------------------
AEF COGs are "bottom-up": the origin is the bottom-left corner, the y-resolution is positive, and
image blocks are ordered from bottom-left to top-right. This is the inverse of a standard
("top-down") COG where the origin is the top-left and y-resolution is negative.

Source Cooperative's README advises using the companion ``.vrt`` files to correct this on-the-fly
for software that assumes standard ordering. This pipeline does **not** need the VRTs because:

1. ``async_geotiff`` reads the affine transform from the TIFF tags and returns it alongside the
   data, correctly reflecting the positive y-scale.
2. ``rasterio.warp.reproject`` uses the affine transform to map pixel coordinates to world
   coordinates, so it handles positive y-scale natively.
3. ``_bounds_to_window`` computes the pixel window by transforming all four geographic corners
   (not just two) and taking the actual min/max of the resulting row/column values, which is
   correct for both positive and negative y-scale transforms.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from async_geotiff import GeoTIFF, Window
from obstore.store import S3Store
from pyproj import Transformer

from aef_bng.constants import AEF_BUCKET, AEF_REGION, BNG_CRS

if TYPE_CHECKING:
    from affine import Affine

logger = logging.getLogger(__name__)

# cache transformers keyed by (src_crs, dst_crs)
_transformer_cache: dict[tuple[str, str], Transformer] = {}


def _bounds_to_window(
    bounds: tuple[float, float, float, float],
    transform: Affine,
    img_width: int,
    img_height: int,
) -> Window | None:
    """Convert geographic bounds to a pixel Window.

    Handles both positive and negative y-scale transforms by computing pixel
    coordinates for all four corners and taking the actual min/max. AEF COGs
    have a positive y-scale (bottom-up origin), so a two-corner approach would
    produce a negative window height; transforming all four corners is required.

    Args:
        bounds: (minx, miny, maxx, maxy) in the tile's native CRS.
        transform: Affine transform of the source tile.
        img_width: Image width in pixels.
        img_height: Image height in pixels.

    Returns:
        Window for the intersection of bounds with the image extent,
        or None if the bounds don't overlap the tile.
    """
    inv = ~transform
    minx, miny, maxx, maxy = bounds

    corners = [
        inv * (minx, miny),
        inv * (maxx, miny),
        inv * (maxx, maxy),
        inv * (minx, maxy),
    ]
    cols = [c for c, _ in corners]  # type: ignore[misc]
    rows = [r for _, r in corners]  # type: ignore[misc]

    # clamp to image extent
    col_off = max(0, int(min(cols)))
    row_off = max(0, int(min(rows)))
    col_end = min(img_width, int(np.ceil(max(cols))))
    row_end = min(img_height, int(np.ceil(max(rows))))

    width = col_end - col_off
    height = row_end - row_off

    if width <= 0 or height <= 0:
        return None

    return Window(col_off=col_off, row_off=row_off, width=width, height=height)


def _strip_s3_prefix(path: str) -> str:
    """Strip the s3://bucket/ prefix from a path, returning just the key.

    The AEF index stores full s3:// URIs, but obstore expects just the key
    relative to the bucket root.

    Args:
        path: Full s3:// URI or bare key.

    Returns:
        Object key without bucket prefix.
    """
    if path.startswith("s3://"):
        parts = path.split("/", 3)
        if len(parts) >= 4:
            return parts[3]
        return ""
    return path


def bng_bounds_to_utm(
    bounds_bng: tuple[int, int, int, int],
    tile_crs: str,
    padding: int = 500,
) -> tuple[float, float, float, float]:
    """Transform BNG bounds to a tile's UTM CRS with padding.

    Transforms all four corners and takes the envelope, then adds
    padding to ensure the reprojection window fully covers the chunk.

    Args:
        bounds_bng: (minx, miny, maxx, maxy) in EPSG:27700.
        tile_crs: CRS of the tile (e.g. "EPSG:32630").
        padding: Extra metres to add on each side (default 500m — generous
            to account for CRS distortion at zone edges).

    Returns:
        (minx, miny, maxx, maxy) in the tile's CRS.
    """
    key = (BNG_CRS, tile_crs)
    if key not in _transformer_cache:
        _transformer_cache[key] = Transformer.from_crs(BNG_CRS, tile_crs, always_xy=True)
    transformer = _transformer_cache[key]

    minx, miny, maxx, maxy = bounds_bng
    corners_x = [minx, maxx, maxx, minx]
    corners_y = [miny, miny, maxy, maxy]
    xs, ys = transformer.transform(corners_x, corners_y)

    return (
        min(xs) - padding,
        min(ys) - padding,
        max(xs) + padding,
        max(ys) + padding,
    )


async def read_tile(
    tile_path: str,
    window_bounds: tuple[float, float, float, float] | None = None,
) -> tuple[np.ndarray, Affine, str] | None:
    """Read an AEF COG tile, optionally with a geographic window.

    Args:
        tile_path: S3 path to the COG tile (may include s3://bucket/ prefix).
        window_bounds: Optional (minx, miny, maxx, maxy) in the tile's native
            CRS to read a subset of the tile.

    Returns:
        Tuple of (data, transform, crs) where:
        - data is (64, H, W) int8 numpy array
        - transform is the Affine transform for the returned data
        - crs is the CRS string (e.g. "EPSG:32630")
        Returns None if window_bounds don't overlap the tile.
    """
    key = _strip_s3_prefix(tile_path)
    store = S3Store(bucket=AEF_BUCKET, region=AEF_REGION, skip_signature=True)

    logger.debug("Opening COG tile: %s", key)
    geotiff = await GeoTIFF.open(path=key, store=store)

    crs_str = str(geotiff.crs)

    if window_bounds:
        window = _bounds_to_window(window_bounds, geotiff.transform, geotiff.width, geotiff.height)
        if window is None:
            logger.debug("Window does not overlap tile %s, skipping", key)
            return None
        array = await geotiff.read(window=window)
    else:
        array = await geotiff.read()

    return array.data, array.transform, crs_str
