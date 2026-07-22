"""UTM → BNG reprojection via rasterio.warp.

Handles reprojection of individual tiles and merging of overlapping tiles at UTM zone boundaries
using a first-valid strategy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from rasterio.enums import Resampling
from rasterio.warp import reproject

from aef_bng.constants import AEF_NODATA, AEF_NUM_BANDS, BNG_CRS

if TYPE_CHECKING:
    from affine import Affine

RESAMPLING_METHODS: dict[str, Resampling] = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
}


def reproject_tile_to_bng(
    src_data: np.ndarray,
    src_transform: Affine,
    src_crs: str,
    dst_transform: Affine,
    dst_shape: tuple[int, int],
    resampling: str = "nearest",
) -> np.ndarray:
    """Reproject a single tile from UTM to a BNG chunk grid.

    Args:
        src_data: Source array of shape (64, H, W) int8.
        src_transform: Affine transform of the source tile.
        src_crs: CRS string of the source tile (e.g. "EPSG:32630").
        dst_transform: Affine transform of the destination BNG chunk.
        dst_shape: Pixel dimensions (rows, cols) of the destination.
        resampling: Resampling method name.

    Returns:
        Reprojected array of shape (64, *dst_shape) int8 with -128 for nodata.
    """
    dst = np.full((AEF_NUM_BANDS, *dst_shape), AEF_NODATA, dtype=np.int8)

    resampling_method = RESAMPLING_METHODS.get(resampling, Resampling.nearest)

    reproject(
        source=src_data,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=BNG_CRS,
        resampling=resampling_method,
        src_nodata=AEF_NODATA,
        dst_nodata=AEF_NODATA,
    )

    return dst


def merge_tiles(arrays: list[np.ndarray]) -> np.ndarray:
    """Merge multiple reprojected tile arrays using first-valid strategy.

    At UTM zone boundaries, the same ground area appears in tiles from adjacent UTM zones.
    After reprojection to BNG, both tiles yield     nearly identical values (same satellite data,
    different projection path). Nearest-neighbour resampling from different UTM zones may produce
    values differing by at most 1 int8 unit due to grid alignment differences, which is within noise
    for quantised embeddings.

    The first-valid strategy takes the first non-nodata value for each pixel. Callers must sort the
    input tiles deterministically (e.g. by path) so the merge result is reproducible across runs.

    A pixel is considered nodata only when ALL 64 bands equal -128. Embeddings are atomic vectors,
    partial band replacement would be incorrect.

    Args:
        arrays: List of (64, H, W) int8 arrays, all same shape.
            Must be in deterministic order.

    Returns:
        Merged array of shape (64, H, W) int8.
    """
    if not arrays:
        raise ValueError("No arrays to merge")

    if len(arrays) == 1:
        return arrays[0]

    merged = arrays[0].copy()

    for arr in arrays[1:]:
        nodata_mask = np.all(merged == AEF_NODATA, axis=0)
        merged[:, nodata_mask] = arr[:, nodata_mask]

    return merged
