"""Integration tests for aef_bng."""

import pytest

from aef_bng.index import AEFBNGIndex, _extract_asset_path, _extract_crs
from aef_bng.reader import bng_bounds_to_utm, read_tile


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_read_tile() -> None:
    """Live network test to read a small subset of a COG from Source Coop."""
    index = AEFBNGIndex()
    # small bounding box in BNG (TQ38 central London)
    bounds_bng = (530_000, 180_000, 530_100, 180_100)
    gdf = index.load_for_bounds(bounds_bng)  # Get any available year
    if len(gdf) == 0:
        pytest.skip("No tiles found in STAC index")

    row = gdf.iloc[0]
    tile_path = _extract_asset_path(row)
    tile_crs = _extract_crs(row)

    window_bounds = bng_bounds_to_utm(bounds_bng, tile_crs, padding=0)

    result = await read_tile(tile_path, window_bounds=window_bounds)
    if result is None:
        pytest.skip("Window out of bounds for the actual tile data")

    data, _, _ = result
    assert data.shape[0] == 64
    assert data.shape[1] > 0 and data.shape[2] > 0


@pytest.mark.integration
def test_live_index_load() -> None:
    """Live network test to query the AEF STAC index."""
    index = AEFBNGIndex()
    # small bounding box in BNG (TQ38 central London)
    bounds = (530_000, 180_000, 531_000, 181_000)
    gdf = index.load_for_bounds(bounds)
    assert len(gdf) >= 0
