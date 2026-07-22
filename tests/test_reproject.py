"""Tests for aef_bng.reproject."""

from __future__ import annotations

import numpy as np
import pytest
from affine import Affine

from aef_bng.constants import AEF_NODATA, AEF_NUM_BANDS
from aef_bng.reproject import merge_tiles, reproject_tile_to_bng


@pytest.mark.unit
class TestReprojectTileToBng:
    """Tests for single-tile reprojection."""

    def test_output_shape_and_dtype(self) -> None:
        """Output must be (64, H, W) int8."""
        src = np.zeros((AEF_NUM_BANDS, 100, 100), dtype=np.int8)
        src_tf = Affine(10, 0, 500000, 0, -10, 5800000)
        dst_tf = Affine(10, 0, 530000, 0, -10, 190000)

        result = reproject_tile_to_bng(src, src_tf, "EPSG:32630", dst_tf, (1000, 1000))
        assert result.shape == (AEF_NUM_BANDS, 1000, 1000)
        assert result.dtype == np.int8

    def test_nodata_fill(self) -> None:
        """Unmapped pixels should be AEF_NODATA."""
        # Source is entirely nodata
        src = np.full((AEF_NUM_BANDS, 10, 10), AEF_NODATA, dtype=np.int8)
        src_tf = Affine(10, 0, 500000, 0, -10, 5800000)
        dst_tf = Affine(10, 0, 530000, 0, -10, 190000)

        result = reproject_tile_to_bng(src, src_tf, "EPSG:32630", dst_tf, (100, 100))
        assert np.all(result == AEF_NODATA)


@pytest.mark.unit
class TestMergeTiles:
    """Tests for multi-tile first-valid merging."""

    def test_single_tile_identity(self) -> None:
        """Single tile merge returns the tile unchanged."""
        arr = np.ones((AEF_NUM_BANDS, 10, 10), dtype=np.int8)
        result = merge_tiles([arr])
        np.testing.assert_array_equal(result, arr)

    def test_first_valid_strategy(self) -> None:
        """Second tile fills nodata gaps in first tile."""
        a = np.full((AEF_NUM_BANDS, 10, 10), AEF_NODATA, dtype=np.int8)
        b = np.ones((AEF_NUM_BANDS, 10, 10), dtype=np.int8) * 42

        # First tile has some data, some nodata
        a[:, :5, :] = 7  # left half has data

        result = merge_tiles([a, b])
        # Left half should keep first tile's values
        np.testing.assert_array_equal(result[:, :5, :], 7)
        # Right half should be filled from second tile
        np.testing.assert_array_equal(result[:, 5:, :], 42)

    def test_empty_list_raises(self) -> None:
        """Empty array list should raise ValueError."""
        with pytest.raises(ValueError, match="No arrays"):
            merge_tiles([])
