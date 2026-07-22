"""Tests for aef_bng.reader."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from affine import Affine

import aef_bng.reader
from aef_bng.reader import _bounds_to_window, _strip_s3_prefix, bng_bounds_to_utm, read_tile


@pytest.mark.unit
class TestStripS3Prefix:
    """Tests for S3 path prefix stripping."""

    def test_full_s3_uri(self) -> None:
        """Full s3:// URI → bare key."""
        path = "s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/30U/2024_30U_00.tif"
        assert _strip_s3_prefix(path) == "tge-labs/aef/v1/annual/30U/2024_30U_00.tif"

    def test_bare_key(self) -> None:
        """Bare key without prefix → unchanged."""
        path = "tge-labs/aef/v1/annual/30U/2024_30U_00.tif"
        assert _strip_s3_prefix(path) == path

    def test_incomplete_s3_uri_returns_empty(self) -> None:
        """s3://bucket with no key → empty string."""
        assert _strip_s3_prefix("s3://bucket") == ""


@pytest.mark.unit
class TestBoundsToWindow:
    """Tests for geographic bounds → pixel window conversion."""

    def test_full_extent(self) -> None:
        """Bounds matching full image → full window."""
        transform = Affine(10, 0, 0, 0, -10, 1000)
        window = _bounds_to_window((0, 0, 1000, 1000), transform, 100, 100)
        assert window.col_off == 0
        assert window.row_off == 0
        assert window.width == 100
        assert window.height == 100

    def test_subset_bounds(self) -> None:
        """Bounds within image → correct subset window."""
        transform = Affine(10, 0, 0, 0, -10, 1000)
        window = _bounds_to_window((100, 200, 500, 800), transform, 100, 100)
        assert window.col_off == 10  # 100 / 10
        assert window.row_off == 20  # (1000 - 800) / 10
        assert window.width == 40  # (500 - 100) / 10
        assert window.height == 60  # (800 - 200) / 10

    def test_non_overlapping_bounds_returns_none(self) -> None:
        """Bounds entirely outside the image extent → None."""
        transform = Affine(10, 0, 0, 0, -10, 1000)
        # Request bounds to the right of the image (x > 1000)
        result = _bounds_to_window((2000, 0, 3000, 1000), transform, 100, 100)
        assert result is None


@pytest.mark.unit
class TestBngBoundsToUtm:
    """Tests for BNG → UTM coordinate transformation."""

    def test_returns_four_floats(self) -> None:
        """Result is a 4-tuple of floats."""
        result = bng_bounds_to_utm((530_000, 180_000, 540_000, 190_000), "EPSG:32630")
        assert len(result) == 4
        assert all(isinstance(v, float) for v in result)

    def test_min_less_than_max(self) -> None:
        """minx < maxx and miny < maxy in the output."""
        minx, miny, maxx, maxy = bng_bounds_to_utm(
            (530_000, 180_000, 540_000, 190_000), "EPSG:32630"
        )
        assert minx < maxx
        assert miny < maxy

    def test_padding_expands_bounds(self) -> None:
        """Result is larger than input when padding > 0."""
        no_pad = bng_bounds_to_utm((530_000, 180_000, 540_000, 190_000), "EPSG:32630", padding=0)
        with_pad = bng_bounds_to_utm(
            (530_000, 180_000, 540_000, 190_000), "EPSG:32630", padding=500
        )
        assert with_pad[0] < no_pad[0]  # minx is smaller
        assert with_pad[1] < no_pad[1]  # miny is smaller
        assert with_pad[2] > no_pad[2]  # maxx is larger
        assert with_pad[3] > no_pad[3]  # maxy is larger


@pytest.mark.unit
class TestReadTile:
    """Tests for async read_tile logic."""

    @pytest.mark.asyncio
    async def test_read_tile_with_window(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test read_tile correctly reads window when bounds overlap."""

        mock_geotiff = AsyncMock()
        mock_geotiff.crs = "EPSG:32630"
        mock_geotiff.width = 1000
        mock_geotiff.height = 1000
        mock_geotiff.transform = Affine(10, 0, 0, 0, -10, 1000)

        mock_array = MagicMock()
        mock_array.data = np.ones((64, 10, 10), dtype=np.int8)
        mock_array.transform = Affine(10, 0, 100, 0, -10, 800)
        mock_geotiff.read.return_value = mock_array

        mock_open = AsyncMock(return_value=mock_geotiff)
        monkeypatch.setattr(aef_bng.reader.GeoTIFF, "open", mock_open)

        result = await read_tile("s3://bucket/test.tif", window_bounds=(100, 200, 200, 300))
        assert result is not None
        data, _, crs = result

        assert data.shape == (64, 10, 10)
        assert crs == "EPSG:32630"
        mock_geotiff.read.assert_called_once()

        _, kwargs = mock_geotiff.read.call_args
        assert "window" in kwargs
