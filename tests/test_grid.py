"""Tests for aef_bng.grid."""

from __future__ import annotations

import pytest

from aef_bng.constants import BNG_RESOLUTION
from aef_bng.grid import BNGOutputGrid, ChunkSpec


@pytest.mark.unit
class TestChunkSpec:
    """Tests for ChunkSpec dataclass."""

    def test_transform_computed_from_bounds(self) -> None:
        """Affine transform aligns with chunk bounds."""
        chunk = ChunkSpec(
            bng_10km_ref="SU14",
            bounds_bng=(410_000, 140_000, 420_000, 150_000),
            bounds_wgs84=(-1.5, 50.9, -1.3, 51.0),
        )
        # Top-left pixel origin: (minx, maxy)
        assert chunk.transform.c == 410_000  # x origin
        assert chunk.transform.f == 150_000  # y origin
        assert chunk.transform.a == BNG_RESOLUTION  # pixel width
        assert chunk.transform.e == -BNG_RESOLUTION  # pixel height (negative = north-up)

    def test_default_shape(self) -> None:
        """Default shape is 1000x1000 for 10km at 10m."""
        chunk = ChunkSpec(
            bng_10km_ref="TQ38",
            bounds_bng=(530_000, 180_000, 540_000, 190_000),
            bounds_wgs84=(-0.15, 51.48, -0.01, 51.57),
        )
        assert chunk.shape == (1000, 1000)

    def test_pixel_centres_on_grid(self) -> None:
        """Pixel centres should land on BNG 10m grid points."""
        chunk = ChunkSpec(
            bng_10km_ref="TQ38",
            bounds_bng=(530_000, 180_000, 540_000, 190_000),
            bounds_wgs84=(-0.15, 51.48, -0.01, 51.57),
        )
        # Top-left pixel centre: (minx + 5, maxy - 5)
        cx, cy = chunk.transform * (0.5, 0.5)
        assert cx == 530_005.0
        assert cy == 189_995.0

        # All pixel centres should be at 5m offsets from origin
        assert cx % 10 == 5
        assert cy % 10 == 5


@pytest.mark.unit
class TestBNGOutputGrid:
    """Tests for BNGOutputGrid chunk enumeration."""

    def test_single_chunk(self) -> None:
        """Bounds covering exactly one 10km square → one chunk."""
        grid = BNGOutputGrid(bounds=(530_000, 180_000, 540_000, 190_000))
        chunks = grid.enumerate_chunks()
        assert len(chunks) == 1
        assert chunks[0].bng_10km_ref == "TQ38"
        assert chunks[0].bounds_bng == (530_000, 180_000, 540_000, 190_000)

    def test_multiple_chunks(self) -> None:
        """Bounds covering multiple 10km squares."""
        grid = BNGOutputGrid(bounds=(530_000, 180_000, 550_000, 200_000))
        chunks = grid.enumerate_chunks()
        refs = {c.bng_10km_ref for c in chunks}
        assert len(chunks) == 4
        assert "TQ38" in refs
        assert "TQ48" in refs
        assert "TQ39" in refs
        assert "TQ49" in refs

    def test_wgs84_bounds_valid(self) -> None:
        """WGS84 bounds should be valid longitude/latitude ranges."""
        grid = BNGOutputGrid(bounds=(530_000, 180_000, 540_000, 190_000))
        chunks = grid.enumerate_chunks()
        for chunk in chunks:
            lon_min, lat_min, lon_max, lat_max = chunk.bounds_wgs84
            assert -10 < lon_min < lon_max < 5  # UK longitude range
            assert 49 < lat_min < lat_max < 62  # UK latitude range

    def test_chunk_transform_alignment(self) -> None:
        """Each chunk's affine transform must produce pixel-aligned coordinates."""
        grid = BNGOutputGrid(bounds=(400_000, 100_000, 420_000, 120_000))
        chunks = grid.enumerate_chunks()
        for chunk in chunks:
            # Origin must be on 10m grid
            assert chunk.transform.c % BNG_RESOLUTION == 0
            assert chunk.transform.f % BNG_RESOLUTION == 0
