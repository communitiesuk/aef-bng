"""Tests for aef_bng.types."""

from __future__ import annotations

import pytest

from aef_bng.types import BoundingBox


@pytest.mark.unit
class TestBoundingBox:
    """Tests for BoundingBox dataclass methods."""

    def test_as_tuple(self) -> None:
        """as_tuple returns (minx, miny, maxx, maxy)."""
        bb = BoundingBox(1.0, 2.0, 3.0, 4.0)
        assert bb.as_tuple() == (1.0, 2.0, 3.0, 4.0)

    def test_as_list(self) -> None:
        """as_list returns [minx, miny, maxx, maxy]."""
        bb = BoundingBox(1.0, 2.0, 3.0, 4.0)
        assert bb.as_list() == [1.0, 2.0, 3.0, 4.0]

    def test_as_dict(self) -> None:
        """as_dict returns all four coordinate fields."""
        bb = BoundingBox(1.0, 2.0, 3.0, 4.0)
        assert bb.as_dict() == {"minx": 1.0, "miny": 2.0, "maxx": 3.0, "maxy": 4.0}

    def test_to_geojson_structure(self) -> None:
        """to_geojson returns a valid GeoJSON Polygon with a closed ring."""
        bb = BoundingBox(1.0, 2.0, 3.0, 4.0)
        geojson = bb.to_geojson()
        assert geojson["type"] == "Polygon"
        ring = geojson["coordinates"][0]
        assert len(ring) == 5  # closed ring
        assert ring[0] == [1.0, 2.0]
        assert ring[-1] == ring[0]  # closed

    def test_to_geodataframe(self) -> None:
        """to_geodataframe returns a single-row GeoDataFrame with correct CRS."""
        bb = BoundingBox(-0.15, 51.48, -0.01, 51.57)
        gdf = bb.to_geodataframe(crs="EPSG:4326")
        assert len(gdf) == 1
        assert gdf.crs.to_epsg() == 4326

    def test_reproject_bng_to_wgs84(self) -> None:
        """Reprojecting TQ38 from BNG to WGS84 gives Central London coordinates."""
        bb = BoundingBox(530_000.0, 180_000.0, 540_000.0, 190_000.0)
        result = bb.reproject(from_crs=27700, to_crs=4326)
        assert -1.0 < result.minx < 0.0
        assert 51.0 < result.miny < 52.0
        assert result.minx < result.maxx
        assert result.miny < result.maxy

    def test_reproject_returns_bounding_box(self) -> None:
        """reproject returns a BoundingBox instance."""
        bb = BoundingBox(530_000.0, 180_000.0, 540_000.0, 190_000.0)
        result = bb.reproject(from_crs=27700, to_crs=4326)
        assert isinstance(result, BoundingBox)
