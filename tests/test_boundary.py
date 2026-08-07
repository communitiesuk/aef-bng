"""Tests for aef_bng.boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import shapely
import shapely.wkb
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Point, box

from aef_bng.boundary import Coverage, _polygonal, classify_chunk, load_boundary

if TYPE_CHECKING:
    from pathlib import Path

MAINLAND = box(0, 0, 25_000, 25_000)
ISLAND = box(60_000, 0, 65_000, 5_000)


@pytest.fixture()
def boundary_file(tmp_path: Path) -> Path:
    """GPKG with a 25km 'mainland' square and a distant 5km 'island' square."""
    import geopandas as gpd

    gdf = gpd.GeoDataFrame(
        {"name": ["mainland", "island"]},
        geometry=[MAINLAND, ISLAND],
        crs="EPSG:27700",
    )
    path = tmp_path / "boundary.gpkg"
    gdf.to_file(path)
    return path


@pytest.mark.unit
class TestLoadBoundary:
    """Tests for boundary loading, querying, and buffering."""

    def test_dissolves_all_features(self, boundary_file: Path) -> None:
        """All features are unioned into one geometry."""
        geometry = load_boundary(str(boundary_file))
        assert geometry.area == pytest.approx(MAINLAND.area + ISLAND.area)

    def test_reads_geoparquet(self, tmp_path: Path) -> None:
        """GeoParquet paths are read natively (recommended format)."""
        import geopandas as gpd

        gdf = gpd.GeoDataFrame(
            {"name": ["mainland", "island"]},
            geometry=[MAINLAND, ISLAND],
            crs="EPSG:27700",
        )
        path = tmp_path / "boundary.parquet"
        gdf.to_parquet(path)
        geometry = load_boundary(str(path), query="name == 'mainland'")
        assert geometry.area == pytest.approx(MAINLAND.area)

    def test_query_filters_features(self, boundary_file: Path) -> None:
        """Attribute query keeps only matching features."""
        geometry = load_boundary(str(boundary_file), query="name == 'mainland'")
        assert geometry.area == pytest.approx(MAINLAND.area)

    def test_query_matching_nothing_raises(self, boundary_file: Path) -> None:
        """A query matching no features should raise, not silently keep nothing."""
        with pytest.raises(ValueError, match="matched no features"):
            load_boundary(str(boundary_file), query="name == 'atlantis'")

    def test_buffer_expands_geometry(self, boundary_file: Path) -> None:
        """Positive buffer grows the dissolved geometry."""
        unbuffered = load_boundary(str(boundary_file), query="name == 'mainland'")
        buffered = load_boundary(str(boundary_file), query="name == 'mainland'", buffer_m=1000)
        assert buffered.area > unbuffered.area
        assert buffered.contains(unbuffered)


@pytest.mark.unit
class TestClassifyChunk:
    """Tests for chunk coverage classification."""

    def test_fully_inside_is_full(self) -> None:
        """A chunk properly inside the boundary needs no mask."""
        coverage, mask_wkb = classify_chunk((5_000, 5_000, 15_000, 15_000), MAINLAND)
        assert coverage is Coverage.FULL
        assert mask_wkb is None

    def test_disjoint_is_outside(self) -> None:
        """A chunk with no overlap is dropped."""
        coverage, mask_wkb = classify_chunk((30_000, 30_000, 40_000, 40_000), MAINLAND)
        assert coverage is Coverage.OUTSIDE
        assert mask_wkb is None

    def test_straddling_is_partial_with_clip(self) -> None:
        """A straddling chunk carries the boundary clipped to the chunk."""
        coverage, mask_wkb = classify_chunk((20_000, 0, 30_000, 10_000), MAINLAND)
        assert coverage is Coverage.PARTIAL
        assert mask_wkb is not None
        clipped = shapely.wkb.loads(mask_wkb)
        assert clipped.equals(box(20_000, 0, 25_000, 10_000))

    def test_edge_touch_only_is_outside(self) -> None:
        """A chunk touching the boundary only along an edge has no area to keep."""
        coverage, mask_wkb = classify_chunk((25_000, 0, 35_000, 10_000), MAINLAND)
        assert coverage is Coverage.OUTSIDE
        assert mask_wkb is None

    def test_prepared_boundary_matches_unprepared(self) -> None:
        """shapely.prepare must not change results (spark.py prepares once)."""
        prepared = box(0, 0, 25_000, 25_000)
        shapely.prepare(prepared)
        for bounds in [
            (5_000, 5_000, 15_000, 15_000),
            (20_000, 0, 30_000, 10_000),
            (30_000, 30_000, 40_000, 40_000),
        ]:
            assert classify_chunk(bounds, prepared)[0] is classify_chunk(bounds, MAINLAND)[0]


@pytest.mark.unit
class TestPolygonal:
    """Tests for polygonal-part extraction from intersection results."""

    def test_polygon_passthrough(self) -> None:
        """Plain polygons are returned unchanged."""
        assert _polygonal(MAINLAND) is MAINLAND

    def test_collection_keeps_only_polygons(self) -> None:
        """Line/point members of a collection are dropped."""
        collection = GeometryCollection([MAINLAND, LineString([(0, 0), (1, 1)]), Point(2, 2)])
        result = _polygonal(collection)
        assert result is not None
        assert result.equals(MAINLAND)

    def test_collection_with_multipolygon_member(self) -> None:
        """MultiPolygon members are flattened and kept."""
        collection = GeometryCollection(
            [MultiPolygon([MAINLAND, ISLAND]), LineString([(0, 0), (1, 1)])]
        )
        result = _polygonal(collection)
        assert result is not None
        assert result.area == pytest.approx(MAINLAND.area + ISLAND.area)

    def test_no_polygonal_parts_returns_none(self) -> None:
        """A collection with no areal member yields None."""
        assert _polygonal(GeometryCollection([LineString([(0, 0), (1, 1)])])) is None
        assert _polygonal(LineString([(0, 0), (1, 1)])) is None
