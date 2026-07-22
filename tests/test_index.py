"""Tests for aef_bng.index."""

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import box

from aef_bng.grid import ChunkSpec
from aef_bng.index import AEFBNGIndex, _extract_asset_path, _extract_crs


def _make_mock_gdf() -> gpd.GeoDataFrame:
    """Create a mock AEF tile index GeoDataFrame."""
    # Two tiles: one covering southern England, one slightly north
    data = {
        "path": [
            "s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/30U/2024_30U_00.tif",
            "s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/31U/2024_31U_00.tif",
        ],
        "year": [2024, 2024],
        "crs": ["EPSG:32630", "EPSG:32631"],
        "utm_zone": ["30U", "31U"],
        "utm_west": [500000.0, 600000.0],
        "utm_south": [5700000.0, 5700000.0],
        "utm_east": [600000.0, 700000.0],
        "utm_north": [5800000.0, 5800000.0],
        "wgs84_west": [-1.5, 0.0],
        "wgs84_south": [51.0, 51.0],
        "wgs84_east": [0.0, 1.5],
        "wgs84_north": [52.0, 52.0],
    }
    geometries = [
        box(
            data["wgs84_west"][i],
            data["wgs84_south"][i],
            data["wgs84_east"][i],
            data["wgs84_north"][i],
        )
        for i in range(len(data["path"]))
    ]
    return gpd.GeoDataFrame(data, geometry=geometries, crs="EPSG:4326")


@pytest.mark.unit
class TestAEFBNGIndex:
    """Tests for AEFBNGIndex spatial querying."""

    def test_tiles_for_chunk_returns_matching(self) -> None:
        """Tiles overlapping the chunk should be returned."""
        index = AEFBNGIndex()
        index._gdf = _make_mock_gdf()

        # Chunk roughly in the overlap zone
        chunk = ChunkSpec(
            bng_10km_ref="TQ38",
            bounds_bng=(530_000, 180_000, 540_000, 190_000),
            bounds_wgs84=(-0.15, 51.48, -0.01, 51.57),
        )
        tiles = index.tiles_for_chunk(chunk, 2024)
        assert len(tiles) >= 1
        assert all(t["year"] == 2024 for t in tiles)

    def test_tiles_for_chunk_empty_for_wrong_year(self) -> None:
        """No tiles should match for a year not in the index."""
        index = AEFBNGIndex()
        index._gdf = _make_mock_gdf()

        chunk = ChunkSpec(
            bng_10km_ref="TQ38",
            bounds_bng=(530_000, 180_000, 540_000, 190_000),
            bounds_wgs84=(-0.15, 51.48, -0.01, 51.57),
        )
        tiles = index.tiles_for_chunk(chunk, 2020)
        assert len(tiles) == 0

    def test_tiles_for_chunk_empty_for_non_overlapping(self) -> None:
        """No tiles for a chunk outside index coverage."""
        index = AEFBNGIndex()
        index._gdf = _make_mock_gdf()

        # Far north Scotland — outside mock tile coverage
        chunk = ChunkSpec(
            bng_10km_ref="NH52",
            bounds_bng=(250_000, 820_000, 260_000, 830_000),
            bounds_wgs84=(-4.5, 57.3, -4.3, 57.4),
        )
        tiles = index.tiles_for_chunk(chunk, 2024)
        assert len(tiles) == 0

    def test_raises_if_not_loaded(self) -> None:
        """Should raise RuntimeError if index not loaded."""
        index = AEFBNGIndex()
        chunk = ChunkSpec(
            bng_10km_ref="TQ38",
            bounds_bng=(530_000, 180_000, 540_000, 190_000),
            bounds_wgs84=(-0.15, 51.48, -0.01, 51.57),
        )
        with pytest.raises(RuntimeError, match="not loaded"):
            index.tiles_for_chunk(chunk, 2024)

    def test_load_for_bounds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test load_for_bounds with mocked network request."""
        mock_gdf = _make_mock_gdf()

        import geopandas as gpd

        def mock_read_parquet(*args: object, **kwargs: object) -> gpd.GeoDataFrame:
            return mock_gdf

        monkeypatch.setattr(gpd, "read_parquet", mock_read_parquet)

        index = AEFBNGIndex()
        bounds = (530_000, 180_000, 540_000, 190_000)
        gdf = index.load_for_bounds(bounds, years=[2024])

        assert len(gdf) >= 0


class _Row(dict):
    """Dict that also supports attribute access — mimics a pandas Series row."""

    def __getattr__(self, name: str) -> object:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


@pytest.mark.unit
class TestExtractAssetPath:
    """Tests for _extract_asset_path helper."""

    def test_flat_path_column(self) -> None:
        """Row with a flat 'path' column returns it directly."""
        row = _Row(path="s3://bucket/tile.tif")
        assert _extract_asset_path(row) == "s3://bucket/tile.tif"

    def test_stac_assets_href(self) -> None:
        """Row with STAC assets structure returns the href."""
        row = _Row(assets={"data": {"href": "s3://bucket/tile.tif"}})
        assert _extract_asset_path(row) == "s3://bucket/tile.tif"

    def test_no_path_raises(self) -> None:
        """Row with neither path nor assets raises ValueError."""
        with pytest.raises(ValueError, match="Cannot extract asset path"):
            _extract_asset_path(_Row())


@pytest.mark.unit
class TestExtractCrs:
    """Tests for _extract_crs helper."""

    def test_string_crs_column(self) -> None:
        """Row with a string CRS value is returned as-is."""
        row = _Row(crs="EPSG:32630")
        assert _extract_crs(row) == "EPSG:32630"

    def test_integer_epsg_column(self) -> None:
        """Row with an integer EPSG value is formatted as 'EPSG:N'."""
        row = _Row(epsg=32630)
        assert _extract_crs(row) == "EPSG:32630"

    def test_properties_dict_fallback(self) -> None:
        """Row with a properties dict containing proj:epsg uses the fallback."""
        row = _Row(properties={"proj:epsg": 32630})
        assert _extract_crs(row) == "EPSG:32630"

    def test_no_crs_raises(self) -> None:
        """Row with no CRS information raises ValueError."""
        with pytest.raises(ValueError, match="Cannot extract CRS"):
            _extract_crs(_Row())
