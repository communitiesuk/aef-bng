"""AEF tile index querying via STAC GeoParquet.

Queries the AEF STAC GeoParquet index directly from Source Cooperative S3 with predicate pushdown
for spatial and temporal filtering. No local download required.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import pandas as pd
import pyarrow.dataset as ds
from obstore.fsspec import FsspecStore
from shapely.geometry import box

from aef_bng.constants import AEF_INDEX_STAC_URL, AEF_REGION, BNG_BOUNDS
from aef_bng.types import BoundingBox

if TYPE_CHECKING:
    from aef_bng.grid import ChunkSpec

logger = logging.getLogger(__name__)


class AEFBNGIndex:
    """Manages the AEF STAC GeoParquet index for spatial/temporal tile queries.

    Queries the index directly from Source Cooperative S3 with predicate
    pushdown, filtering to tiles that overlap the BNG extent.
    """

    def __init__(self) -> None:
        self._gdf: gpd.GeoDataFrame | None = None

    def load_for_bounds(
        self,
        bounds_bng: tuple[int, int, int, int] = BNG_BOUNDS,
        years: list[int] | None = None,
    ) -> gpd.GeoDataFrame:
        """Query the STAC index from S3 with spatial and temporal filters.

        Uses predicate pushdown on the Parquet bbox and datetime columns
        to minimise data transfer, then clips to the BNG extent.

        Args:
            bounds_bng: BNG bounding box (minx, miny, maxx, maxy) in EPSG:27700.
            years: Years to include. If None, no temporal filter is applied.

        Returns:
            GeoDataFrame of matching AEF tiles.
        """
        bounds_wgs84 = BoundingBox(*bounds_bng).reproject(from_crs=27700, to_crs=4326)

        fs_store = FsspecStore("s3", skip_signature=True, region=AEF_REGION)

        # build PyArrow expression for predicate pushdown.
        # struct subfields (bbox.xmin etc.) require ds.field().
        bbox_filter = (
            (ds.field("bbox", "xmax") >= bounds_wgs84.minx)
            & (ds.field("bbox", "xmin") <= bounds_wgs84.maxx)
            & (ds.field("bbox", "ymax") >= bounds_wgs84.miny)
            & (ds.field("bbox", "ymin") <= bounds_wgs84.maxy)
        )

        if years:
            year_exprs = []
            for year in years:
                start = pd.Timestamp(year=year, month=1, day=1, tz="UTC")
                end = pd.Timestamp(year=year + 1, month=1, day=1, tz="UTC")
                year_exprs.append((ds.field("datetime") >= start) & (ds.field("datetime") < end))
            # OR across years (any matching year), AND with spatial filter
            time_filter = year_exprs[0]
            for expr in year_exprs[1:]:
                time_filter = time_filter | expr
            combined_filter = bbox_filter & time_filter
        else:
            combined_filter = bbox_filter

        logger.info(
            "Querying AEF STAC index for bounds=%s, years=%s",
            bounds_wgs84.as_tuple(),
            years,
        )

        gdf = gpd.read_parquet(
            AEF_INDEX_STAC_URL,
            filesystem=fs_store,
            filters=combined_filter,
        )

        # precise spatial clip: remove tiles whose geometry doesn't actually
        # intersect the BNG extent in WGS84 (bbox filter is approximate)
        bng_geom = box(*bounds_wgs84.as_tuple())
        gdf = gdf[gdf.geometry.intersects(bng_geom)].copy()

        # build spatial index for fast per-chunk queries
        gdf.sindex  # noqa: B018

        self._gdf = gdf
        logger.info("Loaded %d tiles from AEF STAC index", len(gdf))
        return gdf

    def tiles_for_chunk(self, chunk: ChunkSpec, year: int) -> list[dict[str, object]]:
        """Find AEF tiles overlapping a BNG chunk for a given year.

        Tiles are sorted by path for deterministic merge ordering when
        multiple tiles overlap at UTM zone boundaries.

        Args:
            chunk: The BNG chunk specification with WGS84 bounds.
            year: The year to filter tiles for.

        Returns:
            List of dicts with keys: path, crs, year — sorted by path.
        """
        if self._gdf is None:
            raise RuntimeError("Index not loaded. Call load_for_bounds() first.")

        gdf = self._gdf

        # spatial filter using WGS84 bounds
        bbox_geom = box(*chunk.bounds_wgs84)
        spatial_mask = gdf.geometry.intersects(bbox_geom)
        filtered = gdf[spatial_mask]

        # temporal filter — extract year from datetime column
        if "datetime" in filtered.columns:
            filtered = filtered[filtered["datetime"].dt.year == year]  # type: ignore[union-attr]
        elif "year" in filtered.columns:
            filtered = filtered[filtered["year"] == year]

        if len(filtered) == 0:
            return []

        tiles: list[dict[str, object]] = []
        for _, row in filtered.iterrows():
            # extract asset path from STAC assets structure
            path = _extract_asset_path(row)
            crs = _extract_crs(row)

            tiles.append({"path": path, "crs": crs, "year": year})

        # sort by path for deterministic overlap merge ordering
        tiles.sort(key=lambda t: str(t["path"]))

        return tiles


def _extract_asset_path(row: Any) -> str:
    """Extract the COG asset path from a STAC GeoParquet row.

    Handles both nested STAC assets dicts and flat path columns.

    Args:
        row: A row from the STAC GeoParquet index.

    Returns:
        S3 path to the COG tile.
    """
    # try flat 'path' column first (simpler index formats)
    if hasattr(row, "path") and row["path"]:
        return str(row["path"])

    # STAC assets structure: assets -> {key: {href: ...}}
    if hasattr(row, "assets") and row["assets"]:
        assets = row["assets"]
        if isinstance(assets, dict):
            for asset in assets.values():
                if isinstance(asset, dict) and "href" in asset:
                    return str(asset["href"])

    raise ValueError(f"Cannot extract asset path from row: {row}")


def _extract_crs(row: Any) -> str:
    """Extract the CRS from a STAC GeoParquet row.

    Args:
        row: A row from the STAC GeoParquet index.

    Returns:
        CRS string (e.g. "EPSG:32630").
    """
    # try common STAC property names
    for col in ("proj:epsg", "crs", "epsg"):
        if hasattr(row, col) and row[col]:
            val = row[col]
            if isinstance(val, (int | float)):
                return f"EPSG:{int(val)}"
            return str(val)

    # fallback: try to get from properties dict
    if hasattr(row, "properties") and isinstance(row["properties"], dict):
        epsg = row["properties"].get("proj:epsg")
        if epsg:
            return f"EPSG:{int(epsg)}"

    raise ValueError(f"Cannot extract CRS from row: {row}")
