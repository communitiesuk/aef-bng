# CLI

The CLI entry point (`aef-bng spark-run`) is designed for Databricks `python_wheel_task`
jobs. It receives parameters as strings and invokes `process_with_spark` internally.

## Command

```bash
aef-bng spark-run \
  --bounds "520830,170402,542137,187507" \
  --years "2024,2025" \
  --table-name "catalog.data.aef_embeddings" \
  --resampling "nearest" \
  --boundary-path "/Volumes/catalog/data/raw/boundaries/countries/Countries_December_2025_Boundaries_UK_BFE.parquet" \
  --boundary-query "CTRY25NM in ['England', 'Scotland', 'Wales']"
```

Omit `--boundary-path` (or pass an empty string) to ingest everything in the
bounds - see [Boundary filtering](#boundary-filtering) below.

## Parameters

| Flag | Description | Example |
| ------ | ------------- | --------- |
| `--bounds` | BNG bounds as comma-separated integers | `"0,0,700000,1300000"` |
| `--years` | Years to process, comma-separated | `"2024,2025"` |
| `--table-name` | Unity Catalog three-level name | `` "catalog.schema.table" `` |
| `--resampling` | Reprojection resampling method | `"nearest"` (default) |
| `--boundary-path` | Boundary file spatially filtering ingestion (empty = off) | `"/Volumes/.../UK_BFE.parquet"` |
| `--boundary-query` | Attribute filter on the boundary file | `"CTRY25NM in ['England', 'Scotland', 'Wales']"` |
| `--boundary-buffer-m` | Outward buffer on the boundary, metres | `0` (default) |

## Boundary filtering

With `--boundary-path` set, only pixels whose 10m BNG cell intersects the boundary
geometry are ingested. Chunks entirely outside the boundary are dropped before any
S3 read, fully-inside chunks skip per-pixel masking, and edge chunks are masked
with `all_touched` ("any overlap") semantics.

Any boundary file works - **GeoParquet is recommended** (compact, fast, typed CRS;
`.parquet`/`.geoparquet` are read natively), and GeoJSON/GPKG/any OGR format is
also accepted. For Great Britain we recommend the ONS **BFE** ("Extent of the
Realm") country boundaries - Mean Low Water including offshore islands. Download
them once as GeoParquet (example below), then:

```bash
aef-bng spark-run \
  --bounds "0,0,700000,1300000" \
  --years "2025" \
  --table-name "catalog.data.aef_embeddings" \
  --boundary-path "/Volumes/catalog/data/raw/boundaries/countries/Countries_December_2025_Boundaries_UK_BFE.parquet" \
  --boundary-query "CTRY25NM in ['England', 'Scotland', 'Wales']"
```

For GB, ~2/3 of the full-bbox chunks (sea, Ireland, continental coast) are dropped
before any S3 read.

### Example: downloading the ONS BFE GB boundary

One-off step - run in a Databricks notebook to write straight to a Unity Catalog
volume (adapt the URL/checks for any other ArcGIS FeatureServer layer). Features
are fetched **one request per country, concurrently** - the server renders and
streams the four huge geometries in parallel instead of one serial response -
and `geometryPrecision=7` (~1cm in WGS84) substantially shrinks each download
without meaningful loss at 10m resolution. Fetching feature-by-feature also
sidesteps GDAL's `OGR_GEOJSON_MAX_OBJ_SIZE` limit that `gpd.read_file(url)`
hits on these full-resolution features.

```python
import json
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd

SERVICE_URL = (
    "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
    "Countries_December_2025_Boundaries_UK_BFE/FeatureServer/0/query"
)
COUNTRIES = ["England", "Scotland", "Wales", "Northern Ireland"]
OUTPUT = (
    "/Volumes/<catalog>/<schema>/raw/boundaries/countries/"
    "Countries_December_2025_Boundaries_UK_BFE.parquet"
)


def fetch_country(name: str) -> dict:
    params = urllib.parse.urlencode(
        {
            "where": f"CTRY25NM='{name}'",
            "outFields": "CTRY25CD,CTRY25NM",
            "geometryPrecision": "7",  # ~1cm - big size saving on full-res BFE
            "f": "geojson",
        }
    )
    with urllib.request.urlopen(f"{SERVICE_URL}?{params}") as response:
        data = json.loads(response.read())
    # ArcGIS truncates large responses and flags it rather than failing.
    assert not data.get("exceededTransferLimit"), f"{name}: truncated response"
    assert len(data["features"]) == 1, f"{name}: expected exactly 1 feature"
    return data["features"][0]


with ThreadPoolExecutor(max_workers=len(COUNTRIES)) as pool:
    features = list(pool.map(fetch_country, COUNTRIES))

gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
gdf.to_parquet(OUTPUT)

# Sanity check: dissolved England+Scotland+Wales BFE area should slightly
# exceed the 228,900 km2 clipped land area (adds intertidal foreshore).
gb = gdf[gdf["CTRY25NM"].isin(["England", "Scotland", "Wales"])].to_crs("EPSG:27700")
area_km2 = gb.geometry.union_all().area / 1e6
assert 228_900 < area_km2 < 250_000, f"implausible GB area: {area_km2:,.0f} km2"
```

## As a Databricks Job (manual setup)

1. Build the wheel:

    ```bash
    uv build --wheel --out-dir dist/
    ```

2. Upload to a UC Volume or Workspace path

3. Create a Job with a `python_wheel_task`:

    - **Package**: `aef_bng`
    - **Entry point**: `aef-bng`
    - **Parameters**:

    ```json
    ["spark-run",
     "--bounds", "0,0,700000,1300000",
     "--years", "2025",
     "--table-name", "`catalog`.schema.table",
     "--boundary-path", "/Volumes/catalog/schema/raw/boundaries/countries/Countries_December_2025_Boundaries_UK_BFE.parquet",
     "--boundary-query", "CTRY25NM in ['England', 'Scotland', 'Wales']"]
    ```

## Cluster requirements

- **Runtime**: DBR 17.3 LTS+ (Spark 4.0, Python 3.12)
- **Network**: Outbound access to `us-west-2.opendata.source.coop` (public S3)
- **Permissions**: Unity Catalog `CREATE TABLE` / `INSERT` on target schema
