<div style="padding: 15px; border-radius: 4px;">
    <strong>⚠️ IMPORTANT</strong>
</div>

> This repository has been archived. Development continues at [communitiesuk/aef-bng](https://github.com/communitiesuk/aef-bng).

# aef-bng

Reproject [AlphaEarth Foundation](https://deepmind.google/blog/alphaearth-foundations-helps-map-our-planet-in-unprecedented-detail/) satellite embeddings to the British National Grid (EPSG:27700) on Databricks.

## Overview

`aef-bng` takes Google DeepMind's AEF 10m-resolution satellite embeddings (stored as Cloud Optimised GeoTIFFs in UTM projection on [Source Cooperative](https://source.coop/repositories/tge-labs/aef)) and reprojects them to the British National Grid.

The output is a Unity Catalog Delta table with 64 int8 embedding bands per 10m pixel, indexed by BNG grid reference.

It was developed as part of the Ministry of Housing, Communities and Local Government's (MHCLG) AAAI lab. The pipeline has been open-sourced in case it is useful for other Databricks users working with AlphaEarth data.

## Architecture

```
AEF COGs (S3, UTM) → [Spark mapInArrow per 10km chunk] → Delta Table (BNG)
                       ├─ Query STAC index (predicate pushdown)
                       ├─ Async COG read (obstore + async-geotiff)
                       ├─ UTM → BNG reprojection (rasterio)
                       ├─ First-valid merge at UTM zone boundaries
                       └─ Vectorised pixel extraction + WKB geometry
```

## Installation

Not published to PyPI. Install from source or wheel.

### From source (development)

```bash
git clone https://github.com/communitiesuk/aef-bng.git
cd aef-bng
make install
```

### From wheel (Databricks)

```bash
uv build --wheel --out-dir dist/
```

Then on Databricks:

```python
%pip install "/Workspace/path/to/aef_bng-*.whl"
dbutils.library.restartPython()
```

Or directly from GitHub:

```python
%pip install git+https://github.com/communitiesuk/aef-bng.git
dbutils.library.restartPython()
```

### Extras

| Extra | Use case |
|-------|----------|
| `[spark]` | Databricks Connect (local IDE → remote cluster) |

## Quickstart

### Notebook

```python
from aef_bng.config import AEFBNGConfig
from aef_bng.spark import process_with_spark

config = AEFBNGConfig(
    years=[2024, 2025],
    bounds=(520830, 170402, 542137, 187507),  # London
    table_name="catalog.schema.aef_embeddings",
)

process_with_spark(config)
```

### CLI

```bash
aef-bng spark-run \
  --bounds "520830,170402,542137,187507" \
  --years "2024,2025" \
  --table-name "catalog.schema.aef_embeddings"
```

### Databricks Asset Bundle

```bash
cp databricks.template.serverless.yml databricks.yml
databricks bundle deploy -t dev
databricks bundle run aef_bng_pipeline -t dev \
    --params bounds=520830,170402,542137,187507 \
    --params years=2024,2025 \
    --params table_name=catalog.schema.aef_embeddings
```

## Output schema

| Column | Type | Description |
|--------|------|-------------|
| `bng_ref` | string | 10-character BNG grid reference (10m cell) |
| `year` | smallint | Year of the AEF embedding |
| `A00`–`A63` | tinyint | 64 int8 embedding bands |
| `easting` | integer | BNG easting (metres) |
| `northing` | integer | BNG northing (metres) |
| `geometry` | geometry | 10m polygon in EPSG:27700 |

## Development

### Initial setup

```bash
make install    # install all deps + pre-commit hooks
```

### Running tests

```bash
make test       # pytest with coverage
```

### Code quality

```bash
make check      # pre-commit (ruff + pyrefly + detect-secrets)
```

### Nox

```bash
make nox        # all sessions (ruff, pyrefly, bandit, tests)
```

### Building

```bash
make build      # wheel to dist/
make clean      # remove artifacts
```

### Documentation

```bash
make serve-docs   # local preview at localhost:8001
make build-docs   # static site to site/
```

Run `make help` to see all available commands.

## Licence

Licensed under the [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

Contains public sector information licensed under the Open Government Licence v3.0.

## Acknowledgements

- [`aef-loader`](https://github.com/jakenotjay/aef-loader) by Jake Wilkins (Apache 2.0) — design patterns and inspiration. See [NOTICE](NOTICE) for details.
- The
[AlphaEarth Foundations](https://deepmind.google/blog/alphaearth-foundations-helps-map-our-planet-in-unprecedented-detail/)
Satellite Embedding dataset is produced by Google and Google DeepMind (CC-BY 4.0). Hosted on
[Google Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL)
or Soource Cooperative as [individual COGs](https://source.coop/tge-labs/aef) or
[Zarr mosaic](https://source.coop/tge-labs/aef-mosaic).
