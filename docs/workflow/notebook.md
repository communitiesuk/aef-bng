# Notebook

Interactive notebook workflow - best for exploration, testing bounds, and verifying output.

## Setup

```python
%pip install git+https://github.com/communitiesuk/aef-bng.git
dbutils.library.restartPython()
```

## Configure

```python
CATALOG = "catalog"
SCHEMA = "data"
TABLE = "aef_embeddings"

YEARS = [2024, 2025]
BOUNDS = (520830, 170402, 542137, 187507)  # London

TABLE_NAME = f"{CATALOG}.{SCHEMA}.{TABLE}"
```

## Boundary filtering (optional)

Restrict ingestion to pixels intersecting a boundary file - chunks entirely
outside it are dropped before any S3 read. Any boundary works (GeoParquet
recommended); for Great Britain we recommend the ONS BFE (Extent of the Realm)
countries - see the download example in [CLI](cli.md).

```python
BOUNDARY_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/raw/boundaries/countries/Countries_December_2025_Boundaries_UK_BFE.parquet"
BOUNDARY_QUERY = "CTRY25NM in ['England', 'Scotland', 'Wales']"
```

## Preview the processing grid

```python
from aef_bng.config import AEFBNGConfig
from aef_bng.grid import BNGOutputGrid

config = AEFBNGConfig(
    years=YEARS,
    bounds=BOUNDS,
    table_name=TABLE_NAME,
    boundary_path=BOUNDARY_PATH,  # omit for unfiltered ingestion
    boundary_query=BOUNDARY_QUERY,
)

grid = BNGOutputGrid(config.bounds, config.chunk_size)
chunks = grid.enumerate_chunks()

print(f"10km chunks: {len(chunks)}")
print(f"Max possible rows: {len(chunks) * len(config.years) * 1_000_000:,}")
```

With a boundary configured, preview how many chunks will actually be processed
(FULL chunks skip per-pixel masking; PARTIAL chunks rasterise their clipped
boundary; OUTSIDE chunks are never read):

```python
from collections import Counter

import shapely

from aef_bng.boundary import classify_chunk, load_boundary

boundary = load_boundary(
    config.boundary_path, config.boundary_query, config.boundary_buffer_m
)
shapely.prepare(boundary)

counts = Counter(classify_chunk(c.bounds_bng, boundary)[0].value for c in chunks)
print(dict(counts))  # e.g. {'full': ..., 'partial': ..., 'outside': ...}
```

## Run the pipeline

```python
from aef_bng.spark import process_with_spark

process_with_spark(config)
```

## Apply liquid clustering

```python
spark.sql(f"ALTER TABLE {TABLE_NAME} CLUSTER BY (year, bng_ref)")
spark.sql(f"OPTIMIZE {TABLE_NAME}")
spark.sql(f"ANALYZE TABLE {TABLE_NAME} COMPUTE STATISTICS")
spark.sql(f"VACUUM {TABLE_NAME}")
```

## Verify output

```python
df = spark.table(TABLE_NAME)
df.printSchema()
print(f"Total rows: {df.count():,}")
df.groupBy("year").count().orderBy("year").show()
```

## Spatial queries

The `geometry` column supports Databricks spatial functions. Liquid clustering
on `(year, bng_ref)` means queries filtering on these columns skip irrelevant
files automatically.

```python
from pyspark.sql import functions as F

result = df.filter(
    F.expr(
        "ST_Intersects(geometry, ST_GeomFromWKT("
        "'POLYGON ((532322 176872, 532322 181313, 526975 181313, 526975 176872, 532322 176872))',"
        " 27700))"
    )
)
print(f"Cells in bbox: {result.count():,}")
```

## Other examples

```python
# Wales
config = AEFBNGConfig(
    years=[2025],
    bounds=(153325, 157362, 381182, 400529),
    table_name="catalog.schema.aef_wales",
)

# All of Great Britain (all years), land + foreshore only via the BFE boundary.
# Without the boundary the full bbox also ingests coastal water, Ireland,
# the Isle of Man, and the continental coast wherever AEF has data.
from aef_bng.constants import BNG_BOUNDS

config = AEFBNGConfig(
    years=[2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
    bounds=BNG_BOUNDS,  # (0, 0, 700_000, 1_300_000)
    table_name="catalog.schema.aef_bng_gb",
    boundary_path=BOUNDARY_PATH,
    boundary_query=BOUNDARY_QUERY,
)
```
