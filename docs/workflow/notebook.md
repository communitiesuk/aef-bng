# Notebook

Interactive notebook workflow — best for exploration, testing bounds, and verifying output.

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

## Preview the processing grid

```python
from aef_bng.config import AEFBNGConfig
from aef_bng.grid import BNGOutputGrid

config = AEFBNGConfig(years=YEARS, bounds=BOUNDS, table_name=TABLE_NAME)

grid = BNGOutputGrid(config.bounds, config.chunk_size)
chunks = grid.enumerate_chunks()

print(f"10km chunks: {len(chunks)}")
print(f"Max possible rows: {len(chunks) * len(config.years) * 1_000_000:,}")
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

# All of Great Britain (all years)
from aef_bng.constants import BNG_BOUNDS

config = AEFBNGConfig(
    years=[2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
    bounds=BNG_BOUNDS,  # (0, 0, 700_000, 1_300_000)
    table_name="catalog.schema.aef_bng_gb",
)
```
