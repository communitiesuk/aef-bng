# Databricks Asset Bundle (DAB)

DABs provide reproducible, version-controlled deployments.

Two templates are provided: serverless (no cluster management) and dedicated job cluster (full
control).

## Serverless

No cluster startup time - Databricks manages compute. Best for ad-hoc runs.

```bash
# 1. Copy the template
cp databricks.template.serverless.yml databricks.yml

# 2. Edit databricks.yml - set your workspace URL, catalog, schema

# 3. Deploy
databricks bundle deploy -t dev

# 4. Run (boundary params optional - omit for unfiltered ingestion)
databricks bundle run aef_bng_pipeline -t dev \
    --params bounds=520830,170402,542137,187507 \
    --params years=2024,2025 \
    --params table_name=catalog.data.aef_embeddings \
    --params boundary_path=/Volumes/catalog/data/raw/boundaries/countries/Countries_December_2025_Boundaries_UK_BFE.parquet \
    --params "boundary_query=CTRY25NM in ['England', 'Scotland', 'Wales']"
```

### Limitations

- Cannot set `spark.task.cpus` or most Spark config values
- No init scripts
- Libraries specified via `environments`, not `libraries`
- Requires UC serverless permissions for the job identity

## Dedicated Job Cluster

Full control over Spark config, node types, and autoscaling. Best for production.

```bash
# 1. Copy the template
cp databricks.template.cluster.yml databricks.yml

# 2. Edit databricks.yml - set workspace URL, node_type_id, max_workers

# 3. Deploy
databricks bundle deploy -t dev

# 4. Run (GB-only: boundary params drop sea/non-GB chunks before any S3 read)
databricks bundle run aef_bng_pipeline -t dev \
    --params bounds=0,0,700000,1300000 \
    --params years=2017,2018,2019,2020,2021,2022,2023,2024,2025 \
    --params "table_name=\`your-catalog\`.data.aef_bng_gb" \
    --params boundary_path=/Volumes/your-catalog/data/raw/boundaries/countries/Countries_December_2025_Boundaries_UK_BFE.parquet \
    --params "boundary_query=CTRY25NM in ['England', 'Scotland', 'Wales']"
```

### Recommended Spark config

Set in the DAB template under `spark_conf`:

```yaml
spark_conf:
  spark.task.cpus: "1"
  spark.sql.adaptive.enabled: "true"
  spark.databricks.delta.optimizeWrite.enabled: "true"
  spark.databricks.delta.autoCompact.enabled: "true"
  spark.sql.execution.arrow.pyspark.enabled: "true"
  spark.sql.execution.arrow.maxRecordsPerBatch: "10000"
  spark.databricks.io.cache.enabled: "true"
```

### Cluster sizing

| Workload | Node type | Workers | Approximate time |
| ---------- | ----------- | --------- | ----------------- |
| London (test) | Standard_DS4_v2 | 2-4 | ~5 min |
| England | Standard_DS4_v2 | 8-10 | ~2 hours |
| All GB (all years) | Standard_DS4_v2 | 10-20 | ~12 hours |

## Parameterised runs

Both templates accept the same parameters:

| Parameter | Default | Description |
| ----------- | --------- | ------------- |
| `bounds` | `"0,0,700000,1300000"` | BNG bounds (minx,miny,maxx,maxy) |
| `years` | `"2025"` | Comma-separated years |
| `table_name` | From variables | UC three-level table name |
| `resampling` | `"nearest"` | Reprojection method |
| `boundary_path` | `""` (off) | Vector file spatially filtering ingestion |
| `boundary_query` | `""` | Attribute filter on the boundary file |
| `boundary_buffer_m` | `"0"` | Outward buffer on the boundary, metres |

Boundary filtering is off unless `boundary_path` is provided at run time. Any
boundary file works (GeoParquet recommended; GeoJSON/GPKG also accepted). For
GB-only ingestion we recommend the ONS BFE (Extent of the Realm) countries -
download them once as GeoParquet (see the example in [CLI](cli.md)), then:

```bash
databricks bundle run aef_bng_pipeline -t dev \
    --params bounds=0,0,700000,1300000 \
    --params years=2025 \
    --params boundary_path=/Volumes/.../Countries_December_2025_Boundaries_UK_BFE.parquet \
    --params "boundary_query=CTRY25NM in ['England', 'Scotland', 'Wales']"
```

With the GB filter active, roughly two thirds of the full-bbox chunks are dropped
before any S3 read, so the "All GB" timing above roughly halves.

Override at runtime:

```bash
databricks bundle run aef_bng_pipeline -t dev \
    --params bounds=153325,157362,381182,400529 \
    --params years=2025 \
    --params "table_name=\`catalog\`.data.aef_wales"
```
