# CLI

The CLI entry point (`aef-bng spark-run`) is designed for Databricks `python_wheel_task`
jobs. It receives parameters as strings and invokes `process_with_spark` internally.

## Command

```bash
aef-bng spark-run \
  --bounds "520830,170402,542137,187507" \
  --years "2024,2025" \
  --table-name "catalog.data.aef_embeddings" \
  --resampling "nearest"
```

## Parameters

| Flag | Description | Example |
|------|-------------|---------|
| `--bounds` | BNG bounds as comma-separated integers | `"0,0,700000,1300000"` |
| `--years` | Years to process, comma-separated | `"2024,2025"` |
| `--table-name` | Unity Catalog three-level name | `` "catalog.schema.table" `` |
| `--resampling` | Reprojection resampling method | `"nearest"` (default) |

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
    ["spark-run", "--bounds", "0,0,700000,1300000", "--years", "2025", "--table-name", "`catalog`.schema.table"]
    ```

## Cluster requirements

- **Runtime**: DBR 17.3 LTS+ (Spark 4.0, Python 3.12)
- **Network**: Outbound access to `us-west-2.opendata.source.coop` (public S3)
- **Permissions**: Unity Catalog `CREATE TABLE` / `INSERT` on target schema
