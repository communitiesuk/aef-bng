# Agent Instructions for `aef-bng`

This document provides AI coding agents with the necessary context, rules, and architectural overview for working within the `aef-bng` codebase.

## 1. Project Overview

**Purpose:** Reproject Google DeepMind's Alpha Earth Foundation (AEF) 10m-resolution embedding layers from their native UTM grids to the British National Grid (EPSG:27700). The pipeline supports two execution modes:

- **Local:** Extracts to spatially-partitioned tabular data (GeoParquet).
- **Spark/Databricks:** Extracts natively to distributed Delta Tables.

**Key Data Workflows:**

- Queries STAC GeoParquet tile index for AEF COG bounding boxes utilizing predicate pushdown.
- Asynchronously reads overlapping COGs directly from Source Cooperative S3 via `obstore` and `async-geotiff` (bypassing local download).
- Reprojects native pixel arrays to BNG coordinates utilizing `rasterio.warp`.
- Merges tiles and extracts valid pixels into an optimized tabular structure consisting of 64 `TINYINT` columns for the embedding space.

## 2. Architecture & Modularization

The project is structured as a standard Python package (`aef_bng/`) targeting Python 3.12, utilizing `hatchling` as the build backend.

**Module Breakdown:**

- **`cli.py`**: Click CLI entry point (`aef-bng spark-run`) invoked by Databricks Asset Bundle (DAB) wheel tasks.
- **`config.py` & `constants.py`**: Centralized, validated configurations via Dataclasses, managing environment constants, CRS definitions, and S3 paths.
- **`grid.py`**: BNG 10km chunk enumeration (`ChunkSpec`, `BNGOutputGrid`) via `osbng`.
- **`index.py`, `reader.py`, `reproject.py`**: Core spatial querying against STAC, asynchronous network COG ingestion, and coordinate reprojection.
- **`boundary.py`**: Optional spatial ingest filter - boundary loading and per-chunk FULL/PARTIAL/OUTSIDE classification (e.g. GB-only via ONS BFE countries).
- **`extract.py` & `dequantise.py`**: Pixel value extraction (including boundary-mask rasterisation), BNG reference string generation, and data type transformation.
- **`spark.py`**: Encapsulates distributed Spark specific logic (`mapInArrow`).
- **`types.py` & `utils.py`**: `BoundingBox` type and Databricks Connect/geopandas helpers (off the ingest path).

## 3. Spark & Databricks Execution

When extending or modifying the Databricks/Spark components, adhere to these operational constraints:

- **Distribution Model:** Uses Spark's `mapInArrow` to dispatch BNG chunk processing workloads efficiently across cluster cores using Apache Arrow.
- **Async Integrity:** Each partition runs a self-contained `asyncio` event loop to orchestrate concurrent S3 reads. Blocking calls must be avoided within the async I/O boundaries.
- **Delta Output:** Writes target Unity Catalog Delta Tables employing `optimizeWrite` and `autoCompact`.
- **Data Optimization:** The tables are structured around Liquid Clustering on `(year, bng_ref)`.
- **Infrastructure Management:** Databricks Asset Bundles (DAB) deploy code via templates (`databricks.template.cluster.yml`, `databricks.template.serverless.yml`). `databricks-connect` is used.

## 4. Developer Tools & CI/CD Pipeline

The workspace relies on a robust and modern Python ecosystem. AI Agents must invoke these exact tools directly or rely on `make` / `uv` for environment tasks.

- **Dependency Management:** Managed natively by `uv` (`pyproject.toml` and `uv.lock`). *Do not use `pip`.*
  - Installation: Run `make install` or `uv sync --all-groups --all-extras`.
  - Add packages: `uv add <package>` followed by `uvx uv-sort` to maintain manifest sequence.
- **Linting & Formatting:** Managed strictly by `ruff` (Line length 100, Google-style docstrings).
  - Format/Lint: `uv run ruff format .` and `uv run ruff check . --fix`.
- **Type Checking:** Strict type verification utilizing `pyrefly`.
  - Run Typecheck: `uv run nox -s pyrefly`.
- **Testing Suite:** Orchestrated by `pytest` with `pytest-asyncio`, `pytest-cov`, and `hypothesis` (property-based testing).
  - Run Tests: `make test` or `uv run pytest`.
  - Test constraints: Tests are segmented via markers (`@pytest.mark.unit` vs `@pytest.mark.integration`).
- **Task Orchestration:** The `Makefile` exposes common developer routines, while `nox` (`noxfile.py`) facilitates consistent matrix-based testing and security scanning (`bandit`).

## 5. Agent Coding Guidelines

- **Immutability:** Configuration should remain functionally immutable (use frozen Dataclasses where appropriate).
- **Type Safety:** Type hints are mandatory. Leverage structural patterns and explicitly avoid `Any` types or type system bypassing (e.g., `# type: ignore`) unless exhaustively necessary.
- **Async Concurrency:** Maintain idiomatic asynchronous logic. Refrain from inadvertently introducing synchronous bottlenecks inside `async def` definitions, especially near S3 network paths.
- **Idiomatic Geospatial Handling:** Prefer built-in PyArrow capabilities for memory-efficient IPC and dataframe handling. Avoid unnecessarily serializing objects in PySpark tasks.
