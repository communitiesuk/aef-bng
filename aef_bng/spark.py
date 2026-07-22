"""Spark/Databricks distributed processing for AEF-BNG.

Distributes 10km chunk processing across a Spark cluster using
``mapInArrow`` for Arrow-native processing, writing output to a
Unity Catalog Delta table with liquid clustering.
"""

from __future__ import annotations

import asyncio
import logging
import pickle
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa

from aef_bng.constants import AEF_BAND_NAMES, AEF_NODATA
from aef_bng.extract import extract_pixels_spark
from aef_bng.grid import BNGOutputGrid, ChunkSpec
from aef_bng.index import AEFBNGIndex
from aef_bng.reader import bng_bounds_to_utm, read_tile
from aef_bng.reproject import merge_tiles, reproject_tile_to_bng

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pyspark.sql import DataFrame, SparkSession  # type: ignore  # noqa: PGH003

    from aef_bng.config import AEFBNGConfig

logger = logging.getLogger(__name__)


def _output_schema() -> pa.Schema:
    """Build the mapInArrow output schema as PyArrow (used internally for RecordBatch creation).

    Returns:
        PyArrow schema for the UDF output, including the ``geometry_wkb`` field.
    """
    fields: list[tuple[str, pa.DataType]] = (
        [("bng_ref", pa.string()), ("year", pa.int16())]
        + [(name, pa.int8()) for name in AEF_BAND_NAMES]
        + [("geometry_wkb", pa.binary())]
    )
    return pa.schema(fields)


def _spark_output_schema() -> Any:
    """Build the mapInArrow output schema as a PySpark StructType.

    ``mapInArrow`` requires a PySpark StructType (or DDL string), not a PyArrow Schema.

    Returns:
        PySpark StructType matching _output_schema().
    """
    from pyspark.sql.types import (  # type: ignore  # noqa: PGH003
        BinaryType,
        ByteType,
        ShortType,
        StringType,
        StructField,
        StructType,
    )

    fields = (
        [StructField("bng_ref", StringType(), True), StructField("year", ShortType(), True)]
        + [StructField(name, ByteType(), True) for name in AEF_BAND_NAMES]
        + [StructField("geometry_wkb", BinaryType(), True)]
    )
    return StructType(fields)


async def _process_chunk_async(
    bng_10km_ref: str,
    bounds_bng: tuple[int, int, int, int],
    bounds_wgs84: tuple[float, float, float, float],
    year: int,
    index: AEFBNGIndex,
    resampling: str,
) -> pa.Table | None:
    """Process a single chunk (async). Called within a shared event loop per partition.

    Args:
        bng_10km_ref: 10km BNG grid reference.
        bounds_bng: BNG bounding box (minx, miny, maxx, maxy).
        bounds_wgs84: WGS84 bounding box (minx, miny, maxx, maxy).
        year: Year to process.
        index: Loaded AEF tile index.
        resampling: Resampling method name.

    Returns:
        Arrow table with (bng_ref, year, embedding, geometry_wkb) or None if no data.
    """
    chunk = ChunkSpec(
        bng_10km_ref=bng_10km_ref,
        bounds_bng=bounds_bng,
        bounds_wgs84=bounds_wgs84,
    )

    tiles = index.tiles_for_chunk(chunk, year)
    if not tiles:
        return None

    reprojected = []
    for tile in tiles:
        try:
            tile_crs = str(tile["crs"])
            window_bounds = bng_bounds_to_utm(chunk.bounds_bng, tile_crs)
            tile_data = await read_tile(str(tile["path"]), window_bounds=window_bounds)
            if tile_data is None:
                continue
            src_data, src_tf, src_crs = tile_data
            result = reproject_tile_to_bng(
                src_data,
                src_tf,
                src_crs,
                chunk.transform,  # type: ignore[arg-type]
                chunk.shape,
                resampling=resampling,
            )
            reprojected.append(result)
        except Exception:
            logger.exception(
                "Failed to read/reproject tile %s for chunk %s",
                tile["path"],
                bng_10km_ref,
            )

    if not reprojected:
        return None

    merged = merge_tiles(reprojected)
    if np.all(merged == AEF_NODATA):
        return None

    return extract_pixels_spark(merged, chunk, year)


def _make_process_partition(  # noqa: C901
    index_bytes: bytes,
    resampling: str,
) -> Any:
    """Create the partition processing function with closure-captured state.

    Returns the function to pass to ``mapInArrow``. The index bytes and resampling
    string are captured in the closure and serialised with the UDF — compatible with
    serverless, Spark Connect, and classic SparkSession.

    The index is deserialised once per partition (not per row), amortising the cost.

    Args:
        index_bytes: Pickled AEFBNGIndex bytes.
        resampling: Resampling method string.

    Returns:
        Callable suitable for ``mapInArrow``.
    """

    def process_partition(batch_iter: Iterator[pa.RecordBatch]) -> Iterator[pa.RecordBatch]:  # noqa: C901
        """Process all chunks in a partition with a single asyncio event loop.

        Collects all rows from the batch iterator first, then processes them
        in one asyncio.run() call. This avoids repeated event loop creation/
        destruction which causes failures with the obstore/tokio runtime.
        """
        index: AEFBNGIndex = pickle.loads(index_bytes)  # noqa: S301
        schema = _output_schema()

        # Collect all chunk specs from all batches in this partition
        chunk_rows: list[tuple[str, tuple, tuple, int]] = []
        for batch in batch_iter:
            bng_refs = batch.column("bng_10km_ref").to_pylist()
            bounds_bng_0 = batch.column("bounds_bng_0").to_pylist()
            bounds_bng_1 = batch.column("bounds_bng_1").to_pylist()
            bounds_bng_2 = batch.column("bounds_bng_2").to_pylist()
            bounds_bng_3 = batch.column("bounds_bng_3").to_pylist()
            bounds_wgs84_0 = batch.column("bounds_wgs84_0").to_pylist()
            bounds_wgs84_1 = batch.column("bounds_wgs84_1").to_pylist()
            bounds_wgs84_2 = batch.column("bounds_wgs84_2").to_pylist()
            bounds_wgs84_3 = batch.column("bounds_wgs84_3").to_pylist()
            years = batch.column("year").to_pylist()

            for i in range(batch.num_rows):
                chunk_rows.append(
                    (
                        bng_refs[i],
                        (bounds_bng_0[i], bounds_bng_1[i], bounds_bng_2[i], bounds_bng_3[i]),
                        (
                            bounds_wgs84_0[i],
                            bounds_wgs84_1[i],
                            bounds_wgs84_2[i],
                            bounds_wgs84_3[i],
                        ),
                        years[i],
                    )
                )

        if not chunk_rows:
            yield _empty_batch(schema)
            return

        # Process ALL chunks in a single event loop
        async def _process_all() -> list[pa.Table]:
            tables: list[pa.Table] = []
            failed = 0
            for bng_ref, bounds_bng, bounds_wgs84, year in chunk_rows:
                try:
                    result = await _process_chunk_async(
                        bng_10km_ref=bng_ref,
                        bounds_bng=bounds_bng,
                        bounds_wgs84=bounds_wgs84,
                        year=year,
                        index=index,
                        resampling=resampling,
                    )
                    if result is not None and result.num_rows > 0:
                        tables.append(result)
                except Exception:
                    failed += 1
                    logger.exception("Failed to process chunk %s year %d", bng_ref, year)
            if failed:
                logger.warning("Partition: %d/%d chunks failed", failed, len(chunk_rows))
            return tables

        output_tables = asyncio.run(_process_all())

        if output_tables:
            combined = pa.concat_tables(output_tables)
            for record_batch in combined.to_batches():
                if record_batch.num_rows > 0:
                    yield record_batch
        else:
            yield _empty_batch(schema)

    return process_partition


def _empty_batch(schema: pa.Schema) -> pa.RecordBatch:
    """Return an empty RecordBatch matching the given schema."""
    columns = {field.name: pa.array([], type=field.type) for field in schema}
    return pa.RecordBatch.from_pydict(columns, schema=schema)


def _build_chunks_dataframe(
    spark: SparkSession,
    config: AEFBNGConfig,
) -> DataFrame:
    """Build a Spark DataFrame with one row per (chunk, year) combination.

    Uses flat columns for Arrow compatibility — no nested structs or arrays.

    Args:
        spark: Active SparkSession.
        config: Pipeline configuration.

    Returns:
        Spark DataFrame with the flat input schema.
    """
    from pyspark.sql.types import (  # type: ignore  # noqa: PGH003
        DoubleType,
        IntegerType,
        StringType,
        StructField,
        StructType,
    )

    grid = BNGOutputGrid(config.bounds, config.chunk_size)
    chunks = grid.enumerate_chunks()

    rows = []
    for chunk in chunks:
        for year in config.years:
            rows.append(
                (
                    chunk.bng_10km_ref,
                    chunk.bounds_bng[0],
                    chunk.bounds_bng[1],
                    chunk.bounds_bng[2],
                    chunk.bounds_bng[3],
                    chunk.bounds_wgs84[0],
                    chunk.bounds_wgs84[1],
                    chunk.bounds_wgs84[2],
                    chunk.bounds_wgs84[3],
                    year,
                )
            )

    input_schema = StructType(
        [
            StructField("bng_10km_ref", StringType(), False),
            StructField("bounds_bng_0", IntegerType(), False),
            StructField("bounds_bng_1", IntegerType(), False),
            StructField("bounds_bng_2", IntegerType(), False),
            StructField("bounds_bng_3", IntegerType(), False),
            StructField("bounds_wgs84_0", DoubleType(), False),
            StructField("bounds_wgs84_1", DoubleType(), False),
            StructField("bounds_wgs84_2", DoubleType(), False),
            StructField("bounds_wgs84_3", DoubleType(), False),
            StructField("year", IntegerType(), False),
        ]
    )

    return spark.createDataFrame(rows, schema=input_schema)


def process_with_spark(config: AEFBNGConfig) -> None:
    """Run the AEF-BNG pipeline distributed across a Spark cluster.

    Creates a DataFrame of chunk specs with flat columns, serialises the
    AEF index, and processes partitions in parallel using ``mapInArrow``.
    Writes results to a Unity Catalog Delta table.

    Args:
        config: Pipeline configuration.
    """
    import time

    from pyspark.sql import SparkSession

    t_total = time.perf_counter()
    spark = SparkSession.builder.getOrCreate()

    # Phase 1: Load tile index
    t0 = time.perf_counter()
    index = AEFBNGIndex()
    index.load_for_bounds(config.bounds, config.years)
    index_bytes = pickle.dumps(index)
    logger.info("Index loaded in %.1fs", time.perf_counter() - t0)

    # Phase 2: Build chunk grid and partition
    t0 = time.perf_counter()
    chunks_df = _build_chunks_dataframe(spark, config)

    grid = BNGOutputGrid(config.bounds, config.chunk_size)
    num_tasks = len(grid.enumerate_chunks()) * len(config.years)
    # Each chunk is ~64MB in memory (64 bands x 1000x1000 int8) + reprojection overhead.
    # Target 2-4 chunks per partition to balance parallelism vs memory on DS4_v2 (28GB, 8 cores).
    # With autoscaling (1-10 workers x 8 cores = 8-80 slots), use ~num_tasks/3 partitions
    # so each partition processes ~3 chunks sequentially within a single event loop.
    num_partitions = max(1, min(num_tasks, num_tasks // 3))
    chunks_df = chunks_df.repartition(num_partitions)
    logger.info(
        "Grid built: %d tasks across %d partitions in %.1fs",
        num_tasks,
        num_partitions,
        time.perf_counter() - t0,
    )

    # Phase 3: Distributed processing (mapInArrow)
    t0 = time.perf_counter()
    spark_schema = _spark_output_schema()
    process_fn = _make_process_partition(index_bytes, config.resampling)
    result_df = chunks_df.mapInArrow(process_fn, schema=spark_schema)

    from pyspark.databricks.sql import functions as dbf  # type: ignore[import-not-found]
    from pyspark.sql import functions as F  # type: ignore  # noqa: PGH003

    result_df = result_df.withColumn(
        "geometry", dbf.st_geomfromwkb(F.col("geometry_wkb"), F.lit(27700))
    ).drop("geometry_wkb")

    # Phase 4: Write to Delta (triggers execution)
    table_name = config.table_name
    if table_name:
        (
            result_df.write.format("delta")
            .mode("append")
            .option("overwriteSchema", "true")
            .saveAsTable(table_name)
        )
        write_elapsed = time.perf_counter() - t0
        logger.info("Processing + write complete in %.1fs -> %s", write_elapsed, table_name)
    else:
        (result_df.write.format("delta").mode("append").save(config.output_path))
        write_elapsed = time.perf_counter() - t0
        logger.info("Processing + write complete in %.1fs -> %s", write_elapsed, config.output_path)

    logger.info("Spark pipeline total: %.1fs", time.perf_counter() - t_total)
