"""Shared test fixtures for aef-bng."""

from __future__ import annotations

import numpy as np
import pytest

from aef_bng.config import AEFBNGConfig
from aef_bng.constants import AEF_NODATA, AEF_NUM_BANDS
from aef_bng.grid import ChunkSpec


@pytest.fixture()
def sample_config(tmp_path: object) -> AEFBNGConfig:
    """Minimal config for testing."""
    return AEFBNGConfig(
        years=[2024],
        bounds=(530_000, 180_000, 540_000, 190_000),  # TQ38 area
        output_path=str(tmp_path),
    )


@pytest.fixture()
def tq38_chunk() -> ChunkSpec:
    """ChunkSpec for TQ38 (central London area)."""
    return ChunkSpec(
        bng_10km_ref="TQ38",
        bounds_bng=(530_000, 180_000, 540_000, 190_000),
        bounds_wgs84=(-0.15, 51.48, -0.01, 51.57),
    )


@pytest.fixture()
def sample_raster() -> np.ndarray:
    """Sample (64, 1000, 1000) int8 raster with some nodata."""
    rng = np.random.default_rng(42)
    data = rng.integers(-127, 128, size=(AEF_NUM_BANDS, 1000, 1000), dtype=np.int8)
    # Set ~30% of pixels to nodata across all bands
    nodata_mask = rng.random((1000, 1000)) < 0.3
    data[:, nodata_mask] = AEF_NODATA
    return data


@pytest.fixture()
def all_nodata_raster() -> np.ndarray:
    """Raster where all pixels are nodata."""
    return np.full((AEF_NUM_BANDS, 1000, 1000), AEF_NODATA, dtype=np.int8)


@pytest.fixture(scope="session")
def spark() -> object:
    """Local SparkSession for testing."""
    try:
        from pyspark.sql import SparkSession
    except ImportError:
        pytest.skip("pyspark not installed")

    return (
        SparkSession.builder.master("local[1]")
        .appName("pytest-pyspark")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )
