"""Shared test fixtures for aef-bng."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from aef_bng.config import AEFBNGConfig
from aef_bng.constants import AEF_NODATA, AEF_NUM_BANDS
from aef_bng.grid import ChunkSpec

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


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
def spark() -> SparkSession:
    """Databricks Connect session (or the cluster session when run on Databricks).

    The ``spark`` extra installs databricks-connect, which replaces pyspark and
    cannot start a local ``master("local[1]")`` session, so Spark tests need a
    workspace. Skips when no credentials are configured (see .env.example).
    """
    try:
        from aef_bng.utils import (
            DatabricksSettings,
            get_or_create_spark_session,
            in_databricks,
        )
    except ImportError:
        pytest.skip("databricks-connect not installed (install the spark extra)")

    settings = DatabricksSettings()
    if not in_databricks() and not (settings.databricks_host and settings.databricks_token):
        pytest.skip("Databricks credentials not configured (see .env.example)")

    # Classic cluster if DATABRICKS_CLUSTER_ID is set, otherwise serverless.
    return get_or_create_spark_session(serverless=not settings.databricks_cluster_id)


_SUITE_MARKERS = ("unit", "integration", "databricks")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Reject tests that don't carry exactly one suite marker.

    CI selects ``-m unit``, so a test with no marker would silently never run there.
    """
    bad = []
    for item in items:
        found = [m for m in _SUITE_MARKERS if item.get_closest_marker(m)]
        if len(found) != 1:
            bad.append(f"{item.nodeid} ({', '.join(found) or 'no marker'})")
    if bad:
        msg = f"Each test needs exactly one of {_SUITE_MARKERS} markers:\n  " + "\n  ".join(bad)
        raise pytest.UsageError(msg)
