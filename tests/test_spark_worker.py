"""Tests for the executor-side worker logic in aef_bng.spark.

``process_partition`` (the function handed to ``mapInArrow``), ``_process_chunk_async``
and ``_classify_chunks`` are plain Python over Arrow data, so they run without a
Spark session. COG reads are replaced with synthetic UTM tiles; reprojection and
pixel extraction run for real.
"""

from __future__ import annotations

import logging
import pickle
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pytest
import shapely
from affine import Affine
from osbng import xy_to_bng
from pyproj import Transformer
from shapely.geometry import box

from aef_bng import spark as spark_mod
from aef_bng.config import AEFBNGConfig
from aef_bng.constants import AEF_NODATA, AEF_NUM_BANDS, BNG_CRS
from aef_bng.grid import ChunkSpec
from aef_bng.reader import bng_bounds_to_utm
from aef_bng.spark import (
    _classify_chunks,
    _make_process_partition,
    _output_schema,
    _process_chunk_async,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

UTM_CRS = "EPSG:32630"
# A 100m x 100m chunk in TQ38: 10 x 10 pixels at 10m.
CHUNK_BOUNDS = (530_000, 180_000, 530_100, 180_100)
CHUNK_WGS84 = (-0.13, 51.50, -0.12, 51.51)  # only read by the index, which is faked here

Tile = tuple[np.ndarray, Affine, str]


class FakeIndex:
    """Stands in for AEFBNGIndex, returning the same tiles for every chunk.

    Module-level so it pickles, as the real index does on its way to the executors.
    """

    def __init__(self, paths: list[str]) -> None:
        self.paths = paths

    def tiles_for_chunk(self, chunk: ChunkSpec, year: int) -> list[dict[str, object]]:
        return [{"path": p, "crs": UTM_CRS, "year": year} for p in self.paths]


def to_utm(point_bng: tuple[float, float]) -> tuple[float, float]:
    return Transformer.from_crs(BNG_CRS, UTM_CRS, always_xy=True).transform(*point_bng)


def utm_tile(fill: int, centre_on: tuple[float, float] | None = None) -> Tile:
    """A 10m UTM tile covering the chunk's read window, ``fill`` in every band.

    With ``centre_on`` (a BNG point), the grid is shifted so that point falls
    exactly on a pixel centre.
    """
    minx, miny, maxx, maxy = bng_bounds_to_utm(CHUNK_BOUNDS, UTM_CRS)
    if centre_on is None:
        x0, y0 = np.floor(minx / 10) * 10, np.ceil(maxy / 10) * 10
    else:
        cx, cy = to_utm(centre_on)
        x0 = cx - 5 - 10 * np.ceil((cx - 5 - minx) / 10)
        y0 = cy + 5 + 10 * np.ceil((maxy - cy - 5) / 10)
    width, height = int(np.ceil((maxx - x0) / 10)), int(np.ceil((y0 - miny) / 10))
    data = np.full((AEF_NUM_BANDS, height, width), fill, dtype=np.int8)
    return data, Affine(10, 0, x0, 0, -10, y0), UTM_CRS


def mark_point(tile: Tile, point_bng: tuple[float, float], value: int) -> None:
    """Set the single source pixel containing a BNG point to ``value``."""
    data, transform, _ = tile
    col, row = (int(v) for v in ~transform * to_utm(point_bng))
    data[:, row, col] = value


@pytest.fixture()
def tiles(monkeypatch: pytest.MonkeyPatch) -> dict[str, Tile | Exception | None]:
    """Tile path -> what the faked ``read_tile`` returns (a tile or None) or raises."""
    by_path: dict[str, Tile | Exception | None] = {}

    async def fake_read_tile(path: str, window_bounds: object = None) -> Tile | None:
        result = by_path[path]
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(spark_mod, "read_tile", fake_read_tile)
    return by_path


async def run_chunk(paths: list[str], mask_wkb: bytes | None = None) -> pa.Table | None:
    return await _process_chunk_async(
        bng_10km_ref="TQ38",
        bounds_bng=CHUNK_BOUNDS,
        bounds_wgs84=CHUNK_WGS84,
        year=2024,
        index=FakeIndex(paths),  # type: ignore[arg-type]
        resampling="nearest",
        mask_wkb=mask_wkb,
    )


@pytest.mark.unit
class TestProcessChunk:
    """One chunk: tile lookup, read, reproject, first-valid merge, extract."""

    async def test_no_tiles_returns_none(self, tiles: dict) -> None:
        assert await run_chunk([]) is None

    async def test_single_tile_fills_every_pixel(self, tiles: dict) -> None:
        tiles["a"] = utm_tile(7)
        table = await run_chunk(["a"])

        assert table is not None
        assert table.num_rows == 100
        assert table.schema.equals(_output_schema())
        assert set(table.column("A00").to_pylist()) == {7}
        assert set(table.column("A63").to_pylist()) == {7}
        assert set(table.column("year").to_pylist()) == {2024}

    async def test_first_valid_tile_wins(self, tiles: dict) -> None:
        """Where tiles overlap, the first tile in index order keeps its values."""
        tiles["a"], tiles["b"] = utm_tile(7), utm_tile(42)
        table = await run_chunk(["a", "b"])

        assert table is not None
        assert set(table.column("A00").to_pylist()) == {7}

    async def test_later_tile_fills_nodata(self, tiles: dict) -> None:
        tiles["a"], tiles["b"] = utm_tile(AEF_NODATA), utm_tile(42)
        table = await run_chunk(["a", "b"])

        assert table is not None
        assert table.num_rows == 100
        assert set(table.column("A00").to_pylist()) == {42}

    async def test_failed_and_missing_tiles_are_skipped(
        self, tiles: dict, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A tile that errors or doesn't overlap must not lose the chunk's other tiles."""
        tiles["bad"] = OSError("S3 read failed")
        tiles["missing"] = None
        tiles["good"] = utm_tile(5)

        with caplog.at_level(logging.ERROR, logger="aef_bng.spark"):
            table = await run_chunk(["bad", "missing", "good"])

        assert table is not None
        assert set(table.column("A00").to_pylist()) == {5}
        assert "Failed to read/reproject tile bad" in caplog.text

    async def test_all_nodata_returns_none(self, tiles: dict) -> None:
        tiles["a"] = utm_tile(AEF_NODATA)
        assert await run_chunk(["a"]) is None

    async def test_mask_restricts_pixels(self, tiles: dict) -> None:
        """A PARTIAL chunk's mask keeps only the cells inside the boundary."""
        tiles["a"] = utm_tile(7)
        west_half = shapely.to_wkb(box(530_000, 180_000, 530_050, 180_100))
        table = await run_chunk(["a"], mask_wkb=west_half)

        assert table is not None
        assert table.num_rows == 50

    async def test_known_point_lands_in_its_bng_cell(self, tiles: dict) -> None:
        """A value at a known UTM location comes out under that location's 10m BNG ref.

        End-to-end geolocation check: UTM read window -> reprojection -> bng_ref.
        The point is a BNG cell centre and sits on a UTM pixel centre, so nearest
        resampling gives the value to exactly one cell: the one containing it.
        """
        point = (530_045.0, 180_055.0)
        tile = utm_tile(AEF_NODATA, centre_on=point)
        mark_point(tile, point, 99)
        tiles["a"] = tile

        table = await run_chunk(["a"])

        assert table is not None
        assert table.column("bng_ref").to_pylist() == [xy_to_bng(*point, 10).bng_ref_compact]
        assert table.column("A00").to_pylist() == [99]


def input_batch(rows: Iterable[tuple[str, int, bytes | None]]) -> pa.RecordBatch:
    """Flat chunk rows, as _build_chunks_dataframe lays them out: (ref, year, mask_wkb)."""
    rows = list(rows)
    n = len(rows)
    columns: dict[str, pa.Array] = {
        "bng_10km_ref": pa.array([r[0] for r in rows], pa.string()),
        **{f"bounds_bng_{i}": pa.array([CHUNK_BOUNDS[i]] * n, pa.int32()) for i in range(4)},
        **{f"bounds_wgs84_{i}": pa.array([CHUNK_WGS84[i]] * n, pa.float64()) for i in range(4)},
        "year": pa.array([r[1] for r in rows], pa.int32()),
        "mask_wkb": pa.array([r[2] for r in rows], pa.binary()),
    }
    return pa.RecordBatch.from_pydict(columns)


def output_table(n: int, ref: str) -> pa.Table:
    """An n-row table in the worker's output schema, with ``bng_ref`` set to ``ref``."""
    schema = _output_schema()
    arrays = [
        pa.array([ref] * n, pa.string()) if f.name == "bng_ref" else pa.nulls(n, f.type)
        for f in schema
    ]
    return pa.Table.from_arrays(arrays, schema=schema)


@pytest.fixture()
def chunk_results(monkeypatch: pytest.MonkeyPatch) -> tuple[dict, list[dict]]:
    """Fake ``_process_chunk_async``: chunk ref -> table / None / exception to raise.

    Also returns the keyword arguments of every call, in order.
    """
    results: dict[str, pa.Table | Exception | None] = {}
    calls: list[dict] = []

    async def fake_process_chunk(**kwargs: object) -> pa.Table | None:
        calls.append(kwargs)
        result = results[str(kwargs["bng_10km_ref"])]
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(spark_mod, "_process_chunk_async", fake_process_chunk)
    return results, calls


def run_partition(
    batches: list[pa.RecordBatch], resampling: str = "nearest"
) -> list[pa.RecordBatch]:
    fn: Callable = _make_process_partition(pickle.dumps(FakeIndex(["p"])), resampling)
    return list(fn(iter(batches)))


@pytest.mark.unit
class TestProcessPartition:
    """The mapInArrow function: unpack rows, run each chunk, stream the batches."""

    def test_empty_partition_yields_one_empty_batch(self, chunk_results: tuple) -> None:
        """mapInArrow needs at least one batch in the declared schema."""
        for batches in ([], [input_batch([])]):
            out = run_partition(batches)
            assert len(out) == 1
            assert out[0].num_rows == 0
            assert out[0].schema.equals(_output_schema())

    def test_rows_reach_chunk_processing(self, chunk_results: tuple) -> None:
        results, calls = chunk_results
        results["TQ38"] = results["TQ39"] = None
        run_partition(
            [input_batch([("TQ38", 2023, b"mask")]), input_batch([("TQ39", 2024, None)])],
            "bilinear",
        )

        assert [c["bng_10km_ref"] for c in calls] == ["TQ38", "TQ39"]
        assert [c["year"] for c in calls] == [2023, 2024]
        assert [c["mask_wkb"] for c in calls] == [b"mask", None]
        assert all(c["bounds_bng"] == CHUNK_BOUNDS for c in calls)
        assert all(c["bounds_wgs84"] == CHUNK_WGS84 for c in calls)
        assert all(c["resampling"] == "bilinear" for c in calls)
        # The index survives the pickle round trip it makes to the executors.
        assert all(c["index"].paths == ["p"] for c in calls)

    def test_failed_chunk_does_not_stop_the_partition(
        self, chunk_results: tuple, caplog: pytest.LogCaptureFixture
    ) -> None:
        results, _ = chunk_results
        results["BAD"] = RuntimeError("boom")
        results["GOOD"] = output_table(3, "GOOD")

        with caplog.at_level(logging.WARNING, logger="aef_bng.spark"):
            out = run_partition([input_batch([("BAD", 2024, None), ("GOOD", 2024, None)])])

        refs = [ref for batch in out for ref in batch.column("bng_ref").to_pylist()]
        assert refs == ["GOOD"] * 3
        assert "1/2 chunks failed" in caplog.text

    def test_no_data_yields_one_empty_batch(self, chunk_results: tuple) -> None:
        results, _ = chunk_results
        results["NONE"] = None
        results["EMPTY"] = output_table(0, "EMPTY")

        out = run_partition([input_batch([("NONE", 2024, None), ("EMPTY", 2024, None)])])

        assert len(out) == 1
        assert out[0].num_rows == 0

    def test_large_chunk_is_split_into_100k_batches(self, chunk_results: tuple) -> None:
        """Small batches cap the Arrow-to-JVM serialisation buffer."""
        results, _ = chunk_results
        results["BIG"] = output_table(250_001, "BIG")

        out = run_partition([input_batch([("BIG", 2024, None)])])

        assert [b.num_rows for b in out] == [100_000, 100_000, 50_001]
        assert all(b.schema.equals(_output_schema()) for b in out)


@pytest.mark.unit
class TestClassifyChunks:
    """Boundary filtering of the enumerated chunks, before any S3 read."""

    @staticmethod
    def _chunk(minx: int) -> ChunkSpec:
        return ChunkSpec(
            bng_10km_ref=f"X{minx}",
            bounds_bng=(minx, 0, minx + 10_000, 10_000),
            bounds_wgs84=(0.0, 0.0, 0.0, 0.0),
        )

    def test_no_boundary_keeps_every_chunk_unmasked(self) -> None:
        chunks = [self._chunk(0), self._chunk(10_000)]
        config = AEFBNGConfig(years=[2024], bounds=(0, 0, 20_000, 10_000))

        assert _classify_chunks(chunks, config) == [(chunks[0], None), (chunks[1], None)]

    def test_boundary_drops_outside_and_masks_partial(self, tmp_path: Path) -> None:
        import geopandas as gpd

        path = tmp_path / "boundary.gpkg"
        # Starts below/left of the first chunk: FULL needs contains_properly, not edge contact.
        boundary = box(-5_000, -5_000, 25_000, 25_000)
        gpd.GeoDataFrame(geometry=[boundary], crs=BNG_CRS).to_file(path)
        full, partial, outside = self._chunk(0), self._chunk(20_000), self._chunk(40_000)
        config = AEFBNGConfig(years=[2024], bounds=(0, 0, 50_000, 10_000), boundary_path=str(path))

        kept = _classify_chunks([full, partial, outside], config)

        assert [chunk for chunk, _ in kept] == [full, partial]
        assert kept[0][1] is None
        mask = kept[1][1]
        assert mask is not None
        assert shapely.from_wkb(mask).equals(box(20_000, 0, 25_000, 10_000))
