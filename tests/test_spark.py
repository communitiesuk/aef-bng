"""Tests for aef_bng.spark — pure functions only, no Spark session required."""

from __future__ import annotations

import pyarrow as pa
import pytest
from pyspark.sql import SparkSession

from aef_bng.constants import AEF_BAND_NAMES
from aef_bng.spark import _build_chunks_dataframe, _empty_batch, _output_schema


@pytest.mark.unit
class TestOutputSchema:
    """Tests for _output_schema."""

    def test_base_fields_present(self) -> None:
        """Schema contains bng_ref, year, all 64 band columns, and geometry_wkb."""
        schema = _output_schema()
        names = schema.names
        assert "bng_ref" in names
        assert "year" in names
        assert "geometry_wkb" in names
        for name in AEF_BAND_NAMES:
            assert name in names

    def test_field_types(self) -> None:
        """bng_ref is string, year is int16, bands are int8, geometry_wkb is binary."""
        schema = _output_schema()
        assert schema.field("bng_ref").type == pa.string()
        assert schema.field("year").type == pa.int16()
        assert schema.field("A00").type == pa.int8()
        assert schema.field("A63").type == pa.int8()
        assert schema.field("geometry_wkb").type == pa.binary()

    def test_band_count(self) -> None:
        """Schema contains exactly 64 band columns."""
        schema = _output_schema()
        band_fields = [n for n in schema.names if n.startswith("A") and n[1:].isdigit()]
        assert len(band_fields) == 64

    def test_field_order(self) -> None:
        """bng_ref and year are first; geometry_wkb is last."""
        schema = _output_schema()
        assert schema.names[0] == "bng_ref"
        assert schema.names[1] == "year"
        assert schema.names[-1] == "geometry_wkb"


@pytest.mark.unit
class TestEmptyBatch:
    """Tests for _empty_batch."""

    def test_empty_batch_has_zero_rows(self) -> None:
        """_empty_batch returns a zero-row RecordBatch."""
        schema = _output_schema()
        batch = _empty_batch(schema)
        assert batch.num_rows == 0

    def test_empty_batch_schema_matches(self) -> None:
        """RecordBatch schema matches the provided schema, including geometry_wkb."""
        schema = _output_schema()
        batch = _empty_batch(schema)
        assert set(batch.schema.names) == set(schema.names)
        assert "geometry_wkb" in batch.schema.names


@pytest.mark.unit
class TestSparkDataFrame:
    """Tests for spark dataframe construction."""

    def test_build_chunks_dataframe(self, spark: object, sample_config: object) -> None:
        """Test _build_chunks_dataframe creates correct schema and row counts."""

        if not isinstance(spark, SparkSession):
            pytest.skip("No spark session")

        config = sample_config
        df = _build_chunks_dataframe(spark, config)

        # sample_config has bounds for 1 10km chunk and 1 year
        assert df.count() == 1

        columns = df.columns
        assert "bng_10km_ref" in columns
        assert "bounds_bng_0" in columns
        assert "bounds_wgs84_0" in columns
        assert "year" in columns
