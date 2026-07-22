"""Tests for aef_bng.dequantise."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from aef_bng.constants import AEF_BAND_NAMES, AEF_NODATA
from aef_bng.dequantise import dequantise, dequantise_dataframe, dequantise_table


def _make_test_table(value: int = 63) -> pa.Table:
    """Create a minimal Arrow table with band columns set to a fixed value."""
    columns: dict = {
        "bng_ref": pa.array(["TQ30001000", "TQ30002000"], type=pa.string()),
        "year": pa.array([2024, 2024], type=pa.int16()),
    }
    for name in AEF_BAND_NAMES:
        columns[name] = pa.array([value, value], type=pa.int8())
    return pa.table(columns)


def _make_test_df(value: int = 63) -> pd.DataFrame:
    """Create a minimal DataFrame with band columns set to a fixed value."""
    data: dict = {
        "bng_ref": ["TQ30001000", "TQ30002000"],
        "year": [2024, 2024],
    }
    for name in AEF_BAND_NAMES:
        data[name] = np.array([value, value], dtype=np.int8)
    return pd.DataFrame(data)


@pytest.mark.unit
class TestDequantise:
    """Tests for the core dequantise function."""

    def test_zero_maps_to_zero(self) -> None:
        """Value 0 maps to 0.0."""
        result = dequantise(np.array([0], dtype=np.int8))
        assert result[0] == pytest.approx(0.0)

    def test_positive_max_value(self) -> None:
        """Value 127 maps to approximately 1.0."""
        result = dequantise(np.array([127], dtype=np.int8))
        expected = (127 / 127.5) ** 2
        assert result[0] == pytest.approx(expected, rel=1e-5)

    def test_negative_max_value(self) -> None:
        """Value -127 maps to approximately -1.0."""
        result = dequantise(np.array([-127], dtype=np.int8))
        expected = -((127 / 127.5) ** 2)
        assert result[0] == pytest.approx(expected, rel=1e-5)

    def test_nodata_maps_to_nan(self) -> None:
        """Nodata value (-128) maps to NaN."""
        result = dequantise(np.array([AEF_NODATA], dtype=np.int8))
        assert np.isnan(result[0])

    def test_output_dtype_is_float32(self) -> None:
        """Output array has float32 dtype."""
        result = dequantise(np.array([1, 10, 100], dtype=np.int8))
        assert result.dtype == np.float32

    def test_symmetry(self) -> None:
        """dequantise(-v) == -dequantise(v) for all valid (non-nodata) values."""
        values = np.array([1, 10, 50, 100, 127], dtype=np.int8)
        pos = dequantise(values)
        neg = dequantise(-values)
        np.testing.assert_allclose(neg, -pos, rtol=1e-5)

    def test_mixed_nodata_and_valid(self) -> None:
        """Nodata and valid values handled correctly in the same array."""
        arr = np.array([AEF_NODATA, 0, 64, AEF_NODATA], dtype=np.int8)
        result = dequantise(arr)
        assert np.isnan(result[0])
        assert result[1] == pytest.approx(0.0)
        assert not np.isnan(result[2])
        assert np.isnan(result[3])


@pytest.mark.unit
class TestDequantiseTable:
    """Tests for dequantise_table."""

    def test_band_columns_become_float32(self) -> None:
        """Band columns (A00..A63) are replaced with float32 arrays."""
        table = _make_test_table(63)
        result = dequantise_table(table)
        for name in AEF_BAND_NAMES:
            assert result.schema.field(name).type == pa.float32()

    def test_non_band_columns_unchanged(self) -> None:
        """Non-band columns pass through with their original type."""
        table = _make_test_table()
        result = dequantise_table(table)
        assert result.schema.field("bng_ref").type == pa.string()
        assert result.schema.field("year").type == pa.int16()

    def test_values_are_dequantised(self) -> None:
        """Band values match the expected dequantisation formula."""
        table = _make_test_table(value=127)
        result = dequantise_table(table)
        expected = float((127 / 127.5) ** 2)
        assert result.column("A00")[0].as_py() == pytest.approx(expected, rel=1e-5)

    def test_row_count_preserved(self) -> None:
        """Row count is unchanged after dequantisation."""
        table = _make_test_table()
        result = dequantise_table(table)
        assert result.num_rows == table.num_rows


@pytest.mark.unit
class TestDequantiseDataframe:
    """Tests for dequantise_dataframe."""

    def test_band_columns_become_float32(self) -> None:
        """Band columns are replaced with float32."""
        df = _make_test_df(63)
        result = dequantise_dataframe(df)
        for name in AEF_BAND_NAMES:
            assert result[name].dtype == np.float32

    def test_non_band_columns_unchanged(self) -> None:
        """Non-band columns are passed through unchanged."""
        df = _make_test_df()
        result = dequantise_dataframe(df)
        assert result["bng_ref"].tolist() == ["TQ30001000", "TQ30002000"]

    def test_original_not_modified(self) -> None:
        """Original DataFrame is not modified (copy semantics)."""
        df = _make_test_df(value=100)
        _ = dequantise_dataframe(df)
        assert df["A00"].dtype == np.int8

    def test_values_are_dequantised(self) -> None:
        """Band values match the expected dequantisation formula."""
        df = _make_test_df(value=127)
        result = dequantise_dataframe(df)
        expected = float((127 / 127.5) ** 2)
        assert result["A00"].iloc[0] == pytest.approx(expected, rel=1e-5)

    def test_missing_band_columns_skipped(self) -> None:
        """Columns absent from the DataFrame are not created."""
        import pandas as pd

        df = pd.DataFrame({"bng_ref": ["TQ30001000"], "A00": np.array([64], dtype=np.int8)})
        result = dequantise_dataframe(df)
        assert result["A00"].dtype == np.float32
        assert "A01" not in result.columns
