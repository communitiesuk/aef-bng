"""AEF embedding dequantisation.

Converts raw signed int8 AEF embeddings to analysis-ready float32 values in the range [-1, 1].

The mapping is:
    de_quantized = (value / 127.5) ** 2 * sign(value)

where -128 is reserved as nodata.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pyspark.sql import functions as F

from aef_bng.constants import AEF_BAND_NAMES, AEF_NODATA

if TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa


def dequantise(values: np.ndarray) -> np.ndarray:
    """Dequantise raw int8 AEF embedding values to float32.

    Maps int8 [-127, 127] to float32 [-1, 1] using the AEF dequantisation formula.
    Nodata pixels (-128) become NaN.

    Args:
        values: Array of raw int8 pixel values.

    Returns:
        Float32 array of analysis-ready values in [-1, 1].
        Nodata (-128) values are set to NaN.
    """
    arr = values.astype(np.float32)
    nodata_mask = values == AEF_NODATA
    result = (arr / 127.5) ** 2 * np.sign(arr)
    result[nodata_mask] = np.nan
    return result


def dequantise_table(table: pa.Table) -> pa.Table:
    """Dequantise all band columns (A00..A63) in an Arrow table.

    Replaces each int8 band column with a float32 dequantised column.
    Non-band columns are passed through unchanged.

    Args:
        table: Arrow table with int8 band columns named A00..A63.

    Returns:
        Arrow table with float32 band columns.
    """
    import pyarrow as pa

    band_set = set(AEF_BAND_NAMES)
    arrays = {}
    for name in table.column_names:
        col = table.column(name)
        if name in band_set:
            raw = col.to_numpy()
            arrays[name] = pa.array(dequantise(raw), type=pa.float32())
        else:
            arrays[name] = col

    return pa.table(arrays)


def dequantise_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Dequantise all band columns (A00..A63) in a pandas DataFrame.

    Replaces each int8 band column with a float32 dequantised column.
    Non-band columns are passed through unchanged.

    Args:
        df: DataFrame with int8 band columns named A00..A63.

    Returns:
        DataFrame with float32 band columns.
    """
    df = df.copy()
    for name in AEF_BAND_NAMES:
        if name in df.columns:
            df[name] = dequantise(df[name].to_numpy())
    return df


def dequantise_spark(df):
    """Dequantise all 64 band columns (A00..A63) in a PySpark DataFrame.

    Maps int8 [-127, 127] → float [-1, 1] using:
        (value / 127.5)² x sign(value)
    Nodata (-128) becomes null.
    """
    for name in AEF_BAND_NAMES:
        col = F.col(name)
        dequantised = F.pow(col / 127.5, 2) * F.signum(col)
        df = df.withColumn(name, F.when(col != AEF_NODATA, dequantised))
    return df
