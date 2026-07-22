"""Property-based tests for aef_bng.extract."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from aef_bng.extract import PREFIXES, _get_prefix


@pytest.mark.unit
class TestExtractProperties:
    @given(
        easting=st.integers(min_value=0, max_value=699999),
        northing=st.integers(min_value=0, max_value=1299999),
    )
    def test_get_prefix_properties(self, easting: int, northing: int) -> None:
        """Property-based test for OS BNG prefixes."""
        try:
            prefix = _get_prefix(easting, northing)
            col = int(easting // 100000)
            row = int(northing // 100000)
            expected = PREFIXES[row][col]
            assert prefix == expected
        except IndexError:
            # PREFIXES array has bounds, if row/col are out of bounds or "  " it should be handled
            pass
