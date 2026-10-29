"""Property-based tests for aef_bng.extract."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st
from osbng import xy_to_bng

from aef_bng.extract import _get_prefix


@pytest.mark.unit
class TestExtractProperties:
    @given(
        easting=st.integers(min_value=0, max_value=699_999),
        northing=st.integers(min_value=0, max_value=1_299_999),
    )
    def test_get_prefix_matches_osbng(self, easting: int, northing: int) -> None:
        """The 100km prefix agrees with osbng, an independent BNG implementation."""
        assert (
            _get_prefix(easting, northing) == xy_to_bng(easting, northing, 100_000).bng_ref_compact
        )
