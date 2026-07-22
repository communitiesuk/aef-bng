"""Tests for aef_bng.config."""

from __future__ import annotations

import pytest

from aef_bng.config import AEFBNGConfig
from aef_bng.constants import BNG_BOUNDS


@pytest.mark.unit
class TestAEFBNGConfig:
    """Tests for AEFBNGConfig validation and loading."""

    def test_default_config(self) -> None:
        """Config with required fields and defaults."""
        config = AEFBNGConfig(years=[2024])
        assert config.years == [2024]
        assert config.bounds == BNG_BOUNDS
        assert config.chunk_size == 10_000
        assert config.resampling == "nearest"

    def test_custom_bounds(self) -> None:
        """Config with custom BNG bounds."""
        bounds = (400_000, 100_000, 500_000, 200_000)
        config = AEFBNGConfig(years=[2024], bounds=bounds)
        assert config.bounds == bounds

    def test_empty_years_raises(self) -> None:
        """Empty years list should raise ValueError."""
        with pytest.raises(ValueError, match="At least one year"):
            AEFBNGConfig(years=[])

    def test_invalid_chunk_size_raises(self) -> None:
        """Non-multiple-of-1000 chunk_size should raise ValueError."""
        with pytest.raises(ValueError, match="positive multiple of 1000"):
            AEFBNGConfig(years=[2024], chunk_size=500)

    def test_invalid_bounds_raises(self) -> None:
        """Reversed bounds should raise ValueError."""
        with pytest.raises(ValueError, match="min must be less than max"):
            AEFBNGConfig(years=[2024], bounds=(500_000, 200_000, 400_000, 100_000))

    def test_bounds_wrong_length_raises(self) -> None:
        """Bounds with fewer than 4 elements should raise ValueError."""
        with pytest.raises(ValueError, match="4-tuple"):
            AEFBNGConfig(years=[2024], bounds=(100_000, 200_000, 300_000))  # type: ignore[arg-type]

    def test_table_name(self) -> None:
        """table_name defaults to None and can be set."""
        config = AEFBNGConfig(years=[2024])
        assert config.table_name is None

        config_with_table = AEFBNGConfig(years=[2024], table_name="cat.schema.tbl")
        assert config_with_table.table_name == "cat.schema.tbl"
