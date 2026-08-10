"""Tests for aef_bng.extract."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from aef_bng.constants import AEF_NODATA, AEF_NUM_BANDS
from aef_bng.extract import PREFIXES, _get_prefix, _wkb_boxes, extract_pixels, extract_pixels_spark
from aef_bng.grid import ChunkSpec


@pytest.mark.unit
class TestGetPrefix:
    """Tests for 100km prefix lookup."""

    def test_known_prefixes(self) -> None:
        """Verify prefix for known coordinates."""
        assert _get_prefix(430_000, 110_000) == "SU"
        assert _get_prefix(530_000, 180_000) == "TQ"
        assert _get_prefix(250_000, 650_000) == "NS"

    def test_prefix_matches_osbng(self) -> None:
        """PREFIXES array matches the osbng reference."""
        # SU is at row=1 (northing 100k-200k), col=4 (easting 400k-500k)
        assert PREFIXES[1][4] == "SU"
        # TQ is at row=1, col=5
        assert PREFIXES[1][5] == "TQ"
        # HU is at row=11, col=4
        assert PREFIXES[11][4] == "HU"


@pytest.mark.unit
class TestExtractPixels:
    """Tests for pixel extraction to Arrow table."""

    def test_output_schema(self, tq38_chunk: ChunkSpec) -> None:
        """Output table should have the correct columns and types."""
        rng = np.random.default_rng(42)
        data = rng.integers(-127, 128, size=(AEF_NUM_BANDS, 1000, 1000), dtype=np.int8)

        table = extract_pixels(data, tq38_chunk, 2024)
        assert "bng_ref" in table.column_names
        assert "year" in table.column_names
        assert "A00" in table.column_names
        assert "A63" in table.column_names
        assert "easting" in table.column_names
        assert "northing" in table.column_names

    def test_all_nodata_returns_empty(self, tq38_chunk: ChunkSpec) -> None:
        """All-nodata raster should return empty table."""
        data = np.full((AEF_NUM_BANDS, 1000, 1000), AEF_NODATA, dtype=np.int8)
        table = extract_pixels(data, tq38_chunk, 2024)
        assert table.num_rows == 0

    def test_nodata_filtered(self, tq38_chunk: ChunkSpec) -> None:
        """Nodata pixels should be excluded from output."""
        data = np.full((AEF_NUM_BANDS, 1000, 1000), AEF_NODATA, dtype=np.int8)
        # Set top-left 10x10 block to valid data
        data[:, :10, :10] = 42
        table = extract_pixels(data, tq38_chunk, 2024)
        assert table.num_rows == 100  # 10 * 10

    def test_band_columns(self, tq38_chunk: ChunkSpec) -> None:
        """All 64 band columns should be present with correct dtype."""
        data = np.ones((AEF_NUM_BANDS, 10, 10), dtype=np.int8)
        chunk = ChunkSpec(
            bng_10km_ref="TQ38",
            bounds_bng=(530_000, 180_000, 530_100, 180_100),
            bounds_wgs84=(-0.15, 51.48, -0.14, 51.49),
            shape=(10, 10),
        )
        table = extract_pixels(data, chunk, 2024)
        from aef_bng.constants import AEF_BAND_NAMES

        for name in AEF_BAND_NAMES:
            assert name in table.column_names
            assert table.column(name).type == pa.int8()

    def test_bng_ref_format(self, tq38_chunk: ChunkSpec) -> None:
        """BNG refs should be 10 characters: 2 prefix + 4 easting + 4 northing."""
        data = np.ones((AEF_NUM_BANDS, 1000, 1000), dtype=np.int8)
        table = extract_pixels(data, tq38_chunk, 2024)
        refs = table.column("bng_ref").to_pylist()
        for ref in refs[:100]:  # Check first 100
            assert len(ref) == 10, f"Bad ref length: {ref}"
            assert ref[:2].isalpha(), f"Bad prefix: {ref}"
            assert ref[2:].isdigit(), f"Bad numeric part: {ref}"

    def test_year_column(self, tq38_chunk: ChunkSpec) -> None:
        """Year column should contain the correct year."""
        data = np.ones((AEF_NUM_BANDS, 1000, 1000), dtype=np.int8)
        table = extract_pixels(data, tq38_chunk, 2024)
        years = table.column("year").to_pylist()
        assert all(y == 2024 for y in years)

    def test_coordinate_columns(self, tq38_chunk: ChunkSpec) -> None:
        """Easting/northing columns should be within chunk bounds."""
        data = np.ones((AEF_NUM_BANDS, 1000, 1000), dtype=np.int8)
        table = extract_pixels(data, tq38_chunk, 2024)
        eastings = table.column("easting").to_pylist()
        northings = table.column("northing").to_pylist()
        assert all(530_000 <= e < 540_000 for e in eastings[:100])
        assert all(180_000 <= n < 190_000 for n in northings[:100])

    def test_valid_pixel_count(self, tq38_chunk: ChunkSpec) -> None:
        """All-valid 1000x1000 → 1M rows."""
        data = np.ones((AEF_NUM_BANDS, 1000, 1000), dtype=np.int8)
        table = extract_pixels(data, tq38_chunk, 2024)
        assert table.num_rows == 1_000_000

    def test_bng_ref_validated_against_osbng(self) -> None:
        """Cross-check a few BNG refs against osbng.indexing.xy_to_bng."""
        try:
            from osbng import xy_to_bng
        except ImportError:
            pytest.skip("osbng not installed")

        # Create a small 3x3 chunk within SU
        chunk = ChunkSpec(
            bng_10km_ref="SU14",
            bounds_bng=(410_000, 140_000, 410_030, 140_030),
            bounds_wgs84=(-1.5, 50.9, -1.49, 50.91),
            shape=(3, 3),
        )
        data = np.ones((AEF_NUM_BANDS, 3, 3), dtype=np.int8)
        table = extract_pixels(data, chunk, 2024)

        refs = table.column("bng_ref").to_pylist()
        # Top-left pixel: easting=410_000, northing=140_020 (lower-left of cell)
        expected_0 = xy_to_bng(410_000, 140_020, 10).bng_ref_compact
        assert refs[0] == expected_0

    def test_chunk_spanning_100km_boundary_uses_multiple_prefixes(self) -> None:
        """Chunk spanning two 100km squares triggers the multi-prefix slow path."""
        # Bounds 299950-300050 cross the SS/ST boundary at easting 300000
        chunk = ChunkSpec(
            bng_10km_ref="SS99",
            bounds_bng=(299_950, 150_000, 300_050, 150_100),
            bounds_wgs84=(-3.3, 50.6, -3.2, 50.7),
            shape=(10, 10),
        )
        data = np.ones((AEF_NUM_BANDS, 10, 10), dtype=np.int8)
        table = extract_pixels(data, chunk, 2024)
        refs = table.column("bng_ref").to_pylist()
        # Should contain refs from both SS and ST prefixes
        prefixes = {r[:2] for r in refs}
        assert len(prefixes) == 2
        assert "SS" in prefixes
        assert "ST" in prefixes


@pytest.mark.unit
class TestParentRefs:
    """Tests for grid_10km_ref / grid_1km_ref derivation."""

    def test_columns_present_in_both_paths(self, tq38_chunk: ChunkSpec) -> None:
        """Both output paths include string parent-ref columns."""
        data = np.ones((AEF_NUM_BANDS, 1000, 1000), dtype=np.int8)
        for table in (
            extract_pixels(data, tq38_chunk, 2024),
            extract_pixels_spark(data, tq38_chunk, 2024),
        ):
            assert table.schema.field("grid_10km_ref").type == pa.string()
            assert table.schema.field("grid_1km_ref").type == pa.string()

    def test_parents_are_digit_picks_of_child(self, tq38_chunk: ChunkSpec) -> None:
        """Parents take the leading digit(s) of each axis, not a string prefix."""
        data = np.ones((AEF_NUM_BANDS, 1000, 1000), dtype=np.int8)
        table = extract_pixels(data, tq38_chunk, 2024)
        refs = table.column("bng_ref").to_pylist()
        refs_10km = table.column("grid_10km_ref").to_pylist()
        refs_1km = table.column("grid_1km_ref").to_pylist()
        for ref, r10, r1 in list(zip(refs, refs_10km, refs_1km, strict=True))[::9973]:
            assert r10 == ref[:3] + ref[6], f"{ref} -> {r10}"
            assert r1 == ref[:4] + ref[6:8], f"{ref} -> {r1}"

    def test_single_chunk_parent_cardinality(self, tq38_chunk: ChunkSpec) -> None:
        """A full 10km chunk has one 10km parent and exactly 100 1km parents."""
        data = np.ones((AEF_NUM_BANDS, 1000, 1000), dtype=np.int8)
        table = extract_pixels(data, tq38_chunk, 2024)
        assert set(table.column("grid_10km_ref").to_pylist()) == {"TQ38"}
        assert len(set(table.column("grid_1km_ref").to_pylist())) == 100

    def test_parent_refs_validated_against_osbng(self) -> None:
        """Cross-check parent refs against osbng.indexing.xy_to_bng."""
        try:
            from osbng import xy_to_bng
        except ImportError:
            pytest.skip("osbng not installed")

        chunk = ChunkSpec(
            bng_10km_ref="SU14",
            bounds_bng=(410_000, 140_000, 410_030, 140_030),
            bounds_wgs84=(-1.5, 50.9, -1.49, 50.91),
            shape=(3, 3),
        )
        data = np.ones((AEF_NUM_BANDS, 3, 3), dtype=np.int8)
        table = extract_pixels(data, chunk, 2024)

        # Top-left pixel: lower-left corner easting=410_000, northing=140_020
        assert (
            table.column("grid_10km_ref")[0].as_py()
            == xy_to_bng(410_000, 140_020, 10_000).bng_ref_compact
        )
        assert (
            table.column("grid_1km_ref")[0].as_py()
            == xy_to_bng(410_000, 140_020, 1_000).bng_ref_compact
        )

    def test_multi_prefix_chunk_parents_follow_row_prefix(self) -> None:
        """Parents stay consistent when a chunk spans two 100km squares."""
        chunk = ChunkSpec(
            bng_10km_ref="SS99",
            bounds_bng=(299_950, 150_000, 300_050, 150_100),
            bounds_wgs84=(-3.3, 50.6, -3.2, 50.7),
            shape=(10, 10),
        )
        data = np.ones((AEF_NUM_BANDS, 10, 10), dtype=np.int8)
        table = extract_pixels(data, chunk, 2024)
        refs = table.column("bng_ref").to_pylist()
        refs_10km = table.column("grid_10km_ref").to_pylist()
        assert {r[:2] for r in refs_10km} == {"SS", "ST"}
        for ref, r10 in zip(refs, refs_10km, strict=True):
            assert r10 == ref[:3] + ref[6]

    def test_empty_table_keeps_parent_columns(self, tq38_chunk: ChunkSpec) -> None:
        """All-nodata output still carries the parent-ref columns."""
        data = np.full((AEF_NUM_BANDS, 1000, 1000), AEF_NODATA, dtype=np.int8)
        for table in (
            extract_pixels(data, tq38_chunk, 2024),
            extract_pixels_spark(data, tq38_chunk, 2024),
        ):
            assert table.num_rows == 0
            assert "grid_10km_ref" in table.column_names
            assert "grid_1km_ref" in table.column_names


@pytest.mark.unit
class TestWkbBoxes:
    """Tests for _wkb_boxes WKB polygon builder."""

    def test_output_length_matches_input(self) -> None:
        """One WKB per input coordinate."""
        eastings = np.array([530_000.0, 530_010.0, 530_020.0])
        northings = np.array([180_000.0, 180_000.0, 180_000.0])
        result = _wkb_boxes(eastings, northings)
        assert len(result) == 3

    def test_output_is_binary(self) -> None:
        """Result is a pyarrow binary array."""
        result = _wkb_boxes(np.array([530_000.0]), np.array([180_000.0]))
        assert result.type == pa.binary()

    def test_wkb_size_is_93_bytes(self) -> None:
        """Each WKB polygon is exactly 93 bytes."""
        eastings = np.array([530_000.0, 531_000.0])
        northings = np.array([180_000.0, 181_000.0])
        result = _wkb_boxes(eastings, northings)
        for i in range(len(result)):
            assert len(result[i].as_py()) == 93

    def test_wkb_header_bytes(self) -> None:
        """First 13 bytes are the correct little-endian Polygon WKB header."""
        result = _wkb_boxes(np.array([530_000.0]), np.array([180_000.0]))
        wkb = result[0].as_py()
        # byte_order=1(LE), type=3(Polygon), num_rings=1, num_points=5
        expected_header = b"\x01\x03\x00\x00\x00\x01\x00\x00\x00\x05\x00\x00\x00"
        assert wkb[:13] == expected_header

    def test_wkb_decodes_to_correct_box(self) -> None:
        """WKB decodes to a 10m x 10m Polygon at the correct coordinates."""
        from shapely import from_wkb

        easting, northing = 530_000.0, 180_000.0
        result = _wkb_boxes(np.array([easting]), np.array([northing]))
        poly = from_wkb(result[0].as_py())
        assert poly.geom_type == "Polygon"
        assert poly.bounds == (easting, northing, easting + 10, northing + 10)

    def test_polygon_area_is_100_sq_metres(self) -> None:
        """Each polygon covers exactly 100 sq m (10m x 10m)."""
        from shapely import from_wkb

        eastings = np.array([530_000.0, 540_000.0, 550_000.0])
        northings = np.array([180_000.0, 190_000.0, 200_000.0])
        result = _wkb_boxes(eastings, northings)
        for i in range(len(result)):
            poly = from_wkb(result[i].as_py())
            assert abs(poly.area - 100.0) < 1e-6

    def test_empty_input(self) -> None:
        """Empty input produces empty result."""
        result = _wkb_boxes(np.array([]), np.array([]))
        assert len(result) == 0


@pytest.mark.unit
class TestExtractPixelsSpark:
    """Tests for extract_pixels_spark."""

    def test_schema_always_has_geometry(self, tq38_chunk: ChunkSpec) -> None:
        """Schema always includes bng_ref, year, A00..A63, and geometry_wkb."""
        data = np.ones((AEF_NUM_BANDS, 1000, 1000), dtype=np.int8)
        table = extract_pixels_spark(data, tq38_chunk, 2024)
        assert "bng_ref" in table.column_names
        assert "A00" in table.column_names
        assert "geometry_wkb" in table.column_names

    def test_geometry_wkb_is_binary_type(self, tq38_chunk: ChunkSpec) -> None:
        """geometry_wkb column has binary type."""
        data = np.ones((AEF_NUM_BANDS, 1000, 1000), dtype=np.int8)
        table = extract_pixels_spark(data, tq38_chunk, 2024)
        assert table.schema.field("geometry_wkb").type == pa.binary()

    def test_geometry_wkb_row_count_matches(self, tq38_chunk: ChunkSpec) -> None:
        """geometry_wkb has same row count as other columns and no nulls."""
        data = np.ones((AEF_NUM_BANDS, 1000, 1000), dtype=np.int8)
        table = extract_pixels_spark(data, tq38_chunk, 2024)
        assert len(table.column("geometry_wkb")) == table.num_rows
        assert table.column("geometry_wkb").null_count == 0

    def test_geometry_wkb_decodes_to_polygon(self, tq38_chunk: ChunkSpec) -> None:
        """WKB entries decode to 10m x 10m Polygon geometries."""
        from shapely import from_wkb

        data = np.ones((AEF_NUM_BANDS, 1000, 1000), dtype=np.int8)
        table = extract_pixels_spark(data, tq38_chunk, 2024)
        poly = from_wkb(table.column("geometry_wkb")[0].as_py())
        assert poly.geom_type == "Polygon"
        assert abs(poly.area - 100.0) < 1e-6

    def test_all_nodata_returns_empty(self, tq38_chunk: ChunkSpec) -> None:
        """All-nodata raster returns empty table with geometry_wkb column present."""
        data = np.full((AEF_NUM_BANDS, 1000, 1000), AEF_NODATA, dtype=np.int8)
        table = extract_pixels_spark(data, tq38_chunk, 2024)
        assert table.num_rows == 0
        assert "geometry_wkb" in table.column_names

    def test_nodata_filtered(self, tq38_chunk: ChunkSpec) -> None:
        """Only valid pixels included in output."""
        data = np.full((AEF_NUM_BANDS, 1000, 1000), AEF_NODATA, dtype=np.int8)
        data[:, :10, :10] = 1  # 100 valid pixels
        table = extract_pixels_spark(data, tq38_chunk, 2024)
        assert table.num_rows == 100

    def test_no_easting_northing_columns(self) -> None:
        """Spark path does not include easting/northing coordinate columns."""
        chunk = ChunkSpec(
            bng_10km_ref="TQ38",
            bounds_bng=(530_000, 180_000, 530_100, 180_100),
            bounds_wgs84=(-0.15, 51.48, -0.14, 51.49),
            shape=(10, 10),
        )
        data = np.ones((AEF_NUM_BANDS, 10, 10), dtype=np.int8)
        table = extract_pixels_spark(data, chunk, 2024)
        assert "easting" not in table.column_names
        assert "northing" not in table.column_names


@pytest.mark.unit
class TestBoundaryMask:
    """Tests for boundary-mask filtering in pixel extraction."""

    @staticmethod
    def _wkb(geometry: object) -> bytes:
        import shapely

        return shapely.to_wkb(geometry)

    def test_mask_restricts_pixels(self, tq38_chunk: ChunkSpec) -> None:
        """Only pixels whose 10m cell intersects the mask geometry are kept.

        Mask covers columns 0-499 fully and half of column 500; with
        ``all_touched`` semantics that keeps 501 columns x 1000 rows.
        """
        from shapely.geometry import box

        data = np.ones((AEF_NUM_BANDS, 1000, 1000), dtype=np.int8)
        mask = self._wkb(box(530_000, 180_000, 535_005, 190_000))
        table = extract_pixels_spark(data, tq38_chunk, 2024, mask_wkb=mask)
        assert table.num_rows == 501 * 1000

    def test_mask_and_nodata_combine(self, tq38_chunk: ChunkSpec) -> None:
        """Mask ANDs with the nodata mask rather than replacing it."""
        from shapely.geometry import box

        data = np.full((AEF_NUM_BANDS, 1000, 1000), AEF_NODATA, dtype=np.int8)
        data[:, :10, :10] = 1  # 100 valid pixels in the top-left (north-west) corner
        # Mask covers the western half - all 100 valid pixels are inside it.
        mask = self._wkb(box(530_000, 180_000, 535_000, 190_000))
        table = extract_pixels_spark(data, tq38_chunk, 2024, mask_wkb=mask)
        assert table.num_rows == 100

    def test_mask_outside_data_returns_empty(self, tq38_chunk: ChunkSpec) -> None:
        """A mask that excludes every valid pixel yields an empty table."""
        from shapely.geometry import box

        data = np.full((AEF_NUM_BANDS, 1000, 1000), AEF_NODATA, dtype=np.int8)
        data[:, :10, :10] = 1  # valid pixels only in the north-west corner
        # Mask covers the far south-east corner only.
        mask = self._wkb(box(539_000, 180_000, 540_000, 181_000))
        table = extract_pixels_spark(data, tq38_chunk, 2024, mask_wkb=mask)
        assert table.num_rows == 0

    def test_no_mask_keeps_all_valid(self, tq38_chunk: ChunkSpec) -> None:
        """mask_wkb=None preserves the unfiltered behaviour."""
        data = np.ones((AEF_NUM_BANDS, 1000, 1000), dtype=np.int8)
        table = extract_pixels(data, tq38_chunk, 2024, mask_wkb=None)
        assert table.num_rows == 1_000_000

    def test_local_and_spark_paths_agree(self, tq38_chunk: ChunkSpec) -> None:
        """extract_pixels and extract_pixels_spark keep the same masked refs."""
        from shapely.geometry import box

        data = np.ones((AEF_NUM_BANDS, 1000, 1000), dtype=np.int8)
        mask = self._wkb(box(530_000, 180_000, 532_000, 182_000))
        local = extract_pixels(data, tq38_chunk, 2024, mask_wkb=mask)
        spark = extract_pixels_spark(data, tq38_chunk, 2024, mask_wkb=mask)
        assert local.column("bng_ref").to_pylist() == spark.column("bng_ref").to_pylist()
