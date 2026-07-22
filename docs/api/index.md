# API Reference

Auto-generated from source docstrings.

## Package Structure

```
aef_bng/
├── config.py       # Configuration dataclass
├── constants.py    # CRS, bounds, band names
├── types.py        # BoundingBox + CRS reprojection
├── grid.py         # BNG 10km grid enumeration
├── index.py        # STAC GeoParquet tile index
├── reader.py       # Async COG reading
├── reproject.py    # UTM → BNG reprojection
├── extract.py      # Pixel extraction + WKB geometry
├── dequantise.py   # int8 → float32 quantization
└── spark.py        # Databricks/Spark distributed pipeline
```
