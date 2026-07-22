# aef-bng

Reproject AlphaEarth Foundation (AEF) satellite embeddings to British National Grid on Databricks.

<figure markdown="span">
    ![London example](assets/london-rgb-pca-composite.jpg)
    <figcaption>AlphaEarth embeddings for London, UK (2025). Copyright Google and Google DeepMind (CC-BY 4.0).</figcaption>
  </figure>

## What is aef-bng?

`aef-bng` takes Google DeepMind's [AlphaEarth Foundation](https://deepmind.google/models-and-capabilities/alpha-earth/) 10m-resolution satellite embeddings (stored as Cloud Optimised GeoTIFFs in UTM projection) and reprojects them to the British National Grid (EPSG:27700).

The output is a Delta table on Databricks with 64 embedding bands per 10m pixel, ready for downstream ML tasks.

## Key Features

- **Distributed processing** via Spark `mapInArrow` on Databricks
- **Async COG reading** via `async-geotiff` + `obstore`
- **UTM→BNG reprojection** with first-valid tile merging for zone boundaries
- **Vectorised pixel extraction** with WKB geometry per pixel
- **10km BNG chunk-based** parallelism (matches OS grid system)

## Architecture

```
AEF COGs (S3, UTM) → [Spark mapInArrow per 10km chunk] → Delta Table (BNG)
                       ├─ Query STAC index for overlapping tiles
                       ├─ Async read + reproject to BNG
                       ├─ Extract pixels + compute WKB geometry
                       └─ Return Arrow RecordBatch
```
