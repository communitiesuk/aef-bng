# aef-bng

Reproject [AlphaEarth Foundation](https://deepmind.google/blog/alphaearth-foundations-helps-map-our-planet-in-unprecedented-detail/)
satellite embeddings to the British National Grid (EPSG:27700) on Databricks.

<figure markdown="span">
    ![London example](assets/london-rgb-pca-composite.jpg)
    <figcaption>AlphaEarth embeddings for London, UK (2025). Copyright Google and Google DeepMind (CC-BY 4.0).</figcaption>
  </figure>

## What is aef-bng?

`aef-bng` takes Google DeepMind's
[AlphaEarth Foundation](https://source.coop/tge-labs/aef/)
10m-resolution satellite embeddings (stored as Cloud Optimised GeoTIFFs in UTM projection) and
reprojects them to the British National Grid (EPSG:27700).

The output is a Unity Catalog Delta table with 64 int8 embedding bands per 10m pixel, indexed by BNG
grid reference and ready for downstream ML tasks.

It was developed as part of the Ministry of Housing, Communities and Local Government's (MHCLG) AAAI
lab to predict the potential of brownfield land - you can
[read more](https://mhclgdigital.blog.gov.uk/2026/07/16/from-pixels-to-policy-the-potential-of-geospatial-embeddings-for-mhclg/)
about the project.

The pipeline has been open-sourced in case it is useful for other Databricks users working with
AlphaEarth data in a vector format using Databricks Spatial functions.

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
