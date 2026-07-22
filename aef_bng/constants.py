"""Constants for AEF embeddings and British National Grid."""

# AEF embedding constants
AEF_NODATA: int = -128
AEF_NUM_BANDS: int = 64
AEF_BAND_NAMES: list[str] = [f"A{i:02d}" for i in range(64)]

# Source Cooperative S3 configuration (public, no auth required)
AEF_INDEX_BLOB: str = "tge-labs/aef/v1/annual/aef_index.parquet"
AEF_INDEX_STAC_URL: str = (
    "s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/aef_index_stac_geoparquet.parquet"
)
AEF_BUCKET: str = "us-west-2.opendata.source.coop"
AEF_REGION: str = "us-west-2"

# British National Grid constants
BNG_CRS: str = "EPSG:27700"
BNG_RESOLUTION: int = 10  # metres
BNG_BOUNDS: tuple[int, int, int, int] = (0, 0, 700_000, 1_300_000)
CHUNK_SIZE: int = 10_000  # 10km grid squares
