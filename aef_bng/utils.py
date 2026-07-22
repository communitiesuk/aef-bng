from __future__ import annotations

import os

import geoarrow.pyarrow as ga
import geopandas as gpd
from geoarrow.types import crs as geoarrow_crs
from geopandas import GeoDataFrame
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pyspark.databricks.sql import functions as dbf
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


class DatabricksSettings(BaseSettings):
    """Databricks connection credentials loaded from environment or .env file.

    Environment variables:
        DATABRICKS_HOST: Workspace URL (e.g. https://adb-123.azuredatabricks.net)
        DATABRICKS_TOKEN: Personal Access Token
        DATABRICKS_CLUSTER_ID: Cluster ID for classic compute (e.g. 0519-091045-abc123)
    """

    databricks_host: str = Field(default="", description="Databricks Workspace URL")
    databricks_token: str = Field(default="", description="Databricks PAT Token")
    databricks_cluster_id: str = Field(default="", description="Databricks Cluster ID")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def clean_host(self) -> str:
        """Host URL without trailing slash."""
        return self.databricks_host.rstrip("/")

    @property
    def host_with_scheme(self) -> str:
        """Host URL guaranteed to have https:// prefix."""
        host = self.clean_host
        if not host.startswith("https://"):
            host = f"https://{host}"
        return host


def in_databricks() -> bool:
    return os.environ.get("DATABRICKS_RUNTIME_VERSION") is not None


class NotInDatabricks(Exception):
    def __init__(self) -> None:
        super().__init__("Code cannot be run on Mac - please run this notebook in the VDI.")


def raise_error_if_not_in_databricks() -> None:
    if not in_databricks():
        raise NotInDatabricks()


def get_or_create_spark_session(serverless: bool = False) -> SparkSession:
    """Gets or creates a SparkSession.

    On Databricks: returns the existing cluster session.
    Locally: connects via Databricks Connect using credentials from .env.

    Args:
        serverless: If True, use serverless compute. If False, connect to
            the cluster specified by DATABRICKS_CLUSTER_ID in .env.
    """
    if in_databricks():
        from databricks.sdk.runtime import spark

        print("SparkSession already available on Databricks cluster.")
        return spark

    from databricks.connect import DatabricksSession

    if serverless:
        print("Initialising SparkSession with serverless compute...")
        spark = DatabricksSession.builder.serverless().getOrCreate()
        print("SparkSession initialised (serverless).")
        return spark

    settings = DatabricksSettings()
    print(f"Connecting to cluster {settings.databricks_cluster_id}...")
    spark = DatabricksSession.builder.remote(
        host=settings.host_with_scheme,
        token=settings.databricks_token,
        cluster_id=settings.databricks_cluster_id,
    ).getOrCreate()
    print("SparkSession initialised (classic compute).")
    return spark


def geopandas_to_spark(gdf: GeoDataFrame, serverless: bool) -> DataFrame:
    """Converts a GeoPandas DataFrame to a geospatial Spark DataFrame.

    Args:
        gdf: Input geopandas.GeoDataFrame

    Returns:
        A geospatial Spark DataFrame.

    Example:
        import geopandas as gpd

        from geo_ai.utils import geopandas_to_spark

        gdf = gpd.read_parquet("some_file.parquet")
        spark_df = geopandas_to_spark(gdf)
    """
    spark = get_or_create_spark_session(serverless=serverless)

    if gdf.crs is None:
        raise ValueError("The input GeoDataFrame must have a CRS defined.")
    else:
        srid = gdf.crs.to_epsg()
    gdf["geometry_wkb"] = gdf["geometry"].to_wkb()
    spark_gdf = spark.createDataFrame(gdf.drop(columns="geometry"))
    return spark_gdf.withColumn("geometry", dbf.st_geomfromwkb(F.col("geometry_wkb"), srid)).drop(
        "geometry_wkb"
    )


def spark_to_geopandas(sdf: DataFrame, srid: int, geometry_column: str) -> gpd.GeoDataFrame:
    """
    Converts a Spark DataFrame with native geometry to a GeoPandas DataFrame using geoarrow.

    Args:
        sdf: the input Spark DataFrame.
        srid: the source coordinate reference system ID (e.g. 27700).
        geometry_column: name of the Spark column containing GEOMETRY type.

    Returns:
        A GeoPandas GeoDataFrame.

    Example:
        from geo_ai.utils import get_or_create_spark_session, spark_to_geopandas

        delta_table = "catalog.schema.table"
        df = spark.read.table(metadata_table)
        gdf = spark_to_geopandas(df, 27700, "geometry")
    """
    wkb_column = "geometry_wkb"
    final_geometry_name = "geometry"

    wkb_sdf = sdf.withColumn(
        wkb_column, dbf.st_aswkb(dbf.st_transform(F.col(geometry_column), F.lit(srid)))
    )

    other_cols = [c for c in sdf.columns if c != geometry_column]

    arrow_table = wkb_sdf.select(*other_cols, wkb_column).toArrow()

    crs = geoarrow_crs.create(f"EPSG:{srid}")

    geom_col_index = arrow_table.schema.get_field_index(wkb_column)
    wkb_array = arrow_table.column(wkb_column)

    geo_array = ga.with_crs(ga.as_geoarrow(wkb_array), crs)

    arrow_table = arrow_table.remove_column(geom_col_index).append_column(
        final_geometry_name, geo_array
    )

    return gpd.GeoDataFrame.from_arrow(arrow_table)
