"""Type definitions used.

Currently focuses on bounding box definitions and helper methods for simple use."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import geopandas
from pyproj import CRS, Transformer
from shapely.geometry import box


@dataclass
class BoundingBox:
    minx: float
    miny: float
    maxx: float
    maxy: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    def as_tuple(self) -> tuple[float, float, float, float]:
        """Returns the bounding box coordinates as a tuple."""
        return (self.minx, self.miny, self.maxx, self.maxy)

    def as_list(self) -> list[float]:
        """Returns the bounding box coordinates as a list."""
        return [self.minx, self.miny, self.maxx, self.maxy]

    def to_geojson(self) -> dict[str, Any]:
        """
        Returns the bounding box as a GeoJSON Polygon geometry dictionary.
        Note: GeoJSON expects coordinates in EPSG:4326 (WGS84).
        """
        return {
            "type": "Polygon",
            "coordinates": [
                [
                    [self.minx, self.miny],
                    [self.maxx, self.miny],
                    [self.maxx, self.maxy],
                    [self.minx, self.maxy],
                    [self.minx, self.miny],
                ]
            ],
        }

    def to_geodataframe(self, crs: str | int | CRS | None = None) -> geopandas.GeoDataFrame:
        """
        Returns the bounding box as a GeoPandas GeoDataFrame.

        Args:
            crs: The Coordinate Reference System of the bounding box.

        Returns:
            A GeoDataFrame containing the bounding box as a single polygon geometry.
        """

        geom = box(self.minx, self.miny, self.maxx, self.maxy)
        return geopandas.GeoDataFrame({"geometry": [geom]}, geometry="geometry", crs=crs)

    def reproject(self, from_crs: str | int, to_crs: str | int) -> BoundingBox:
        """
        Reprojects the bounding box coordinates to a new CRS.

        Args:
            from_crs: The source CRS (e.g., "EPSG:4326" or 4326).
            to_crs: The target CRS (e.g., "EPSG:27700" or 27700).

        Returns:
            A new BoundingBox instance with transformed coordinates.
        """
        transformer = Transformer.from_crs(
            crs_from=CRS.from_user_input(from_crs),
            crs_to=CRS.from_user_input(to_crs),
            always_xy=True,
        )

        new_xs, new_ys = transformer.transform(xx=[self.minx, self.maxx], yy=[self.miny, self.maxy])

        return BoundingBox(
            minx=min(new_xs),
            miny=min(new_ys),
            maxx=max(new_xs),
            maxy=max(new_ys),
        )
