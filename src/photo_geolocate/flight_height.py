from __future__ import annotations

import math
from typing import Any

from photo_geolocate.config import Config
from photo_geolocate.google_tiles import EARTH_RADIUS_M, TILE_SIZE

# Google Static Maps setup used in this project: size=640, scale=2 -> 1280 output pixels.
DEFAULT_TILE_OUTPUT_PIXELS = 1280


def meters_per_pixel_at_lat_zoom(latitude: float, zoom: int) -> float:
    return float(math.cos(math.radians(latitude)) * 2.0 * math.pi * EARTH_RADIUS_M / (TILE_SIZE * (2**zoom)))


def tile_ground_span_m(latitude: float, zoom: int, *, tile_output_pixels: int = DEFAULT_TILE_OUTPUT_PIXELS) -> float:
    return float(meters_per_pixel_at_lat_zoom(latitude, zoom) * float(tile_output_pixels))


def _effective_fov_deg(config: Config) -> tuple[float | None, str]:
    if config.camera_fov_horizontal_deg is not None:
        return float(config.camera_fov_horizontal_deg), "camera_fov_horizontal_deg"
    if config.camera_fov_diagonal_deg is not None:
        # We treat diagonal FoV as the effective framing FoV for this helper's practical estimate.
        return float(config.camera_fov_diagonal_deg), "camera_fov_diagonal_deg"
    if config.camera_fov_vertical_deg is not None:
        return float(config.camera_fov_vertical_deg), "camera_fov_vertical_deg"
    return None, "missing"


def recommended_altitude_for_tile_coverage(
    *,
    config: Config,
    center_lat: float,
    zoom: int,
    tile_output_pixels: int = DEFAULT_TILE_OUTPUT_PIXELS,
) -> dict[str, Any] | None:
    fov_deg, fov_source = _effective_fov_deg(config)
    if fov_deg is None or fov_deg <= 0 or fov_deg >= 179:
        return None
    target_span_m = tile_ground_span_m(center_lat, zoom, tile_output_pixels=tile_output_pixels)
    altitude_m = target_span_m / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    return {
        "camera_model": config.camera_model,
        "fov_deg": float(fov_deg),
        "fov_source": str(fov_source),
        "zoom": int(zoom),
        "center_latitude": float(center_lat),
        "tile_output_pixels": int(tile_output_pixels),
        "target_tile_ground_span_m": float(target_span_m),
        "recommended_altitude_m": float(altitude_m),
    }
