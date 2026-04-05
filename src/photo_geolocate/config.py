from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Config:
    google_maps_api_key: str | None
    query_image: Path
    query_mode: str
    drone_altitude_m: float | None
    drone_yaw_deg: float | None
    camera_model: str | None
    camera_fov_horizontal_deg: float | None
    camera_fov_vertical_deg: float | None
    camera_fov_diagonal_deg: float | None
    latitude: float
    longitude: float
    radius_meters: float


def load_config(path: Path) -> Config:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw_api_key = payload.get("google_maps_api_key")
    google_maps_api_key = str(raw_api_key).strip() if raw_api_key not in (None, "") else None
    query_payload = payload.get("query", {})
    query_image = Path(query_payload.get("image", "data/input/example.jpg"))
    query_mode = str(query_payload.get("mode", "drone_nadir"))
    drone_altitude_m = query_payload.get("drone_altitude_m")
    drone_yaw_deg = query_payload.get("drone_yaw_deg")
    camera_payload = query_payload.get("camera", {}) if isinstance(query_payload.get("camera"), dict) else {}
    camera_model = camera_payload.get("model", query_payload.get("camera_model"))
    camera_fov_horizontal_deg = camera_payload.get(
        "fov_horizontal_deg", query_payload.get("camera_fov_horizontal_deg")
    )
    camera_fov_vertical_deg = camera_payload.get(
        "fov_vertical_deg", query_payload.get("camera_fov_vertical_deg")
    )
    camera_fov_diagonal_deg = camera_payload.get(
        "fov_diagonal_deg", query_payload.get("camera_fov_diagonal_deg")
    )

    # Sensible default for DJI Mini 4K if only model is supplied.
    model_norm = str(camera_model or "").strip().lower()
    if model_norm in {"dji mini 4k", "dji_mini_4k", "mini 4k"} and camera_fov_diagonal_deg is None:
        camera_fov_diagonal_deg = 83.0

    return Config(
        google_maps_api_key=google_maps_api_key,
        query_image=query_image,
        query_mode=query_mode,
        drone_altitude_m=float(drone_altitude_m) if drone_altitude_m is not None else None,
        drone_yaw_deg=float(drone_yaw_deg) if drone_yaw_deg is not None else None,
        camera_model=str(camera_model) if camera_model not in (None, "") else None,
        camera_fov_horizontal_deg=float(camera_fov_horizontal_deg) if camera_fov_horizontal_deg is not None else None,
        camera_fov_vertical_deg=float(camera_fov_vertical_deg) if camera_fov_vertical_deg is not None else None,
        camera_fov_diagonal_deg=float(camera_fov_diagonal_deg) if camera_fov_diagonal_deg is not None else None,
        latitude=float(payload["prior_location"]["latitude"]),
        longitude=float(payload["prior_location"]["longitude"]),
        radius_meters=float(payload["search"]["radius_meters"]),
    )
