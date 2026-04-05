from __future__ import annotations

import json
import math
import os
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from PIL import Image


EARTH_RADIUS_M = 6_378_137.0
TILE_SIZE = 256


def load_google_maps_api_key(project_root: Path) -> str | None:
    env_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if env_key:
        return env_key.strip()

    env_path = project_root / ".env"
    if not env_path.exists():
        return None
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip().lstrip("\ufeff")
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if line.startswith("GOOGLE_MAPS_API_KEY="):
            value = line.split("=", 1)[1].strip().strip("\"'")
            return value or None
    return None


def latlon_to_tile_xy(lat: float, lon: float, zoom: int) -> tuple[float, float]:
    lat_rad = math.radians(lat)
    n = 2.0**zoom
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def tile_xy_to_latlon(x: float, y: float, zoom: int) -> tuple[float, float]:
    n = 2.0**zoom
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat = math.degrees(lat_rad)
    return lat, lon


def latlon_to_world_px(lat: float, lon: float, zoom: int) -> tuple[float, float]:
    tile_x, tile_y = latlon_to_tile_xy(lat, lon, zoom)
    return tile_x * TILE_SIZE, tile_y * TILE_SIZE


def world_px_to_latlon(world_x: float, world_y: float, zoom: int) -> tuple[float, float]:
    return tile_xy_to_latlon(world_x / TILE_SIZE, world_y / TILE_SIZE, zoom)


def fetch_separate_google_tiles(
    *,
    center_lat: float,
    center_lon: float,
    radius_m: float,
    zoom: int,
    api_key: str,
    out_dir: Path,
    annotate_tiles: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    size_px = 640
    scale = 2
    # Google Static Maps: `scale` increases output pixels but does not increase map coverage.
    # Coverage is controlled by `size` at the requested zoom level.
    tile_world_pixels = size_px
    tile_output_pixels = size_px * scale
    meters_per_pixel = (
        math.cos(math.radians(center_lat)) * 2 * math.pi * EARTH_RADIUS_M / (TILE_SIZE * (2**zoom))
    )
    tile_coverage_m = tile_world_pixels * meters_per_pixel
    tiles_per_side = max(1, int(math.ceil((2.0 * radius_m) / tile_coverage_m)))
    if tiles_per_side % 2 == 0:
        tiles_per_side += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("tile_r*_c*.png"):
        stale.unlink()

    half_tiles = tiles_per_side // 2
    center_world_x, center_world_y = latlon_to_world_px(center_lat, center_lon, zoom)
    tiles: list[dict[str, Any]] = []

    for row in range(tiles_per_side):
        for col in range(tiles_per_side):
            tile_world_x = center_world_x + ((col - half_tiles) * tile_world_pixels)
            tile_world_y = center_world_y + ((row - half_tiles) * tile_world_pixels)
            tile_lat, tile_lon = world_px_to_latlon(tile_world_x, tile_world_y, zoom)
            response = requests.get(
                "https://maps.googleapis.com/maps/api/staticmap",
                params={
                    "center": f"{tile_lat},{tile_lon}",
                    "zoom": zoom,
                    "size": f"{size_px}x{size_px}",
                    "scale": scale,
                    "maptype": "satellite",
                    "key": api_key,
                },
                timeout=60,
                headers={"User-Agent": "photo-geolocate-eloftr"},
            )
            response.raise_for_status()
            tile_image = Image.open(BytesIO(response.content)).convert("RGB")
            if annotate_tiles:
                # Optional debug overlay only; keep default tiles clean for matching/stitching.
                from PIL import ImageDraw

                draw = ImageDraw.Draw(tile_image)
                draw.rectangle((6, 6, tile_image.width - 6, tile_image.height - 6), outline="white", width=3)
                draw.rectangle((10, 10, 240, 48), fill="black")
                draw.text((18, 18), f"row={row} col={col}", fill="white")

            tile_path = out_dir / f"tile_r{row}_c{col}.png"
            tile_image.save(tile_path)

            left_world_x = tile_world_x - (tile_world_pixels / 2.0)
            right_world_x = tile_world_x + (tile_world_pixels / 2.0)
            top_world_y = tile_world_y - (tile_world_pixels / 2.0)
            bottom_world_y = tile_world_y + (tile_world_pixels / 2.0)
            north, west = world_px_to_latlon(left_world_x, top_world_y, zoom)
            south, east = world_px_to_latlon(right_world_x, bottom_world_y, zoom)

            tiles.append(
                {
                    "row": row,
                    "col": col,
                    "path": str(tile_path),
                    "center_latitude": tile_lat,
                    "center_longitude": tile_lon,
                    "south": south,
                    "north": north,
                    "west": west,
                    "east": east,
                }
            )

    meta = {
        "provider": "google_static_satellite",
        "center_lat": center_lat,
        "center_lon": center_lon,
        "radius_meters": radius_m,
        "zoom": zoom,
        "size_px": size_px,
        "scale": scale,
        "tile_world_pixels": tile_world_pixels,
        "tile_output_pixels": tile_output_pixels,
        # Backward-compatible field name used by older debug code.
        "tile_pixels": tile_world_pixels,
        "meters_per_pixel": meters_per_pixel,
        "tile_coverage_m": tile_coverage_m,
        "tiles_per_side": tiles_per_side,
    }
    (out_dir / "index.json").write_text(json.dumps({"meta": meta, "tiles": tiles}, indent=2))
    return tiles, meta
