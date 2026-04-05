from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from photo_geolocate.config import load_config
from photo_geolocate.efficient_loftr_matcher import EfficientLoFTRMatcher
from photo_geolocate.flight_height import recommended_altitude_for_tile_coverage
from photo_geolocate.google_tiles import (
    EARTH_RADIUS_M,
    TILE_SIZE,
    fetch_separate_google_tiles,
    load_google_maps_api_key,
)

SCORING_MODE_MATCH_CONF = "match_count_mean_confidence"
SCORING_MODE_INLIER_CONF = "inlier_count_mean_confidence"
SCORING_MODE_CHOICES = [SCORING_MODE_MATCH_CONF, SCORING_MODE_INLIER_CONF]
MAX_DIAGNOSTIC_TILES_WITH_MATCH_POINTS = 200


def resolve_google_maps_api_key(config_value: str | None, project_root: Path) -> str | None:
    raw = str(config_value or "").strip()
    if raw:
        if raw.lower().startswith("env:"):
            env_name = raw.split(":", 1)[1].strip()
            if env_name:
                env_value = os.environ.get(env_name, "").strip()
                if env_value:
                    return env_value
        elif raw.startswith("${") and raw.endswith("}") and len(raw) > 3:
            env_name = raw[2:-1].strip()
            if env_name:
                env_value = os.environ.get(env_name, "").strip()
                if env_value:
                    return env_value
        else:
            return raw
    return load_google_maps_api_key(project_root)


def structure_preprocess(image: Image.Image) -> Image.Image:
    """Emphasize stable man-made structure and suppress texture noise."""
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    edges = cv2.Canny(gray_eq, 80, 180)
    edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)
    # Blend equalized luminance with strong edges.
    stacked = np.stack([gray_eq, gray_eq, gray_eq], axis=-1).astype(np.float32)
    edge_boost = np.stack([edges, edges, edges], axis=-1).astype(np.float32)
    fused = np.clip((0.75 * stacked) + (0.55 * edge_boost), 0, 255).astype(np.uint8)
    return Image.fromarray(fused, mode="RGB")


def compute_expected_tiles_per_side(center_lat: float, radius_m: float, zoom: int, size_px: int = 640) -> int:
    meters_per_pixel = (
        math.cos(math.radians(center_lat)) * 2 * math.pi * EARTH_RADIUS_M / (TILE_SIZE * (2**zoom))
    )
    tile_coverage_m = size_px * meters_per_pixel
    tiles_per_side = max(1, int(math.ceil((2.0 * radius_m) / tile_coverage_m)))
    if tiles_per_side % 2 == 0:
        tiles_per_side += 1
    return tiles_per_side


def _compute_homography_inliers(
    keypoints0: np.ndarray,
    keypoints1: np.ndarray,
    use_ransac: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(len(keypoints0))
    if n < 4:
        return np.zeros((n,), dtype=bool), np.full((n,), 9999.0, dtype=np.float32)
    if not use_ransac:
        return np.zeros((n,), dtype=bool), np.full((n,), 9999.0, dtype=np.float32)
    homography, inlier_mask = cv2.findHomography(keypoints0.astype(np.float32), keypoints1.astype(np.float32), cv2.RANSAC, 5.0)
    if homography is None or inlier_mask is None:
        return np.zeros((n,), dtype=bool), np.full((n,), 9999.0, dtype=np.float32)
    mask = inlier_mask.ravel().astype(bool)
    projected = cv2.perspectiveTransform(keypoints0.astype(np.float32).reshape(-1, 1, 2), homography.astype(np.float64)).reshape(-1, 2)
    reproj_error = np.linalg.norm(projected - keypoints1, axis=1).astype(np.float32)
    return mask, reproj_error


def serialize_branch_match(match: Any) -> dict[str, Any]:
    points: list[dict[str, Any]] = []
    n = int(match.match_count)
    inlier_mask, reproj_error = _compute_homography_inliers(match.keypoints0, match.keypoints1, use_ransac=True)
    for idx in range(n):
        x0, y0 = match.keypoints0[idx]
        x1, y1 = match.keypoints1[idx]
        points.append(
            {
                "id": int(idx),
                "x0": float(x0),
                "y0": float(y0),
                "x1": float(x1),
                "y1": float(y1),
                "confidence": float(match.matching_scores[idx]),
                "inlier": bool(inlier_mask[idx]) if idx < len(inlier_mask) else False,
                "selected": False,
                "reproj_quality": float(reproj_error[idx]) if idx < len(reproj_error) else 0.0,
            }
        )
    return {
        "match_count": int(match.match_count),
        "mean_confidence": float(getattr(match, "mean_confidence", 0.0)),
        "inlier_count": int(match.inlier_count),
        "inlier_ratio": float(match.inlier_ratio),
        "reprojection_error_px": float(getattr(match, "reprojection_error_px", 9999.0)),
        "spread_score": float(getattr(match, "spread_score", 0.0)),
        "raw_score": float(getattr(match, "raw_score", 0.0)),
        "score": float(match.score),
        "points": points,
    }


def empty_branch_match() -> dict[str, Any]:
    return {
        "match_count": 0,
        "mean_confidence": 0.0,
        "inlier_count": 0,
        "inlier_ratio": 0.0,
        "reprojection_error_px": 9999.0,
        "spread_score": 0.0,
        "raw_score": 0.0,
        "score": 0.0,
        "points": [],
    }


def rank_tiles_coarse_to_fine(
    *,
    query_image_path: Path,
    tiles: list[dict[str, Any]],
    out_dir: Path,
    shortlist_k: int,
    write_match_visualizations: bool = True,
    use_ransac: bool = True,
    loftr_branch: str = "both",
    loftr_input_width: int = 640,
    loftr_input_height: int = 480,
    scoring_mode: str = SCORING_MODE_MATCH_CONF,
) -> list[dict[str, Any]]:
    t0 = time.perf_counter()
    query_raw_full = Image.open(query_image_path).convert("RGB")
    if loftr_branch not in {"raw", "structure", "both"}:
        raise ValueError(f"Invalid loftr_branch: {loftr_branch}")
    use_raw_branch = loftr_branch in {"raw", "both"}
    use_structure_branch = loftr_branch in {"structure", "both"}
    query_struct_full = structure_preprocess(query_raw_full) if use_structure_branch else None
    matcher = EfficientLoFTRMatcher(
        use_ransac=use_ransac,
        input_width=int(loftr_input_width),
        input_height=int(loftr_input_height),
    )
    viz_dir = out_dir / "match_visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    for stale in viz_dir.glob("*.png"):
        stale.unlink()

    # Stage A: full-image LoFTR for all tiles.
    t_loftr_full_start = time.perf_counter()
    loftr_calls = 0
    candidates: list[dict[str, Any]] = []
    for tile in tiles:
        tile_image = Image.open(Path(tile["path"])).convert("RGB")
        tile_struct = structure_preprocess(tile_image) if use_structure_branch else None
        full_result_raw = matcher.match_images(query_raw_full, tile_image) if use_raw_branch else None
        full_result_struct = (
            matcher.match_images(query_struct_full, tile_struct) if use_structure_branch else None
        )
        loftr_calls += (1 if full_result_raw is not None else 0) + (1 if full_result_struct is not None else 0)
        if full_result_raw is not None and full_result_struct is not None:
            full_result = full_result_raw if full_result_raw.score >= full_result_struct.score else full_result_struct
            full_geometric_fused = (0.55 * full_result_raw.score) + (0.45 * full_result_struct.score)
            full_branch_used = "raw" if full_result is full_result_raw else "structure"
        elif full_result_raw is not None:
            full_result = full_result_raw
            full_geometric_fused = float(full_result_raw.score)
            full_branch_used = "raw"
        elif full_result_struct is not None:
            full_result = full_result_struct
            full_geometric_fused = float(full_result_struct.score)
            full_branch_used = "structure"
        else:
            raise RuntimeError("No LoFTR branch was executed")

        match_conf_score = float(full_result.match_count * full_result.mean_confidence)
        inlier_conf_score = float(full_result.inlier_count * full_result.mean_confidence)
        if scoring_mode == SCORING_MODE_MATCH_CONF:
            base_score = match_conf_score
            score_source = SCORING_MODE_MATCH_CONF
        else:
            base_score = inlier_conf_score
            score_source = SCORING_MODE_INLIER_CONF

        candidates.append(
            {
                **tile,
                "full_match_count": full_result.match_count,
                "full_mean_confidence": full_result.mean_confidence,
                "full_inlier_count": full_result.inlier_count,
                "full_inlier_ratio": full_result.inlier_ratio,
                "full_reprojection_error_px": full_result.reprojection_error_px,
                "full_spread_score": full_result.spread_score,
                "full_geometric_score": full_geometric_fused,
                "full_geometric_score_raw": float(full_result_raw.score) if full_result_raw is not None else 0.0,
                "full_geometric_score_struct": float(full_result_struct.score) if full_result_struct is not None else 0.0,
                "full_branch_used": full_branch_used,
                "score_match_count_mean_confidence": match_conf_score,
                "score_inlier_count_mean_confidence": inlier_conf_score,
                "score": base_score,
                "score_source": score_source,
                "stage": "loftr_full_verified",
            }
        )
        del tile_image
        if tile_struct is not None:
            del tile_struct
        if full_result_raw is not None:
            del full_result_raw
        if full_result_struct is not None:
            del full_result_struct
    t_loftr_full = time.perf_counter() - t_loftr_full_start
    candidates.sort(key=lambda item: item["score"], reverse=True)
    shortlist = candidates[: min(shortlist_k, len(candidates))]

    # Score shortlist using configured LoFTR score source.
    for tile in shortlist:
        match_conf_score = float(tile["full_match_count"] * tile["full_mean_confidence"])
        inlier_conf_score = float(tile["full_inlier_count"] * tile["full_mean_confidence"])
        tile["score_match_count_mean_confidence"] = match_conf_score
        tile["score_inlier_count_mean_confidence"] = inlier_conf_score

        if scoring_mode == SCORING_MODE_MATCH_CONF:
            tile["score"] = match_conf_score
            tile["score_source"] = SCORING_MODE_MATCH_CONF
        else:
            tile["score"] = inlier_conf_score
            tile["score_source"] = SCORING_MODE_INLIER_CONF

    final_ranked = sorted(candidates, key=lambda item: item["score"], reverse=True)

    # Visualize top candidate using full-query matches.
    if write_match_visualizations:
        for rank, tile in enumerate(final_ranked[:1], start=1):
            tile_image = Image.open(Path(tile["path"])).convert("RGB")
            result = matcher.match_images(query_raw_full, tile_image)
            matcher.visualize_match(
                query_raw_full,
                tile_image,
                result,
                viz_dir / f"rank_{rank:02d}_tile_r{tile['row']}_c{tile['col']}.png",
                text=[
                    f"Rank #{rank}",
                    f"tile row={tile['row']} col={tile['col']}",
                    f"stage {tile['stage']}",
                    f"geom {tile['full_geometric_score']:.4f}",
                    f"final {tile['score']:.4f}",
                ],
            )
            del tile_image

    elapsed = time.perf_counter() - t0
    print(
        "timing: "
        f"loftr_full={t_loftr_full:.2f}s "
        f"loftr_branch={loftr_branch} "
        f"loftr_input={int(loftr_input_width)}x{int(loftr_input_height)} "
        f"scoring_mode={scoring_mode} "
        f"ransac={'on' if use_ransac else 'off'} "
        f"loftr_calls={loftr_calls} "
        f"total={elapsed:.2f}s"
    )

    diagnostics_payload = {
        "query_image": str(query_image_path),
        "scoring_mode": str(scoring_mode),
        "diagnostics_tile_limit": int(MAX_DIAGNOSTIC_TILES_WITH_MATCH_POINTS),
        "diagnostics_truncated": len(shortlist) > int(MAX_DIAGNOSTIC_TILES_WITH_MATCH_POINTS),
        "tiles": [],
    }
    diagnostics_tiles = shortlist[: int(MAX_DIAGNOSTIC_TILES_WITH_MATCH_POINTS)]
    for tile in diagnostics_tiles:
        full_diag_raw = None
        full_diag_struct = None
        if use_raw_branch or use_structure_branch:
            tile_image = Image.open(Path(tile["path"])).convert("RGB")
            tile_struct = structure_preprocess(tile_image) if use_structure_branch else None
            if use_raw_branch:
                full_diag_raw = matcher.match_images(query_raw_full, tile_image)
            if use_structure_branch:
                full_diag_struct = matcher.match_images(query_struct_full, tile_struct)
            del tile_image
            if tile_struct is not None:
                del tile_struct
        diagnostics_payload["tiles"].append(
            {
                "row": int(tile["row"]),
                "col": int(tile["col"]),
                "path": str(tile["path"]),
                "score": float(tile.get("score", 0.0)),
                "stage": str(tile.get("stage", "")),
                "full_geometric_score": float(tile.get("full_geometric_score", 0.0)),
                "branches": {
                    "raw": (
                        serialize_branch_match(full_diag_raw)
                        if full_diag_raw is not None
                        else empty_branch_match()
                    ),
                    "structure": (
                        serialize_branch_match(full_diag_struct)
                        if full_diag_struct is not None
                        else empty_branch_match()
                    ),
                },
            }
        )
        if full_diag_raw is not None:
            del full_diag_raw
        if full_diag_struct is not None:
            del full_diag_struct
    (out_dir / "interactive_diagnostics.json").write_text(json.dumps(diagnostics_payload, indent=2))

    (out_dir / "tile_match_summary.json").write_text(json.dumps({"ranked_tiles": final_ranked}, indent=2))
    return final_ranked


def load_cached_tiles(
    out_dir: Path,
    *,
    center_lat: float,
    center_lon: float,
    radius_m: float,
    zoom: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    index_path = out_dir / "index.json"
    if not index_path.exists():
        return None
    payload = json.loads(index_path.read_text())
    meta = payload.get("meta", {})
    tiles = payload.get("tiles", [])
    if not isinstance(tiles, list) or not tiles:
        return None
    if int(meta.get("zoom", -1)) != int(zoom):
        return None
    if abs(float(meta.get("radius_meters", -1.0)) - float(radius_m)) > 1e-6:
        return None
    if abs(float(meta.get("center_lat", 999.0)) - float(center_lat)) > 1e-9:
        return None
    if abs(float(meta.get("center_lon", 999.0)) - float(center_lon)) > 1e-9:
        return None
    expected_tiles_per_side = compute_expected_tiles_per_side(center_lat=center_lat, radius_m=radius_m, zoom=zoom)
    expected_count = expected_tiles_per_side * expected_tiles_per_side
    if int(meta.get("tiles_per_side", -1)) != expected_tiles_per_side:
        return None
    if len(tiles) != expected_count:
        return None
    for tile in tiles:
        tile_path = Path(tile.get("path", ""))
        if not tile_path.exists():
            return None
    return tiles, meta


def parse_shortlist_k(value: str) -> int:
    raw = str(value).strip().lower()
    if raw in {"all", "max", "*"}:
        return 0
    try:
        return int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--shortlist-k must be an integer, 0, or 'all'.") from exc


def parse_loftr_input_size(value: str) -> tuple[int, int]:
    raw = str(value).strip().lower().replace(" ", "")
    parts = raw.split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--loftr-input-size must be in WIDTHxHEIGHT format, e.g. 640x480")
    try:
        width = int(parts[0])
        height = int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--loftr-input-size must be in WIDTHxHEIGHT format, e.g. 640x480") from exc
    if width < 160 or height < 120:
        raise argparse.ArgumentTypeError("--loftr-input-size is too small; minimum is 160x120")
    return width, height


def _make_unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    counter = 1
    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def prepare_query_for_run(*, query_image: Path, out_dir: Path, input_root: Path) -> Path:
    out_query_dir = out_dir / "query"
    out_query_dir.mkdir(parents=True, exist_ok=True)
    staged_query = _make_unique_path(out_query_dir / query_image.name)
    if query_image.is_relative_to(input_root):
        shutil.move(str(query_image), str(staged_query))
    else:
        shutil.copy2(str(query_image), str(staged_query))
    return staged_query


def make_run_out_dir(*, base_out_dir: Path, requested_shortlist_k: int, zoom: int) -> Path:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    k_label = "all" if requested_shortlist_k <= 0 else str(requested_shortlist_k)
    run_name = f"run_{timestamp}_k{k_label}_z{int(zoom)}"
    if (base_out_dir / "tile_match_summary.json").exists() or base_out_dir.name.startswith("run_"):
        runs_root = base_out_dir.parent
    else:
        runs_root = base_out_dir
    out_dir = runs_root / run_name
    suffix = 1
    while out_dir.exists():
        out_dir = runs_root / f"{run_name}_{suffix}"
        suffix += 1
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/search_area.current.yaml")
    parser.add_argument("--center-lat", type=float, default=None)
    parser.add_argument("--center-lon", type=float, default=None)
    parser.add_argument("--radius-m", type=float, default=None)
    parser.add_argument("--zoom", type=int, default=20)
    parser.add_argument(
        "--shortlist-k",
        type=parse_shortlist_k,
        default=3,
        help="Number of top full-image LoFTR tiles to verify. Use 0 or 'all' to include every tile.",
    )
    parser.add_argument("--out-dir", default="data/archive")
    parser.add_argument(
        "--fixed-out-dir",
        action="store_true",
        help="Write directly to --out-dir instead of creating a unique timestamped run folder.",
    )
    parser.add_argument(
        "--skip-match-visualizations",
        action="store_true",
        help="Skip expensive top-rank LoFTR match visualization rendering.",
    )
    parser.add_argument(
        "--loftr-branch",
        choices=["raw", "structure", "both"],
        default="both",
        help="LoFTR branch selection. Use 'raw' as fast mode, 'both' for raw+structure fusion.",
    )
    parser.add_argument(
        "--loftr-input-size",
        type=parse_loftr_input_size,
        default=(640, 480),
        help="LoFTR processor input size as WIDTHxHEIGHT (e.g., 640x480). Smaller is faster.",
    )
    parser.add_argument(
        "--disable-ransac",
        action="store_true",
        help="Disable RANSAC homography/inlier estimation in LoFTR.",
    )
    parser.add_argument(
        "--scoring-mode",
        choices=SCORING_MODE_CHOICES,
        default=SCORING_MODE_MATCH_CONF,
        help=(
            "Final shortlist scoring mode: "
            "'match_count_mean_confidence' or "
            "'inlier_count_mean_confidence'."
        ),
    )
    parser.set_defaults(reuse_cached_tiles=True)
    parser.add_argument(
        "--reuse-cached-tiles",
        dest="reuse_cached_tiles",
        action="store_true",
        help="Reuse cached tiles from --out-dir when available (default).",
    )
    parser.add_argument(
        "--no-reuse-cached-tiles",
        dest="reuse_cached_tiles",
        action="store_false",
        help="Always fetch a fresh tile set and overwrite existing tile images.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    project_root = config_path.parent.parent
    config = load_config(config_path)
    center_lat = float(args.center_lat) if args.center_lat is not None else float(config.latitude)
    center_lon = float(args.center_lon) if args.center_lon is not None else float(config.longitude)
    source_query_image = (project_root / config.query_image).resolve()
    if not source_query_image.exists():
        raise FileNotFoundError(f"Missing query image: {source_query_image}")
    radius_m = args.radius_m if args.radius_m is not None else config.radius_meters
    flight_height_hint = recommended_altitude_for_tile_coverage(
        config=config,
        center_lat=center_lat,
        zoom=args.zoom,
    )
    if flight_height_hint is not None:
        print(
            "recommended_drone_altitude_m: "
            f"{flight_height_hint['recommended_altitude_m']:.1f} "
            f"(target_span={flight_height_hint['target_tile_ground_span_m']:.1f}m, "
            f"fov={flight_height_hint['fov_deg']:.1f}deg from {flight_height_hint['fov_source']})"
        )
        if config.drone_altitude_m is not None:
            delta = float(config.drone_altitude_m) - float(flight_height_hint["recommended_altitude_m"])
            print(f"configured_drone_altitude_m: {float(config.drone_altitude_m):.1f} (delta={delta:+.1f}m)")

    api_key = resolve_google_maps_api_key(config.google_maps_api_key, project_root)
    if not api_key:
        raise RuntimeError(
            "Google Maps API key not found. Set `google_maps_api_key` in config "
            "(literal key, `env:VAR_NAME`, or `${VAR_NAME}`) or use GOOGLE_MAPS_API_KEY/.env."
        )

    requested_shortlist_k = int(args.shortlist_k)
    base_out_dir = (project_root / args.out_dir).resolve()
    out_dir = base_out_dir if bool(args.fixed_out_dir) else make_run_out_dir(
        base_out_dir=base_out_dir,
        requested_shortlist_k=requested_shortlist_k,
        zoom=args.zoom,
    )
    if bool(args.fixed_out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = out_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(config_path), str(meta_dir / config_path.name))

    query_image = prepare_query_for_run(
        query_image=source_query_image,
        out_dir=out_dir,
        input_root=(project_root / "data" / "input").resolve(),
    )
    print(f"query_image_staged: {query_image}")
    print(f"source_query_image: {source_query_image}")
    out_dir.mkdir(parents=True, exist_ok=True)
    cached = (
        load_cached_tiles(
            out_dir,
            center_lat=center_lat,
            center_lon=center_lon,
            radius_m=radius_m,
            zoom=args.zoom,
        )
        if args.reuse_cached_tiles
        else None
    )
    if cached is not None:
        print(f"reusing cached tiles from {out_dir}")
        tiles, meta = cached
    else:
        try:
            tiles, meta = fetch_separate_google_tiles(
                center_lat=center_lat,
                center_lon=center_lon,
                radius_m=radius_m,
                zoom=args.zoom,
                api_key=api_key,
                out_dir=out_dir,
            )
        except Exception as exc:
            cached = (
                load_cached_tiles(
                    out_dir,
                    center_lat=center_lat,
                    center_lon=center_lon,
                    radius_m=radius_m,
                    zoom=args.zoom,
                )
                if args.reuse_cached_tiles
                else None
            )
            if cached is None:
                raise
            print(f"tile fetch failed ({exc}); reusing cached tiles from {out_dir}")
            tiles, meta = cached

    effective_shortlist_k = len(tiles) if requested_shortlist_k <= 0 else requested_shortlist_k
    if requested_shortlist_k <= 0:
        print(f"shortlist_k: all ({effective_shortlist_k} tiles)")
    loftr_input_width, loftr_input_height = args.loftr_input_size
    ranked = rank_tiles_coarse_to_fine(
        query_image_path=query_image,
        tiles=tiles,
        out_dir=out_dir,
        shortlist_k=max(1, effective_shortlist_k),
        write_match_visualizations=not args.skip_match_visualizations,
        use_ransac=not bool(args.disable_ransac),
        loftr_branch=str(args.loftr_branch),
        loftr_input_width=int(loftr_input_width),
        loftr_input_height=int(loftr_input_height),
        scoring_mode=str(args.scoring_mode),
    )
    winner = ranked[0] if ranked else None

    (out_dir / "winner.json").write_text(
        json.dumps(
            {
                "query_image": str(query_image),
                "source_query_image": str(source_query_image),
                "query_mode": config.query_mode,
                "drone_altitude_m": config.drone_altitude_m,
                "drone_yaw_deg": config.drone_yaw_deg,
                "camera": {
                    "model": config.camera_model,
                    "fov_horizontal_deg": config.camera_fov_horizontal_deg,
                    "fov_vertical_deg": config.camera_fov_vertical_deg,
                    "fov_diagonal_deg": config.camera_fov_diagonal_deg,
                },
                "flight_height_hint": flight_height_hint,
                "search_center_latitude": center_lat,
                "search_center_longitude": center_lon,
                "radius_meters": radius_m,
                "zoom": int(args.zoom),
                "shortlist_k": int(effective_shortlist_k),
                "meta": meta,
                "ransac_enabled": not bool(args.disable_ransac),
                "loftr_branch": str(args.loftr_branch),
                "loftr_input_size": f"{int(loftr_input_width)}x{int(loftr_input_height)}",
                "scoring_mode": str(args.scoring_mode),
                "winner": winner,
            },
            indent=2,
        )
    )
    print(f"tiles_dir: {out_dir}")
    if winner is not None:
        print(
            "winner: "
            f"row={winner['row']} col={winner['col']} "
            f"full_matches={winner.get('full_match_count', 0)} "
            f"full_inliers={winner.get('full_inlier_count', 0)} "
            f"stage={winner.get('stage', 'unknown')} "
            f"score={winner['score']:.4f}"
        )


if __name__ == "__main__":
    main()
