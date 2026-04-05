from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from photo_geolocate.google_tiles import latlon_to_world_px, world_px_to_latlon

def _build_shortlist_tiles(payload: dict) -> list[dict]:
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return []
    ranked_raw = summary.get("ranked_tiles")
    if not isinstance(ranked_raw, list):
        return []
    ranked = [item for item in ranked_raw if isinstance(item, dict)]
    if not ranked:
        return []
    winner = payload.get("winner")
    shortlist_k = int(_safe_float((winner or {}).get("shortlist_k"), 0.0)) if isinstance(winner, dict) else 0
    if shortlist_k <= 0:
        return ranked
    return ranked[: min(shortlist_k, len(ranked))]


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        v = float(value)
    except Exception:
        return default
    if not np.isfinite(v):
        return default
    return v


def _query_image_size(payload: dict) -> tuple[int, int] | None:
    winner = payload.get("winner")
    if not isinstance(winner, dict):
        return None
    raw_path = winner.get("query_image")
    if not isinstance(raw_path, str) or not raw_path:
        return None
    path = Path(raw_path)
    if not path.exists():
        return None
    try:
        with Image.open(path) as image:
            return int(image.width), int(image.height)
    except Exception:
        return None


def _tile_pixel_to_world_xy(*, px_x: float, px_y: float, tile: dict, meta: dict) -> tuple[float, float] | None:
    zoom = int(_safe_float(meta.get("zoom", 20), 20.0))
    tile_world_pixels = _safe_float(meta.get("tile_world_pixels", meta.get("tile_pixels", meta.get("size_px", 640))), 640.0)
    tile_output_pixels = _safe_float(meta.get("tile_output_pixels", tile_world_pixels), tile_world_pixels)
    center_lat = _safe_float(tile.get("center_latitude"), np.nan)
    center_lon = _safe_float(tile.get("center_longitude"), np.nan)
    if not (np.isfinite(center_lat) and np.isfinite(center_lon)):
        return None
    if tile_world_pixels <= 0 or tile_output_pixels <= 0:
        return None
    center_world_x, center_world_y = latlon_to_world_px(center_lat, center_lon, zoom)
    world_x = center_world_x - (tile_world_pixels / 2.0) + ((float(px_x) / tile_output_pixels) * tile_world_pixels)
    world_y = center_world_y - (tile_world_pixels / 2.0) + ((float(px_y) / tile_output_pixels) * tile_world_pixels)
    return float(world_x), float(world_y)


def _world_xy_to_tile_pixel(*, world_x: float, world_y: float, tile: dict, meta: dict) -> tuple[float, float] | None:
    tile_world_pixels = _safe_float(meta.get("tile_world_pixels", meta.get("tile_pixels", meta.get("size_px", 640))), 640.0)
    tile_output_pixels = _safe_float(meta.get("tile_output_pixels", tile_world_pixels), tile_world_pixels)
    center_lat = _safe_float(tile.get("center_latitude"), np.nan)
    center_lon = _safe_float(tile.get("center_longitude"), np.nan)
    zoom = int(_safe_float(meta.get("zoom", 20), 20.0))
    if not (np.isfinite(center_lat) and np.isfinite(center_lon)):
        return None
    if tile_world_pixels <= 0 or tile_output_pixels <= 0:
        return None
    center_world_x, center_world_y = latlon_to_world_px(center_lat, center_lon, zoom)
    left_world_x = center_world_x - (tile_world_pixels / 2.0)
    top_world_y = center_world_y - (tile_world_pixels / 2.0)
    tile_x_px = ((float(world_x) - left_world_x) / tile_world_pixels) * tile_output_pixels
    tile_y_px = ((float(world_y) - top_world_y) / tile_world_pixels) * tile_output_pixels
    return float(tile_x_px), float(tile_y_px)


def _world_xy_to_latlon(world_xy: tuple[float, float] | None, *, meta: dict) -> tuple[float, float] | None:
    if world_xy is None:
        return None
    zoom = int(_safe_float(meta.get("zoom", 20), 20.0))
    lat, lon = world_px_to_latlon(float(world_xy[0]), float(world_xy[1]), zoom)
    if not (np.isfinite(lat) and np.isfinite(lon)):
        return None
    return float(lat), float(lon)


def _estimate_query_midpoint_for_branch(
    *,
    points: list[dict],
    query_size: tuple[int, int],
    tile: dict,
    meta: dict,
) -> dict | None:
    q_w, q_h = query_size
    if q_w <= 0 or q_h <= 0:
        return None
    valid: list[tuple[float, float, float, float, bool]] = []
    for point in points:
        if not isinstance(point, dict):
            continue
        x0 = _safe_float(point.get("x0"), np.nan)
        y0 = _safe_float(point.get("y0"), np.nan)
        x1 = _safe_float(point.get("x1"), np.nan)
        y1 = _safe_float(point.get("y1"), np.nan)
        if not (np.isfinite(x0) and np.isfinite(y0) and np.isfinite(x1) and np.isfinite(y1)):
            continue
        valid.append((x0, y0, x1, y1, bool(point.get("inlier", False))))
    if not valid:
        return None

    inliers = [p for p in valid if p[4]]
    src = np.array([[p[0], p[1]] for p in (inliers if len(inliers) >= 4 else valid)], dtype=np.float32).reshape(-1, 1, 2)
    dst = np.array([[p[2], p[3]] for p in (inliers if len(inliers) >= 4 else valid)], dtype=np.float32).reshape(-1, 1, 2)
    if src.shape[0] >= 4:
        homography, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if homography is not None:
            center_q = np.array([[[q_w / 2.0, q_h / 2.0]]], dtype=np.float32)
            projected = cv2.perspectiveTransform(center_q, homography)[0, 0]
            world_xy = _tile_pixel_to_world_xy(px_x=float(projected[0]), px_y=float(projected[1]), tile=tile, meta=meta)
            latlon = _world_xy_to_latlon(world_xy, meta=meta)
            if latlon is not None:
                ransac_inliers = int(np.count_nonzero(inlier_mask)) if inlier_mask is not None else 0
                return {
                    "latitude": float(latlon[0]),
                    "longitude": float(latlon[1]),
                    "tile_x_px": float(projected[0]),
                    "tile_y_px": float(projected[1]),
                    "method": "homography_center_projection",
                    "points_used": int(src.shape[0]),
                    "points_available": int(len(valid)),
                    "points_inlier_flagged": int(len(inliers)),
                    "ransac_inliers": int(ransac_inliers),
                }

    # Fallback: translate each correspondence by query-center offset and average in world space.
    q_cx = q_w / 2.0
    q_cy = q_h / 2.0
    pool = inliers if inliers else valid
    world_points: list[tuple[float, float]] = []
    for x0, y0, x1, y1, _inlier in pool:
        cx = x1 + (q_cx - x0)
        cy = y1 + (q_cy - y0)
        world_xy = _tile_pixel_to_world_xy(px_x=float(cx), px_y=float(cy), tile=tile, meta=meta)
        if world_xy is not None:
            world_points.append(world_xy)
    if not world_points:
        return None
    avg_world_x = float(np.mean([p[0] for p in world_points]))
    avg_world_y = float(np.mean([p[1] for p in world_points]))
    latlon = _world_xy_to_latlon((avg_world_x, avg_world_y), meta=meta)
    if latlon is None:
        return None
    tile_px = _world_xy_to_tile_pixel(world_x=avg_world_x, world_y=avg_world_y, tile=tile, meta=meta)
    return {
        "latitude": float(latlon[0]),
        "longitude": float(latlon[1]),
        "tile_x_px": float(tile_px[0]) if tile_px is not None else float("nan"),
        "tile_y_px": float(tile_px[1]) if tile_px is not None else float("nan"),
        "method": "inlier_offset_average",
        "points_used": int(len(world_points)),
        "points_available": int(len(valid)),
        "points_inlier_flagged": int(len(inliers)),
        "ransac_inliers": 0,
    }


def _select_best_branch_estimate(estimates: dict[str, dict]) -> dict | None:
    if not estimates:
        return None
    best_name = ""
    best_score = None
    for name, estimate in estimates.items():
        branch_score = _safe_float(estimate.get("branch_score"), 0.0)
        ransac_inliers = _safe_float(estimate.get("ransac_inliers"), 0.0)
        points_used = _safe_float(estimate.get("points_used"), 0.0)
        score_tuple = (branch_score, ransac_inliers, points_used, 1.0 if estimate.get("method") == "homography_center_projection" else 0.0)
        if best_score is None or score_tuple > best_score:
            best_score = score_tuple
            best_name = name
    if not best_name:
        return None
    return estimates.get(best_name)


def _attach_query_midpoint_estimates(payload: dict) -> None:
    query_size = _query_image_size(payload)
    if query_size is None:
        return
    index = payload.get("index")
    if not isinstance(index, dict):
        return
    meta = index.get("meta")
    tiles = index.get("tiles")
    diagnostics = payload.get("diagnostics")
    if not isinstance(meta, dict) or not isinstance(tiles, list) or not isinstance(diagnostics, dict):
        return
    diag_tiles = diagnostics.get("tiles")
    if not isinstance(diag_tiles, list):
        return

    tile_by_key: dict[str, dict] = {}
    for tile in tiles:
        if not isinstance(tile, dict):
            continue
        key = f"{tile.get('row')}|{tile.get('col')}"
        tile_by_key[key] = tile

    diag_estimate_by_key: dict[str, dict] = {}
    diag_branch_estimates_by_key: dict[str, dict[str, dict]] = {}
    for diag in diag_tiles:
        if not isinstance(diag, dict):
            continue
        key = f"{diag.get('row')}|{diag.get('col')}"
        tile_meta = tile_by_key.get(key)
        if tile_meta is None:
            continue
        branches = diag.get("branches")
        if not isinstance(branches, dict):
            continue
        branch_estimates: dict[str, dict] = {}
        for branch_name in ("raw", "structure"):
            branch = branches.get(branch_name)
            if not isinstance(branch, dict):
                continue
            points = branch.get("points")
            if not isinstance(points, list) or not points:
                continue
            estimate = _estimate_query_midpoint_for_branch(points=points, query_size=query_size, tile=tile_meta, meta=meta)
            if estimate is None:
                continue
            estimate["branch"] = branch_name
            estimate["branch_score"] = _safe_float(branch.get("score"), 0.0)
            estimate["branch_match_count"] = int(_safe_float(branch.get("match_count"), 0.0))
            estimate["branch_inlier_count"] = int(_safe_float(branch.get("inlier_count"), 0.0))
            branch_estimates[branch_name] = estimate
        if not branch_estimates:
            continue
        selected = _select_best_branch_estimate(branch_estimates)
        if selected is None:
            continue
        diag["query_center_estimate"] = selected
        diag["query_center_estimates_by_branch"] = branch_estimates
        diag_estimate_by_key[key] = selected
        diag_branch_estimates_by_key[key] = branch_estimates

    def _attach_to_tile_list(items: object) -> None:
        if not isinstance(items, list):
            return
        for item in items:
            if not isinstance(item, dict):
                continue
            key = f"{item.get('row')}|{item.get('col')}"
            estimate = diag_estimate_by_key.get(key)
            if estimate is None:
                continue
            item["query_center_estimate"] = estimate
            item["query_center_estimates_by_branch"] = diag_branch_estimates_by_key.get(key, {})

    summary = payload.get("summary")
    if isinstance(summary, dict):
        _attach_to_tile_list(summary.get("ranked_tiles"))
    _attach_to_tile_list(payload.get("shortlist_tiles"))

    winner = payload.get("winner")
    if isinstance(winner, dict):
        nested_winner = winner.get("winner")
        if isinstance(nested_winner, dict):
            key = f"{nested_winner.get('row')}|{nested_winner.get('col')}"
            if key in diag_estimate_by_key:
                winner["query_center_estimate"] = diag_estimate_by_key[key]
                winner["query_center_estimates_by_branch"] = diag_branch_estimates_by_key.get(key, {})


def load_payload(out_dir: Path) -> dict:
    summary_path = out_dir / "tile_match_summary.json"
    winner_path = out_dir / "winner.json"
    index_path = out_dir / "index.json"
    diagnostics_path = out_dir / "interactive_diagnostics.json"
    for required in (summary_path, winner_path, index_path):
        if not required.exists():
            raise FileNotFoundError(f"Missing required run artifact: {required}")

    payload: dict = {
        "summary": json.loads(summary_path.read_text()),
        "winner": json.loads(winner_path.read_text()),
        "index": json.loads(index_path.read_text()),
        "diagnostics": json.loads(diagnostics_path.read_text()) if diagnostics_path.exists() else {"tiles": []},
    }
    if not isinstance(payload.get("summary"), dict):
        raise ValueError("Invalid summary payload.")
    if not isinstance(payload.get("winner"), dict):
        raise ValueError("Invalid winner payload.")
    if not isinstance(payload.get("index"), dict):
        raise ValueError("Invalid index payload.")

    payload["shortlist_tiles"] = _build_shortlist_tiles(payload)
    _attach_query_midpoint_estimates(payload)
    return payload


def _has_required_run_artifacts(out_dir: Path) -> bool:
    return (
        (out_dir / "tile_match_summary.json").exists()
        and (out_dir / "winner.json").exists()
        and (out_dir / "index.json").exists()
    )


def _is_loadable_run(out_dir: Path) -> bool:
    if not _has_required_run_artifacts(out_dir):
        return False
    try:
        summary = json.loads((out_dir / "tile_match_summary.json").read_text(encoding="utf-8"))
        winner = json.loads((out_dir / "winner.json").read_text(encoding="utf-8"))
        index = json.loads((out_dir / "index.json").read_text(encoding="utf-8"))
    except Exception:
        return False
    return isinstance(summary, dict) and isinstance(winner, dict) and isinstance(index, dict)


def _discover_runs(runs_root: Path) -> list[Path]:
    runs: list[Path] = []
    seen: set[Path] = set()
    if _is_loadable_run(runs_root):
        root_resolved = runs_root.resolve()
        runs.append(root_resolved)
        seen.add(root_resolved)
    if runs_root.exists() and runs_root.is_dir():
        for summary_path in runs_root.rglob("tile_match_summary.json"):
            candidate = summary_path.parent.resolve()
            if candidate in seen:
                continue
            if not _is_loadable_run(candidate):
                continue
            runs.append(candidate)
            seen.add(candidate)
    # Newest first for convenience.
    runs.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
    return runs


def _run_id_from_path(run_dir: Path, runs_root: Path) -> str:
    try:
        return run_dir.resolve().relative_to(runs_root.resolve()).as_posix()
    except Exception:
        return str(run_dir.resolve())


def _resolve_run_dir(
    *,
    run_id: str,
    project_root: Path,
    runs_root: Path,
    default_out_dir: Path,
) -> Path:
    if run_id:
        candidate = Path(run_id)
        if not candidate.is_absolute():
            candidate = (runs_root / candidate).resolve()
        else:
            candidate = candidate.resolve()
        try:
            candidate.relative_to(project_root)
        except ValueError as exc:
            raise ValueError("Run path is outside the project root.") from exc
        if _has_required_run_artifacts(candidate):
            return candidate
        raise FileNotFoundError(f"Run not found: {candidate}")

    if _is_loadable_run(default_out_dir):
        return default_out_dir
    runs = _discover_runs(runs_root)
    if runs:
        return runs[0]
    raise FileNotFoundError(f"No runs found under {runs_root}")


def build_runs_payload(*, project_root: Path, runs_root: Path, default_out_dir: Path) -> dict:
    runs = _discover_runs(runs_root)
    entries: list[dict[str, str]] = []
    for run_dir in runs:
        entries.append(
            {
                "id": _run_id_from_path(run_dir, runs_root),
                "label": run_dir.name,
                "out_dir": str(run_dir),
            }
        )

    default_run_dir: Path | None = None
    if _is_loadable_run(default_out_dir):
        default_run_dir = default_out_dir.resolve()
    elif runs:
        default_run_dir = runs[0]

    default_run_id = _run_id_from_path(default_run_dir, runs_root) if default_run_dir is not None else ""
    return {
        "runs_root": str(runs_root),
        "default_run_id": default_run_id,
        "runs": entries,
    }
