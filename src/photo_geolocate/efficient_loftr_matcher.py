from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, AutoModelForKeypointMatching


@dataclass
class MatchResult:
    match_count: int
    mean_confidence: float
    inlier_count: int
    inlier_ratio: float
    reprojection_error_px: float
    spread_score: float
    raw_score: float
    score: float
    keypoints0: np.ndarray
    keypoints1: np.ndarray
    matching_scores: np.ndarray


class EfficientLoFTRMatcher:
    def __init__(
        self,
        model_name: str = "zju-community/efficientloftr",
        threshold: float = 0.2,
        use_ransac: bool = True,
        input_width: int = 640,
        input_height: int = 480,
    ) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.threshold = threshold
        self.use_ransac = use_ransac
        self.input_width = int(input_width)
        self.input_height = int(input_height)
        self.processor = AutoImageProcessor.from_pretrained(
            model_name,
            use_fast=False,
            size={"width": self.input_width, "height": self.input_height},
        )
        self.model = AutoModelForKeypointMatching.from_pretrained(model_name).to(self.device).eval()

    def match_images(self, image0: Image.Image, image1: Image.Image) -> MatchResult:
        images = [image0.convert("RGB"), image1.convert("RGB")]
        inputs = self.processor(images, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        image_sizes = [[(images[0].height, images[0].width), (images[1].height, images[1].width)]]
        processed = self.processor.post_process_keypoint_matching(outputs, image_sizes, threshold=self.threshold)[0]

        keypoints0 = processed["keypoints0"].detach().cpu().numpy()
        keypoints1 = processed["keypoints1"].detach().cpu().numpy()
        matching_scores = processed["matching_scores"].detach().cpu().numpy()

        match_count = int(len(keypoints0))
        mean_confidence = float(matching_scores.mean()) if match_count else 0.0
        inlier_count = 0
        inlier_ratio = 0.0
        reprojection_error = 9999.0
        spread_score = 0.0
        raw_score = float(match_count * mean_confidence)

        if self.use_ransac and match_count >= 4:
            homography, inlier_mask = cv2.findHomography(keypoints0.astype(np.float32), keypoints1.astype(np.float32), cv2.RANSAC, 5.0)
            if homography is not None and inlier_mask is not None:
                inlier_mask = inlier_mask.ravel().astype(bool)
                inlier_count = int(inlier_mask.sum())
                if inlier_count >= 4:
                    inlier_ratio = inlier_count / max(match_count, 1)
                    src_inliers = keypoints0[inlier_mask]
                    dst_inliers = keypoints1[inlier_mask]
                    projected = cv2.perspectiveTransform(
                        src_inliers.astype(np.float32).reshape(-1, 1, 2),
                        homography.astype(np.float64),
                    ).reshape(-1, 2)
                    reprojection_error = float(np.linalg.norm(projected - dst_inliers, axis=1).mean())
                    spread_x = float(np.ptp(src_inliers[:, 0])) / max(images[0].width, 1)
                    spread_y = float(np.ptp(src_inliers[:, 1])) / max(images[0].height, 1)
                    spread_score = min(1.0, max(spread_x, spread_y))

        raw_component = min(raw_score / 120.0, 1.0)
        inlier_component = min(inlier_count / 16.0, 1.0)
        ratio_component = min(inlier_ratio / 0.12, 1.0)
        spread_component = spread_score
        reproj_component = 0.0 if reprojection_error >= 9999.0 else 1.0 / (1.0 + (reprojection_error / 4.0))
        score = (
            0.30 * raw_component
            + 0.30 * inlier_component
            + 0.20 * ratio_component
            + 0.10 * spread_component
            + 0.10 * reproj_component
        )
        return MatchResult(
            match_count=match_count,
            mean_confidence=mean_confidence,
            inlier_count=inlier_count,
            inlier_ratio=inlier_ratio,
            reprojection_error_px=reprojection_error,
            spread_score=spread_score,
            raw_score=raw_score,
            score=float(score),
            keypoints0=keypoints0,
            keypoints1=keypoints1,
            matching_scores=matching_scores,
        )

    def visualize_match(self, image0: Image.Image, image1: Image.Image, result: MatchResult, out_path: Path, text: list[str]) -> None:
        left = image0.convert("RGB")
        right = image1.convert("RGB")
        canvas = Image.new("RGB", (left.width + right.width, max(left.height, right.height)), "black")
        canvas.paste(left, (0, 0))
        canvas.paste(right, (left.width, 0))
        draw = ImageDraw.Draw(canvas)
        rng = np.random.default_rng(0)
        if result.match_count:
            sample_size = min(80, result.match_count)
            indices = np.linspace(0, result.match_count - 1, sample_size, dtype=int)
            colors = rng.integers(64, 255, size=(sample_size, 3))
            for idx, color in zip(indices, colors):
                x0, y0 = result.keypoints0[idx]
                x1, y1 = result.keypoints1[idx]
                fill = tuple(int(c) for c in color)
                draw.line((float(x0), float(y0), float(x1 + left.width), float(y1)), fill=fill, width=2)
                draw.ellipse((x0 - 3, y0 - 3, x0 + 3, y0 + 3), fill=fill)
                draw.ellipse((x1 + left.width - 3, y1 - 3, x1 + left.width + 3, y1 + 3), fill=fill)

        text_y = 10
        for line in text:
            draw.rectangle((10, text_y - 2, 500, text_y + 16), fill="black")
            draw.text((14, text_y), line, fill="white")
            text_y += 20
        out_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_path)
