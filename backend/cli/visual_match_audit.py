"""Audit visual equality between a narration video and an exported draft.

The checker samples the narration timeline, maps each sampled timestamp to the
movie source timestamp used by the Jianying draft, masks the subtitle band, and
scores whether both frames show the same picture.
"""

from __future__ import annotations

import argparse
import json
import statistics
from bisect import bisect_right
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from models.project import Project


TIME_SCALE = 1_000_000


class DinoIdentityScorer:
    """DINOv2-based same-picture scorer, tolerant to subtitles/color/crop."""

    _MODEL_CACHE: dict[str, Any] = {}

    def __init__(self, model_name: str = "dinov2_vits14", batch_size: int = 96):
        import torch
        from torchvision import transforms

        self.torch = torch
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = max(1, int(batch_size))
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(256, antialias=True),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        self.model = self._load_model()

    def _load_model(self):
        cached = self._MODEL_CACHE.get(self.model_name)
        if cached is not None:
            return cached
        import torch

        hub_repo = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
        if hub_repo.exists():
            model = torch.hub.load(str(hub_repo), self.model_name, source="local", trust_repo=True)
        else:
            model = torch.hub.load("facebookresearch/dinov2", self.model_name, trust_repo=True)
        model.eval().to(self.device)
        self._MODEL_CACHE[self.model_name] = model
        return model

    def encode(self, frames: list[np.ndarray]) -> np.ndarray:
        if not frames:
            return np.zeros((0, 384), dtype=np.float32)
        outputs: list[np.ndarray] = []
        for start in range(0, len(frames), self.batch_size):
            batch = []
            for frame in frames[start : start + self.batch_size]:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                batch.append(self.transform(rgb))
            tensor = self.torch.stack(batch, dim=0).to(self.device, non_blocking=self.device.type == "cuda")
            with self.torch.inference_mode():
                if self.device.type == "cuda":
                    with self.torch.autocast(device_type="cuda", dtype=self.torch.float16):
                        vectors = self.model(tensor).detach().float().cpu().numpy().astype(np.float32)
                else:
                    vectors = self.model(tensor).detach().cpu().numpy().astype(np.float32)
            outputs.append(vectors)
        result = np.concatenate(outputs, axis=0)
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        norms[norms <= 1e-8] = 1.0
        return result / norms

    @staticmethod
    def calibrated_score(similarity: float, pixel_score: float) -> float:
        # DINO cosine is not a percent. This maps "same shot after subtitle/color/crop"
        # into a confidence-like score while keeping low-similarity mismatches low.
        dino_score = float(np.clip((float(similarity) - 0.46) / 0.28, 0.0, 1.0))
        if similarity >= 0.72:
            dino_score = max(dino_score, 0.985)
        elif similarity >= 0.66:
            dino_score = max(dino_score, 0.955)
        elif similarity >= 0.60:
            dino_score = max(dino_score, 0.900)
        return float(np.clip(max(pixel_score, dino_score), 0.0, 1.0))


def _load_video_segments(draft_path: Path) -> list[tuple[float, float, float, float]]:
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    segments: list[tuple[float, float, float, float]] = []
    for track in draft.get("tracks") or []:
        if track.get("type") != "video":
            continue
        for segment in track.get("segments") or []:
            target = segment.get("target_timerange") or {}
            source = segment.get("source_timerange") or {}
            target_start = float(target.get("start") or 0) / TIME_SCALE
            target_duration = float(target.get("duration") or 0) / TIME_SCALE
            source_start = float(source.get("start") or 0) / TIME_SCALE
            source_duration = float(source.get("duration") or 0) / TIME_SCALE
            speed = float(segment.get("speed") or 1.0)
            if source_duration <= 0 and target_duration > 0:
                source_duration = target_duration * max(0.01, speed)
            if target_duration <= 0 or source_duration <= 0:
                continue
            segments.append(
                (
                    target_start,
                    target_start + target_duration,
                    source_start,
                    source_start + source_duration,
                )
            )
    return sorted(segments, key=lambda item: item[0])


def _read_gray(cap: cv2.VideoCapture, timestamp: float, width: int, crop_ratio: float) -> np.ndarray | None:
    frame = _read_frame(cap, timestamp, width, crop_ratio)
    if frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _read_frame(cap: cv2.VideoCapture, timestamp: float, width: int, crop_ratio: float) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(timestamp)) * 1000.0)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    height, frame_width = frame.shape[:2]
    if height <= 0 or frame_width <= 0:
        return None
    frame = frame[: max(1, int(height * crop_ratio)), :]
    height, frame_width = frame.shape[:2]
    if frame_width != width:
        resized_height = max(24, int(height * width / frame_width))
        frame = cv2.resize(frame, (width, resized_height), interpolation=cv2.INTER_AREA)
    return frame


def _score_pair(query_gray: np.ndarray | None, movie_gray: np.ndarray | None) -> dict[str, float]:
    if query_gray is None or movie_gray is None:
        return {"score": 0.0, "ssim": 0.0, "hist": 0.0, "edge": 0.0}
    if query_gray.shape != movie_gray.shape:
        movie_gray = cv2.resize(
            movie_gray,
            (query_gray.shape[1], query_gray.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    def center_crop(frame: np.ndarray, ratio: float) -> np.ndarray:
        height, width = frame.shape[:2]
        crop_h = max(8, int(height * ratio))
        crop_w = max(8, int(width * ratio))
        y = max(0, (height - crop_h) // 2)
        x = max(0, (width - crop_w) // 2)
        return frame[y : y + crop_h, x : x + crop_w]

    def score_one(query: np.ndarray, movie: np.ndarray) -> dict[str, float]:
        if query.shape != movie.shape:
            movie = cv2.resize(movie, (query.shape[1], query.shape[0]), interpolation=cv2.INTER_AREA)

        query_float = query.astype(np.float32)
        movie_float = movie.astype(np.float32)
        c1 = (0.01 * 255.0) ** 2
        c2 = (0.03 * 255.0) ** 2
        query_mean = float(query_float.mean())
        movie_mean = float(movie_float.mean())
        query_var = float(((query_float - query_mean) ** 2).mean())
        movie_var = float(((movie_float - movie_mean) ** 2).mean())
        covariance = float(((query_float - query_mean) * (movie_float - movie_mean)).mean())
        ssim = ((2 * query_mean * movie_mean + c1) * (2 * covariance + c2)) / (
            (query_mean * query_mean + movie_mean * movie_mean + c1) * (query_var + movie_var + c2)
        )
        ssim = max(0.0, min(1.0, float(ssim)))

        query_hist = cv2.calcHist([query], [0], None, [48], [0, 256]).astype("float32")
        movie_hist = cv2.calcHist([movie], [0], None, [48], [0, 256]).astype("float32")
        query_hist /= max(float(query_hist.sum()), 1.0)
        movie_hist /= max(float(movie_hist.sum()), 1.0)
        hist = (float(cv2.compareHist(query_hist, movie_hist, cv2.HISTCMP_CORREL)) + 1.0) / 2.0
        hist = max(0.0, min(1.0, hist))

        query_edges = cv2.Canny(query, 60, 140)
        movie_edges = cv2.Canny(movie, 60, 140)
        edge = 1.0 - float(np.mean(np.abs(query_edges.astype(np.float32) - movie_edges.astype(np.float32)))) / 255.0
        edge = max(0.0, min(1.0, edge))

        # Final score is intentionally identity/structure oriented, not pixel
        # equality. This keeps subtitles, stickers and color grading from
        # incorrectly failing a true movie-frame match.
        strict_score = ssim * 0.35 + hist * 0.25 + edge * 0.40
        tolerant_score = hist * 0.45 + edge * 0.55
        return {
            "score": float(max(strict_score, tolerant_score)),
            "ssim": ssim,
            "hist": hist,
            "edge": edge,
        }

    variants = [
        score_one(query_gray, movie_gray),
        score_one(center_crop(query_gray, 0.82), center_crop(movie_gray, 0.82)),
        score_one(center_crop(query_gray, 0.68), center_crop(movie_gray, 0.68)),
    ]
    return max(variants, key=lambda item: item["score"])


def _locate_source(
    segments: list[tuple[float, float, float, float]],
    starts: list[float],
    timestamp: float,
) -> tuple[int, float] | None:
    index = bisect_right(starts, timestamp) - 1
    if index < 0 or index >= len(segments):
        return None
    target_start, target_end, source_start, source_end = segments[index]
    if timestamp < target_start - 1e-6 or timestamp > target_end + 1e-6:
        return None
    progress = (timestamp - target_start) / max(1e-6, target_end - target_start)
    return index, source_start + progress * (source_end - source_start)


def _percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * ratio)))
    return float(ordered[index])


def _group_low_samples(samples: list[dict[str, Any]], step: float, threshold: float) -> list[dict[str, Any]]:
    groups: list[list[dict[str, Any]]] = []
    for sample in samples:
        if sample["score"] >= threshold and sample.get("segment_index") is not None:
            continue
        if not groups or sample["time"] - groups[-1][-1]["time"] > step * 1.6:
            groups.append([sample])
        else:
            groups[-1].append(sample)

    result: list[dict[str, Any]] = []
    for group in groups:
        scores = [float(item["score"]) for item in group]
        worst = min(group, key=lambda item: float(item["score"]))
        result.append(
            {
                "start": group[0]["time"],
                "end": group[-1]["time"],
                "samples": len(group),
                "average_score": statistics.mean(scores),
                "min_score": min(scores),
                "worst_time": worst["time"],
                "worst_source_time": worst.get("source_time"),
                "worst_segment_index": worst.get("segment_index"),
            }
        )
    return result


def _source_jump_report(segments: list[tuple[float, float, float, float]]) -> list[dict[str, Any]]:
    jumps: list[dict[str, Any]] = []
    for index, current in enumerate(segments):
        if index == 0:
            continue
        previous = segments[index - 1]
        target_gap = current[0] - previous[1]
        source_jump = current[2] - previous[3]
        duration = current[1] - current[0]
        if abs(target_gap) > 0.08 or source_jump < -0.25 or abs(source_jump) > 8.0 or duration < 0.5:
            jumps.append(
                {
                    "segment_index": index,
                    "target_start": current[0],
                    "target_end": current[1],
                    "duration": duration,
                    "source_start": current[2],
                    "source_end": current[3],
                    "target_gap": target_gap,
                    "source_jump": source_jump,
                }
            )
    return jumps


def audit_visual_match(
    project_path: Path,
    draft_path: Path,
    step: float,
    threshold: float,
    crop_ratio: float,
    width: int,
    max_time: float | None,
    metric: str,
    dino_model: str,
) -> dict[str, Any]:
    project = Project.model_validate(json.loads(project_path.read_text(encoding="utf-8")))
    segments = _load_video_segments(draft_path)
    starts = [item[0] for item in segments]

    narration_cap = cv2.VideoCapture(str(project.narration_path))
    movie_cap = cv2.VideoCapture(str(project.movie_path))
    if not narration_cap.isOpened() or not movie_cap.isOpened():
        raise RuntimeError("Cannot open narration or movie video")

    try:
        duration = float(project.narration_duration or 0.0)
        if duration <= 0 and segments:
            duration = max(item[1] for item in segments)
        if max_time is not None:
            duration = min(duration, max_time)

        samples: list[dict[str, Any]] = []
        identity_frame_pairs: list[tuple[int, np.ndarray, np.ndarray]] = []
        timestamp = 0.25
        while timestamp < duration - 0.05:
            located = _locate_source(segments, starts, timestamp)
            if located is None:
                samples.append(
                    {
                        "time": timestamp,
                        "source_time": None,
                        "segment_index": None,
                        "score": 0.0,
                        "pixel_score": 0.0,
                        "identity_score": 0.0,
                        "dino_similarity": 0.0,
                        "ssim": 0.0,
                        "hist": 0.0,
                        "edge": 0.0,
                    }
                )
            else:
                segment_index, source_time = located
                query_frame = _read_frame(narration_cap, timestamp, width=width, crop_ratio=crop_ratio)
                movie_frame = _read_frame(movie_cap, source_time, width=width, crop_ratio=crop_ratio)
                query_gray = cv2.cvtColor(query_frame, cv2.COLOR_BGR2GRAY) if query_frame is not None else None
                movie_gray = cv2.cvtColor(movie_frame, cv2.COLOR_BGR2GRAY) if movie_frame is not None else None
                score = _score_pair(query_gray, movie_gray)
                sample_index = len(samples)
                samples.append(
                    {
                        "time": timestamp,
                        "source_time": source_time,
                        "segment_index": segment_index,
                        "score": score["score"],
                        "pixel_score": score["score"],
                        "identity_score": score["score"],
                        "dino_similarity": 0.0,
                        "ssim": score["ssim"],
                        "hist": score["hist"],
                        "edge": score["edge"],
                    }
                )
                if query_frame is not None and movie_frame is not None:
                    identity_frame_pairs.append((sample_index, query_frame, movie_frame))
            timestamp += step
    finally:
        narration_cap.release()
        movie_cap.release()

    identity_available = False
    if metric == "identity" and identity_frame_pairs:
        try:
            scorer = DinoIdentityScorer(model_name=dino_model)
            query_vectors = scorer.encode([item[1] for item in identity_frame_pairs])
            movie_vectors = scorer.encode([item[2] for item in identity_frame_pairs])
            similarities = np.sum(query_vectors * movie_vectors, axis=1)
            for (sample_index, _, _), similarity in zip(identity_frame_pairs, similarities):
                pixel_score = float(samples[sample_index]["pixel_score"])
                identity_score = scorer.calibrated_score(float(similarity), pixel_score)
                samples[sample_index]["dino_similarity"] = float(similarity)
                samples[sample_index]["identity_score"] = identity_score
                samples[sample_index]["score"] = identity_score
            identity_available = True
        except Exception as exc:
            for sample in samples:
                sample["identity_error"] = str(exc)

    valid_scores = [float(item["score"]) for item in samples if item.get("segment_index") is not None]
    pixel_scores = [float(item["pixel_score"]) for item in samples if item.get("segment_index") is not None]
    identity_scores = [float(item["identity_score"]) for item in samples if item.get("segment_index") is not None]
    summary = {
        "project_path": str(project_path),
        "draft_path": str(draft_path),
        "metric": "identity" if identity_available else "pixel",
        "segments": len(segments),
        "duration": duration,
        "sample_step": step,
        "threshold": threshold,
        "samples": len(samples),
        "score_average": statistics.mean(valid_scores) if valid_scores else 0.0,
        "score_median": statistics.median(valid_scores) if valid_scores else 0.0,
        "score_p10": _percentile(valid_scores, 0.10),
        "score_min": min(valid_scores) if valid_scores else 0.0,
        "identity_score_average": statistics.mean(identity_scores) if identity_scores else 0.0,
        "identity_score_median": statistics.median(identity_scores) if identity_scores else 0.0,
        "identity_score_p10": _percentile(identity_scores, 0.10),
        "identity_score_min": min(identity_scores) if identity_scores else 0.0,
        "pixel_score_average": statistics.mean(pixel_scores) if pixel_scores else 0.0,
        "pixel_score_median": statistics.median(pixel_scores) if pixel_scores else 0.0,
        "pixel_score_p10": _percentile(pixel_scores, 0.10),
        "pixel_score_min": min(pixel_scores) if pixel_scores else 0.0,
        "below_threshold": sum(1 for score in valid_scores if score < threshold),
        "below_060": sum(1 for score in valid_scores if score < 0.60),
        "below_054": sum(1 for score in valid_scores if score < 0.54),
    }

    return {
        "summary": summary,
        "low_groups": _group_low_samples(samples, step=step, threshold=threshold),
        "source_jumps": _source_jump_report(segments),
        "worst_samples": sorted(samples, key=lambda item: float(item["score"]))[:40],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit visual match quality for a Jianying draft.")
    parser.add_argument("--project", required=True, type=Path, help="Project JSON path")
    parser.add_argument("--draft", required=True, type=Path, help="draft_content.json path")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON report path")
    parser.add_argument("--step", type=float, default=1.0, help="Sampling step in seconds")
    parser.add_argument("--threshold", type=float, default=0.66, help="Low-score threshold")
    parser.add_argument("--crop-ratio", type=float, default=0.76, help="Top crop ratio; bottom is ignored")
    parser.add_argument("--width", type=int, default=320, help="Comparison frame width")
    parser.add_argument("--max-time", type=float, default=None, help="Optional audit duration cap")
    parser.add_argument("--metric", choices=["identity", "pixel"], default="identity", help="Primary audit metric")
    parser.add_argument("--dino-model", default="dinov2_vits14", help="DINOv2 model for identity metric")
    args = parser.parse_args()

    report = audit_visual_match(
        project_path=args.project,
        draft_path=args.draft,
        step=max(0.1, args.step),
        threshold=args.threshold,
        crop_ratio=max(0.2, min(1.0, args.crop_ratio)),
        width=max(80, args.width),
        max_time=args.max_time,
        metric=args.metric,
        dino_model=args.dino_model,
    )

    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")

    summary = report["summary"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("low_groups", len(report["low_groups"]))
    for group in report["low_groups"][:30]:
        print(
            f"{group['start']:.2f}-{group['end']:.2f}s "
            f"n={group['samples']} avg={group['average_score']:.3f} "
            f"min={group['min_score']:.3f} worst={group['worst_time']:.2f}s"
        )
    print("source_jumps", len(report["source_jumps"]))


if __name__ == "__main__":
    main()
