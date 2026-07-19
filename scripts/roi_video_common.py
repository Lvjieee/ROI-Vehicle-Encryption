"""Shared helpers for the lossless ROI-video encryption demos."""

from __future__ import annotations

import hashlib
import hmac
import os
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_pool_key(key_hex: str | None) -> bytes:
    """Load a 256-bit pool key from --key-hex or ROI_POOL_KEY."""
    value = key_hex or os.environ.get("ROI_POOL_KEY")
    if not value:
        raise ValueError(
            "A 64-hex-character pool key is required via --key-hex "
            "or the ROI_POOL_KEY environment variable."
        )
    value = value.strip()
    if len(value) != 64:
        raise ValueError("The pool key must contain exactly 64 hex characters.")
    try:
        result = bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError("The pool key contains non-hexadecimal characters.") from exc
    if len(result) != 32:
        raise ValueError("The decoded pool key must be 32 bytes.")
    return result


def key_fingerprint(key: bytes) -> str:
    """Return a non-secret identifier used to catch wrong-key mistakes."""
    return hashlib.sha256(key).hexdigest()[:16]


def derive_session_key(base_key: bytes, session_nonce: bytes) -> bytes:
    """Derive a fresh deterministic pool key for one encryption run."""
    return hmac.new(
        base_key,
        b"recoverable-roi-video/pool-session/v1" + session_nonce,
        hashlib.sha256,
    ).digest()


def required_pool_rows(height: int, width: int, depth: int = 3) -> int:
    permutation_rows = (height * width * depth + 5) // 6
    diffusion_rows = (height * width + 1) // 2
    return permutation_rows + diffusion_rows


class PoolAllocator:
    """Allocate non-overlapping pool segments near keyed preferred indices."""

    def __init__(self, pool_size: int):
        if pool_size < 1:
            raise ValueError("Pool size must be positive.")
        self.pool_size = pool_size
        self.intervals: list[tuple[int, int]] = []

    def allocate(self, preferred_start: int, length: int) -> int:
        if length < 1 or length > self.pool_size:
            raise ValueError(
                f"Cannot allocate {length} rows from a {self.pool_size}-row pool."
            )
        maximum_start = self.pool_size - length + 1
        preferred_start = max(1, min(maximum_start, preferred_start))

        gaps: list[tuple[int, int]] = []
        cursor = 1
        for start, end in sorted(self.intervals):
            if cursor < start:
                gaps.append((cursor, start - 1))
            cursor = max(cursor, end + 1)
        if cursor <= self.pool_size:
            gaps.append((cursor, self.pool_size))

        selected: int | None = None
        for gap_start, gap_end in gaps:
            if gap_start <= preferred_start and preferred_start + length - 1 <= gap_end:
                selected = preferred_start
                break
        if selected is None:
            for gap_start, gap_end in gaps:
                if gap_start >= preferred_start and gap_end - gap_start + 1 >= length:
                    selected = gap_start
                    break
        if selected is None:
            for gap_start, gap_end in gaps:
                if gap_end - gap_start + 1 >= length:
                    selected = gap_start
                    break
        if selected is None:
            used = sum(end - start + 1 for start, end in self.intervals)
            raise ValueError(
                f"Key pool exhausted: {used}/{self.pool_size} rows are already "
                f"allocated and the next ROI needs {length} contiguous rows."
            )

        self.intervals.append((selected, selected + length - 1))
        self.intervals.sort()
        return selected


def intersection_area(first: list[int], second: list[int]) -> int:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def merge_overlapping_boxes(boxes: list[list[int]]) -> list[list[int]]:
    """Merge connected overlapping detections into disjoint ROIs."""
    pending = [box[:] for box in boxes]
    merged: list[list[int]] = []
    while pending:
        current = pending.pop(0)
        changed = True
        while changed:
            changed = False
            remaining: list[list[int]] = []
            for candidate in pending:
                if intersection_area(current, candidate) > 0:
                    current = [
                        min(current[0], candidate[0]),
                        min(current[1], candidate[1]),
                        max(current[2], candidate[2]),
                        max(current[3], candidate[3]),
                    ]
                    changed = True
                else:
                    remaining.append(candidate)
            pending = remaining
        merged.append(current)
    return merged


class FrameSource:
    """Read either a video file or a directory of image frames."""

    def __init__(self, source: Path):
        self.source = source
        self.capture: cv2.VideoCapture | None = None
        self.files: list[Path] = []
        self.fps = 25.0

        if source.is_dir():
            self.files = sorted(
                path for path in source.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
            if not self.files:
                raise ValueError(f"No image frames were found in {source}.")
        elif source.is_file():
            self.capture = cv2.VideoCapture(str(source))
            if not self.capture.isOpened():
                raise ValueError(f"OpenCV could not open video {source}.")
            detected_fps = self.capture.get(cv2.CAP_PROP_FPS)
            if detected_fps and np.isfinite(detected_fps):
                self.fps = float(detected_fps)
        else:
            raise ValueError(f"Input source does not exist: {source}")

    def __iter__(self) -> Iterator[tuple[int, np.ndarray, str]]:
        if self.files:
            for index, path in enumerate(self.files):
                frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError(f"OpenCV could not read frame {path}.")
                yield index, frame, path.name
            return

        assert self.capture is not None
        index = 0
        while True:
            ok, frame = self.capture.read()
            if not ok:
                break
            yield index, frame, f"video_frame_{index:06d}"
            index += 1

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()


def create_preview_writer(
    path: Path | None,
    fps: float,
    frame_size: tuple[int, int],
) -> cv2.VideoWriter | None:
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        frame_size,
    )
    if not writer.isOpened():
        raise ValueError(f"Could not create preview video {path}.")
    return writer
