"""Encrypt detected vehicle ROIs in a video or image-frame directory.

The decryptable artifact is the lossless PNG frame directory plus
metadata.json. The optional MP4 is a lossy visual preview only.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import secrets
import struct
import sys
import time
from pathlib import Path

import cv2
import matlab
import matlab.engine
import numpy as np
from ultralytics import YOLO

from roi_video_common import (
    FrameSource,
    PoolAllocator,
    create_preview_writer,
    derive_session_key,
    key_fingerprint,
    load_pool_key,
    merge_overlapping_boxes,
    required_pool_rows,
)

os.environ.setdefault("MWLOCALE_TRANSLATED", "OFF")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Losslessly encrypt YOLO-detected vehicle ROIs."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--key-hex", help="64 hex characters; or set ROI_POOL_KEY.")
    parser.add_argument("--pool-size", type=int, default=1_000_000)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--min-width", type=int, default=10)
    parser.add_argument("--min-height", type=int, default=10)
    parser.add_argument("--classes", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--matlab-dir",
        type=Path,
        default=repo_root / "matlab",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Also create a lossy MP4 preview that cannot be decrypted.",
    )
    return parser.parse_args()


def derive_start_index(
    session_key: bytes,
    frame_index: int,
    roi_index: int,
    roi: np.ndarray,
    pool_size: int,
) -> tuple[int, int]:
    required = required_pool_rows(*roi.shape)
    maximum_start = pool_size - required + 1
    if maximum_start < 1:
        raise ValueError(
            f"ROI {roi.shape} requires {required} pool rows, "
            f"but pool size is only {pool_size}."
        )
    header = struct.pack(">III", frame_index, roi_index, required)
    digest = hmac.new(
        session_key,
        header + roi.tobytes(order="C"),
        hashlib.sha256,
    ).digest()
    start_index = int.from_bytes(digest[:8], "big") % maximum_start + 1
    return start_index, required


def extract_boxes(
    result,
    frame_width: int,
    frame_height: int,
    allowed_classes: set[int],
    minimum_width: int,
    minimum_height: int,
) -> list[list[int]]:
    boxes: list[list[int]] = []
    if result.boxes is None:
        return boxes

    for detection in result.boxes:
        class_id = int(detection.cls[0])
        if class_id not in allowed_classes:
            continue
        x1, y1, x2, y2 = map(int, detection.xyxy[0].tolist())
        x1 = max(0, min(frame_width, x1))
        x2 = max(0, min(frame_width, x2))
        y1 = max(0, min(frame_height, y1))
        y2 = max(0, min(frame_height, y2))
        if x2 - x1 < minimum_width or y2 - y1 < minimum_height:
            continue
        boxes.append([x1, y1, x2, y2])
    return merge_overlapping_boxes(boxes)


def main() -> int:
    args = parse_args()
    base_key = load_pool_key(args.key_hex)
    session_nonce = secrets.token_bytes(16)
    session_key = derive_session_key(base_key, session_nonce)
    if args.pool_size < 1 or args.rounds < 0:
        raise ValueError("--pool-size must be positive and --rounds nonnegative.")
    if not args.model.is_file():
        raise ValueError(f"Model weights do not exist: {args.model}")
    if not args.matlab_dir.is_dir():
        raise ValueError(f"MATLAB source directory does not exist: {args.matlab_dir}")

    output_dir = args.output_dir.resolve()
    frames_dir = output_dir / "frames"
    if (output_dir / "metadata.json").exists() or (
        frames_dir.exists() and any(frames_dir.iterdir())
    ):
        raise ValueError(
            f"Output directory already contains an encryption run: {output_dir}. "
            "Choose a new directory to avoid mixing metadata and frames."
        )
    frames_dir.mkdir(parents=True, exist_ok=True)

    source = FrameSource(args.input.resolve())
    model = YOLO(str(args.model.resolve()))
    try:
        engine = matlab.engine.start_matlab("-nodesktop -nosplash")
    except UnicodeDecodeError as exc:
        raise RuntimeError(
            "MATLAB Engine startup returned non-UTF-8 localized output. "
            "Set the Windows environment variable MWLOCALE_TRANSLATED=OFF, "
            "restart the IDE, and try again."
        ) from exc
    preview_writer = None

    metadata: dict = {
        "format_version": 1,
        "cipher_storage": "lossless PNG frames",
        "source": str(args.input),
        "model": args.model.name,
        "pool_size": args.pool_size,
        "pool_key_fingerprint": key_fingerprint(base_key),
        "session_nonce": session_nonce.hex(),
        "rounds": args.rounds,
        "fps": source.fps,
        "frames": [],
    }

    try:
        engine.addpath(str(args.matlab_dir.resolve()), nargout=0)
        engine.init_global_pool(
            matlab.uint8(list(session_key)),
            float(args.pool_size),
            nargout=0,
        )

        processed = 0
        allocator = PoolAllocator(args.pool_size)
        for frame_index, frame, source_name in source:
            if args.max_frames is not None and processed >= args.max_frames:
                break
            frame_timer = time.perf_counter()
            frame_height, frame_width = frame.shape[:2]

            if preview_writer is None and args.preview:
                preview_writer = create_preview_writer(
                    output_dir / "encrypted_preview.mp4",
                    source.fps,
                    (frame_width, frame_height),
                )

            detection_start = time.perf_counter()
            result = model.predict(
                frame,
                conf=args.confidence,
                imgsz=args.imgsz,
                verbose=False,
            )[0]
            detection_ms = (time.perf_counter() - detection_start) * 1000

            boxes = extract_boxes(
                result,
                frame_width,
                frame_height,
                set(args.classes),
                args.min_width,
                args.min_height,
            )

            encrypted_frame = frame.copy()
            roi_records = []
            encryption_ms = 0.0
            for roi_index, (x1, y1, x2, y2) in enumerate(boxes):
                # Always crop plaintext from the untouched original frame.
                roi = frame[y1:y2, x1:x2].copy()
                preferred_index, required = derive_start_index(
                    session_key,
                    frame_index,
                    roi_index,
                    roi,
                    args.pool_size,
                )
                start_index = allocator.allocate(preferred_index, required)
                matlab_roi = matlab.uint8(roi.tolist())

                call_start = time.perf_counter()
                _, cipher_mat, used_index, matlab_seconds, _, _ = engine.encrypt(
                    matlab_roi,
                    float(args.rounds),
                    float(start_index),
                    nargout=6,
                )
                call_ms = (time.perf_counter() - call_start) * 1000
                cipher = np.asarray(cipher_mat, dtype=np.uint8)
                if cipher.shape != roi.shape:
                    raise RuntimeError(
                        f"MATLAB returned shape {cipher.shape}, expected {roi.shape}."
                    )
                if int(used_index) != start_index:
                    raise RuntimeError("MATLAB returned an unexpected pool index.")

                encrypted_frame[y1:y2, x1:x2] = cipher
                encryption_ms += call_ms
                roi_records.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "start_idx": start_index,
                        "preferred_start_idx": preferred_index,
                        "required_pool_rows": required,
                        "matlab_seconds": float(matlab_seconds),
                        "engine_call_ms": call_ms,
                    }
                )

            frame_name = f"frame_{frame_index:06d}.png"
            if not cv2.imwrite(str(frames_dir / frame_name), encrypted_frame):
                raise RuntimeError(f"Could not write encrypted frame {frame_name}.")
            if preview_writer is not None:
                preview_writer.write(encrypted_frame)

            total_ms = (time.perf_counter() - frame_timer) * 1000
            metadata["frames"].append(
                {
                    "frame_index": frame_index,
                    "source_name": source_name,
                    "file": frame_name,
                    "height": frame_height,
                    "width": frame_width,
                    "detection_ms": detection_ms,
                    "encryption_ms": encryption_ms,
                    "total_ms": total_ms,
                    "rois": roi_records,
                }
            )
            processed += 1
            print(
                f"frame={frame_index:06d} rois={len(roi_records):2d} "
                f"detect={detection_ms:8.2f}ms "
                f"encrypt={encryption_ms:8.2f}ms total={total_ms:8.2f}ms"
            )
    finally:
        source.close()
        if preview_writer is not None:
            preview_writer.release()
        engine.quit()

    if not metadata["frames"]:
        raise RuntimeError("No frames were processed.")

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Decryptable PNG frames: {frames_dir}")
    print(f"Decryption metadata:     {metadata_path}")
    if args.preview:
        print("Warning: encrypted_preview.mp4 is lossy and is not decryptable.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
