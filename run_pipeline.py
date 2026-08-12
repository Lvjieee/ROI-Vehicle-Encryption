"""Decrypt lossless ROI-video frames produced by video_encrypt.py."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import matlab
import matlab.engine
import numpy as np

from roi_video_common import (
    create_preview_writer,
    derive_session_key,
    key_fingerprint,
    load_pool_key,
)

os.environ.setdefault("MWLOCALE_TRANSLATED", "OFF")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Decrypt PNG frames and metadata from video_encrypt.py."
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--key-hex", help="64 hex characters; or set ROI_POOL_KEY.")
    parser.add_argument(
        "--enable-recovery",
        action="store_true",
        help="Enable experimental CCGR repair for deliberately damaged ciphertext.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Also create a lossy visual preview of recovered frames.",
    )
    parser.add_argument(
        "--matlab-dir",
        type=Path,
        default=repo_root / "matlab",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    secret_key = load_pool_key(args.key_hex)
    input_dir = args.input_dir.resolve()
    metadata_path = input_dir / "metadata.json"
    if not metadata_path.is_file():
        raise ValueError(f"Metadata file does not exist: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format_version") != 1:
        raise ValueError("Unsupported metadata format version.")
    if metadata.get("pool_key_fingerprint") != key_fingerprint(secret_key):
        raise ValueError("The supplied pool key does not match this ciphertext.")
    try:
        session_nonce = bytes.fromhex(metadata["session_nonce"])
    except (KeyError, ValueError) as exc:
        raise ValueError("Metadata contains an invalid session nonce.") from exc
    session_key = derive_session_key(secret_key, session_nonce)
    if not args.matlab_dir.is_dir():
        raise ValueError(f"MATLAB source directory does not exist: {args.matlab_dir}")

    output_dir = args.output_dir.resolve()
    frames_dir = output_dir / "frames"
    if frames_dir.exists() and any(frames_dir.iterdir()):
        raise ValueError(
            f"Output frame directory is not empty: {frames_dir}. "
            "Choose a new directory to avoid mixing runs."
        )
    frames_dir.mkdir(parents=True, exist_ok=True)
    encrypted_frames_dir = input_dir / "frames"

    try:
        engine = matlab.engine.start_matlab("-nodesktop -nosplash")
    except UnicodeDecodeError as exc:
        raise RuntimeError(
            "MATLAB Engine startup returned non-UTF-8 localized output. "
            "Set the Windows environment variable MWLOCALE_TRANSLATED=OFF, "
            "restart the IDE, and try again."
        ) from exc
    preview_writer = None
    try:
        engine.addpath(str(args.matlab_dir.resolve()), nargout=0)
        engine.init_global_pool(
            matlab.uint8(list(session_key)),
            float(metadata["pool_size"]),
            nargout=0,
        )

        for frame_record in metadata["frames"]:
            frame_path = encrypted_frames_dir / frame_record["file"]
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"Could not read encrypted frame {frame_path}.")
            expected_shape = (
                int(frame_record["height"]),
                int(frame_record["width"]),
            )
            if frame.shape[:2] != expected_shape:
                raise ValueError(
                    f"Frame {frame_path.name} has shape {frame.shape[:2]}, "
                    f"expected {expected_shape}."
                )

            recovered = frame.copy()
            for roi_record in frame_record["rois"]:
                x1, y1, x2, y2 = map(int, roi_record["bbox"])
                cipher_roi = frame[y1:y2, x1:x2].copy()
                matlab_roi = matlab.uint8(cipher_roi.tolist())
                plain_mat = engine.decrypt(
                    matlab_roi,
                    float(roi_record["start_idx"]),
                    float(metadata["rounds"]),
                    float(1 if args.enable_recovery else 0),
                    nargout=1,
                )
                plain_roi = np.asarray(plain_mat, dtype=np.uint8)
                if plain_roi.shape != cipher_roi.shape:
                    raise RuntimeError(
                        f"MATLAB returned shape {plain_roi.shape}, "
                        f"expected {cipher_roi.shape}."
                    )
                recovered[y1:y2, x1:x2] = plain_roi

            output_name = frame_record["file"]
            if not cv2.imwrite(str(frames_dir / output_name), recovered):
                raise RuntimeError(f"Could not write recovered frame {output_name}.")

            if preview_writer is None and args.preview:
                preview_writer = create_preview_writer(
                    output_dir / "recovered_preview.mp4",
                    float(metadata.get("fps", 25.0)),
                    (recovered.shape[1], recovered.shape[0]),
                )
            if preview_writer is not None:
                preview_writer.write(recovered)
            print(
                f"frame={int(frame_record['frame_index']):06d} "
                f"rois={len(frame_record['rois']):2d} recovered"
            )
    finally:
        if preview_writer is not None:
            preview_writer.release()
        engine.quit()

    print(f"Recovered PNG frames: {frames_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
