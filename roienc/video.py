"""Video / frame-folder pipeline: YOLO detection, ROI encryption, decryption.

Encryption writes:
    frames/*.png      encrypted frames (lossless; decrypt from these)
    metadata.json     per ROI: bbox and digest; per frame: timings
    encrypted_preview.mp4 (optional, lossy, viewing only)
Decryption uses the stored boxes and does not run the detector again.
"""

import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

from . import chaos
from .cipher import ROUNDS, decrypt_roi, encrypt_roi

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DETECT = dict(conf=0.1, imgsz=960, classes=(0, 1, 2), min_side=10)


def key_fingerprint(key_c):
    return hashlib.sha256(chaos.check_key(key_c).tobytes()).hexdigest()[:16]


def read_frames(source, max_frames=None):
    """Yield (index, BGR frame) from a video file or an image folder."""
    source = Path(source)
    if source.is_dir():
        files = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXT)
        if not files:
            raise ValueError(f"no images in {source}")
        for i, p in enumerate(files[:max_frames]):
            yield i, cv2.imread(str(p), cv2.IMREAD_COLOR)
        return
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError(f"cannot open {source}")
    i = 0
    while max_frames is None or i < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        yield i, frame
        i += 1
    cap.release()


def _fps(source):
    if Path(source).is_dir():
        return 25.0
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    return float(fps)


def merge_overlapping(boxes):
    """Merge overlapping boxes so no two ROIs share pixels."""
    boxes, merged = [b[:] for b in boxes], []
    while boxes:
        cur, grown = boxes.pop(0), True
        while grown:
            grown, rest = False, []
            for b in boxes:
                if min(cur[2], b[2]) > max(cur[0], b[0]) and min(cur[3], b[3]) > max(cur[1], b[1]):
                    cur = [min(cur[0], b[0]), min(cur[1], b[1]), max(cur[2], b[2]), max(cur[3], b[3])]
                    grown = True
                else:
                    rest.append(b)
            boxes = rest
        merged.append(cur)
    return merged


def detect_vehicles(model, frame, conf=DETECT["conf"], imgsz=DETECT["imgsz"],
                    classes=DETECT["classes"], min_side=DETECT["min_side"]):
    h, w = frame.shape[:2]
    boxes = []
    for r in model(frame, conf=conf, imgsz=imgsz, verbose=False):
        for b in r.boxes:
            if int(b.cls[0]) not in classes:
                continue
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            if min(x2 - x1, y2 - y1) >= min_side:
                boxes.append([x1, y1, x2, y2])
    return merge_overlapping(boxes)


def encrypt_frame(frame, boxes, key_c, rounds=ROUNDS):
    out, records = frame.copy(), []
    for x1, y1, x2, y2 in boxes:
        roi = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        cipher, digest = encrypt_roi(roi, key_c, rounds)
        out[y1:y2, x1:x2] = cv2.cvtColor(cipher, cv2.COLOR_RGB2BGR)
        records.append({"bbox": [x1, y1, x2, y2], "digest": digest.hex()})
    return out, records


def decrypt_frame(frame, records, key_c, rounds=ROUNDS, recover=False):
    out = frame.copy()
    for r in records:
        x1, y1, x2, y2 = r["bbox"]
        cipher = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        plain = decrypt_roi(cipher, bytes.fromhex(r["digest"]), key_c, rounds, recover)
        out[y1:y2, x1:x2] = cv2.cvtColor(plain, cv2.COLOR_RGB2BGR)
    return out


def _writer(path, fps, frame):
    return cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                           (frame.shape[1], frame.shape[0]))


def encrypt_video(source, model_path, out_dir, key_c=chaos.DEMO_KEY, rounds=ROUNDS,
                  max_frames=None, preview=False, **detect_kw):
    from ultralytics import YOLO

    out_dir = Path(out_dir)
    if (out_dir / "metadata.json").exists():
        raise FileExistsError(f"{out_dir} already contains an encrypted run")
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))
    fps = _fps(source)
    meta = {"rounds": rounds, "fps": fps, "key_fingerprint": key_fingerprint(key_c), "frames": []}
    writer = None
    for i, frame in read_frames(source, max_frames):
        if i == 0:   # warm-up (model load, numba compile), not timed
            model(frame, imgsz=DETECT["imgsz"], verbose=False)
            encrypt_roi(np.zeros((8, 8, 3), np.uint8), key_c, rounds)
        t0 = time.perf_counter()
        boxes = detect_vehicles(model, frame, **detect_kw)
        t1 = time.perf_counter()
        enc, records = encrypt_frame(frame, boxes, key_c, rounds)
        t2 = time.perf_counter()
        name = f"frame_{i:06d}.png"
        cv2.imwrite(str(out_dir / "frames" / name), enc)
        meta["frames"].append({"index": i, "file": name, "rois": records,
                               "detect_ms": round((t1 - t0) * 1000, 3),
                               "encrypt_ms": round((t2 - t1) * 1000, 3)})
        if preview:
            writer = writer or _writer(out_dir / "encrypted_preview.mp4", fps, enc)
            writer.write(enc)
        print(f"frame {i:5d}  rois {len(records):2d}  detect {(t1 - t0) * 1000:7.2f} ms"
              f"  encrypt {(t2 - t1) * 1000:7.2f} ms")
    if writer:
        writer.release()
    if not meta["frames"]:
        raise RuntimeError("no frames read")
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    det = np.array([f["detect_ms"] for f in meta["frames"]])
    enc = np.array([f["encrypt_ms"] for f in meta["frames"]])
    print(f"{len(det)} frames: detect {det.mean():.2f}±{det.std():.2f} ms, "
          f"encrypt {enc.mean():.2f}±{enc.std():.2f} ms, total {(det + enc).mean():.2f} ms/frame")
    return meta


def decrypt_video(enc_dir, out_dir, key_c=chaos.DEMO_KEY, recover=False, preview=False):
    enc_dir, out_dir = Path(enc_dir), Path(out_dir)
    meta = json.loads((enc_dir / "metadata.json").read_text(encoding="utf-8"))
    if meta["key_fingerprint"] != key_fingerprint(key_c):
        raise ValueError("wrong key")
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)
    writer = None
    for f in meta["frames"]:
        frame = cv2.imread(str(enc_dir / "frames" / f["file"]), cv2.IMREAD_COLOR)
        plain = decrypt_frame(frame, f["rois"], key_c, meta["rounds"], recover)
        cv2.imwrite(str(out_dir / "frames" / f["file"]), plain)
        if preview:
            writer = writer or _writer(out_dir / "decrypted_preview.mp4", meta["fps"], plain)
            writer.write(plain)
    if writer:
        writer.release()
    print(f"decrypted {len(meta['frames'])} frames -> {out_dir / 'frames'}")
