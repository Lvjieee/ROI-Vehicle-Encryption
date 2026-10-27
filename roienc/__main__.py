"""python -m roienc encrypt|decrypt ...

Key c: --key "c1,c2,c3,c4,c5,c6" or env ROIENC_KEY (default: the paper's key).
"""

import argparse
import os
import sys

from . import chaos, video


def _key(text):
    text = text or os.environ.get("ROIENC_KEY")
    if not text:
        print("warning: using the public demo key", file=sys.stderr)
        return chaos.DEMO_KEY
    return chaos.check_key([float(v) for v in text.split(",")])


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m roienc")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("encrypt")
    e.add_argument("--input", required=True, help="video file or image folder")
    e.add_argument("--model", required=True, help="YOLO weights")
    e.add_argument("--out", required=True)
    e.add_argument("--key")
    e.add_argument("--rounds", type=int, default=3)
    e.add_argument("--max-frames", type=int)
    e.add_argument("--conf", type=float, default=video.DETECT["conf"])
    e.add_argument("--imgsz", type=int, default=video.DETECT["imgsz"])
    e.add_argument("--preview", action="store_true", help="also write a lossy mp4")

    d = sub.add_parser("decrypt")
    d.add_argument("--input", required=True, help="output folder of 'encrypt'")
    d.add_argument("--out", required=True)
    d.add_argument("--key")
    d.add_argument("--recover", action="store_true", help="CCGR for damaged ciphertext")
    d.add_argument("--preview", action="store_true")

    a = p.parse_args(argv)
    if a.cmd == "encrypt":
        video.encrypt_video(a.input, a.model, a.out, _key(a.key), a.rounds, a.max_frames,
                            a.preview, conf=a.conf, imgsz=a.imgsz)
    else:
        video.decrypt_video(a.input, a.out, _key(a.key), a.recover, a.preview)


if __name__ == "__main__":
    main()
