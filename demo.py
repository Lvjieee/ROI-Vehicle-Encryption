"""Demo.

    python demo.py                                  # synthetic ROI
    python demo.py --image car.png --resize 512
    python demo.py --roi-dir rois/                  # statistics over a folder of ROIs
    python demo.py --image car.png --video MVI_39821/ --model yolo26_best.pt

1. encryption / decryption of one ROI
2. entropy, correlation, histogram, NPCR/UACI, key sensitivity
3. cropping and noise, decrypted with and without CCGR
4. figure  <out>/demo.png
5. (with --roi-dir) mean and std over all ROIs in the folder
6. (with --video and --model) detection + encryption + decryption of a video
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image

import roienc
from roienc import chaos, metrics
from roienc.cipher import decrypt_roi, encrypt_roi

NPCR_IDEAL, UACI_IDEAL = 99.6094, 33.4635


def load(path, size):
    if path:
        img = Image.open(path).convert("RGB")
    else:
        y, x = np.mgrid[0:256, 0:256].astype(float)
        r = 128 + 100 * np.sin(x / 23) * np.cos(y / 31)
        g = 128 + 90 * np.cos((x + y) / 37)
        img = Image.fromarray(np.clip(np.dstack([r, g, (r + g) / 2]), 0, 255).astype(np.uint8))
    if size:
        img = img.resize((size, size), Image.BICUBIC)
    return np.asarray(img).copy()


def one_pixel_test(roi, key, T):
    """NPCR/UACI between ciphertexts of roi and roi with pixel (1,1,1) + 1."""
    roi2 = roi.copy()
    roi2[0, 0, 0] = (int(roi2[0, 0, 0]) + 1) % 256
    return metrics.npcr_uaci(encrypt_roi(roi, key, T)[0], encrypt_roi(roi2, key, T)[0])


def dB(x):
    return "   inf" if math.isinf(x) else f"{x:6.2f}"


def single(roi, key, T, out):
    print("\n[1] encryption / decryption")
    encrypt_roi(roi[:8, :8].copy(), key, T)          # numba compile
    t0 = time.perf_counter(); cipher, digest = encrypt_roi(roi, key, T)
    t1 = time.perf_counter(); plain = decrypt_roi(cipher, digest, key, T)
    t2 = time.perf_counter()
    assert np.array_equal(plain, roi)
    print(f"    lossless: yes, encrypt {(t1 - t0) * 1e3:.1f} ms, decrypt {(t2 - t1) * 1e3:.1f} ms")
    print(f"    digest: {digest.hex()}")
    Image.fromarray(cipher).save(out / "cipher.png")

    print("\n[2] security")
    print(f"    entropy           plain {metrics.entropy(roi):.4f}  cipher {metrics.entropy(cipher):.4f}")
    cp, cc = metrics.adjacent_correlation(roi), metrics.adjacent_correlation(cipher)
    for d in "HVD":
        print(f"    correlation {d}     plain {np.round(cp[d], 4)}  cipher {np.round(cc[d], 4)}")
    print(f"    cipher mean {cipher.mean():.2f} (127.5), variance {cipher.var():.1f} (5461.3)")
    n, u = one_pixel_test(roi, key, T)
    print(f"    one-pixel change  NPCR {n:.4f}% ({NPCR_IDEAL})  UACI {u:.4f}% ({UACI_IDEAL})")
    for i in (0, 1, 2, 4):                           # x4, x6 do not feed back: c4, c6 have no effect
        k = key.copy(); k[i] += 1e-13
        n, u = metrics.npcr_uaci(cipher, encrypt_roi(roi, k, T)[0])
        p = metrics.psnr(roi, decrypt_roi(cipher, digest, k, T), gray=False)
        print(f"    c{i + 1} + 1e-13       NPCR {n:.4f}%  UACI {u:.4f}%  wrong-key PSNR {p:.2f} dB")

    print("\n[3] channel damage, PSNR (dB) / SSIM")
    print("    damage        no recovery        CCGR")
    keep = {}
    for name, dmg in [("crop 12.5%", metrics.crop_center(cipher, 0.125)),
                      ("crop 25%", metrics.crop_center(cipher, 0.25)),
                      ("crop 50%", metrics.crop_center(cipher, 0.5)),
                      ("noise 0.05", metrics.salt_and_pepper(cipher, 0.05)),
                      ("noise 0.10", metrics.salt_and_pepper(cipher, 0.10)),
                      ("noise 0.20", metrics.salt_and_pepper(cipher, 0.20))]:
        raw = decrypt_roi(dmg, digest, key, T)
        rep = decrypt_roi(dmg, digest, key, T, recover=True)
        print(f"    {name:<11} {dB(metrics.psnr(roi, raw))} / {metrics.ssim(roi, raw):.4f}"
              f"   {dB(metrics.psnr(roi, rep))} / {metrics.ssim(roi, rep):.4f}")
        if name in ("crop 25%", "noise 0.10"):
            keep[name] = (dmg, raw, rep)
    figure(roi, cipher, keep, out / "demo.png")
    print(f"\n[4] figure: {out / 'demo.png'}")


def figure(roi, cipher, keep, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 5, figsize=(16, 6.5))
    for a in ax.ravel():
        a.axis("off")
    tiles = [(0, 0, roi, "plain"), (0, 1, cipher, "cipher")]
    for row, name in ((0, "crop 25%"), (1, "noise 0.10")):
        dmg, raw, rep = keep[name]
        tiles += [(row, 2, dmg, f"cipher, {name}"), (row, 3, raw, "decrypted"),
                  (row, 4, rep, "decrypted + CCGR")]
    for r, c, img, title in tiles:
        ax[r, c].imshow(img)
        ax[r, c].set_title(title, fontsize=10)
    for c, img, title in ((0, roi, "plain histogram"), (1, cipher, "cipher histogram")):
        a = ax[1, c]
        a.axis("on"); a.set_yticks([]); a.set_title(title, fontsize=10)
        for ch, col in enumerate("rgb"):
            a.hist(img[:, :, ch].ravel(), bins=256, range=(0, 256), color=col, histtype="step", lw=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def batch(folder, size, key, T):
    files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
    print(f"\n[5] {len(files)} ROIs in {folder}, resized to {size}x{size}")
    rows = []
    for f in files:
        roi = load(str(f), size)
        cipher, digest = encrypt_roi(roi, key, T)
        assert np.array_equal(decrypt_roi(cipher, digest, key, T), roi), f.name
        n, u = one_pixel_test(roi, key, T)
        rows.append((metrics.entropy(roi), metrics.entropy(cipher), n, u))
    r = np.array(rows)
    for i, name in enumerate(["entropy plain", "entropy cipher", "NPCR (%)", "UACI (%)"]):
        print(f"    {name:<15} {r[:, i].mean():.4f} ± {r[:, i].std(ddof=1):.4f}")
    print(f"    ideal NPCR {NPCR_IDEAL}, UACI {UACI_IDEAL}; all ROIs decrypted losslessly")


def video_test(src, model, key, T, n, out):
    import cv2
    from roienc import video
    print(f"\n[6] video {src}, first {n} frames")
    meta = video.encrypt_video(src, model, out / "video_enc", key, T, n, preview=True)
    video.decrypt_video(out / "video_enc", out / "video_dec", key)
    orig = dict(video.read_frames(src, n))
    same = all(np.array_equal(cv2.imread(str(out / "video_dec" / "frames" / f["file"])), orig[f["index"]])
               for f in meta["frames"])
    print(f"    {sum(len(f['rois']) for f in meta['frames'])} ROIs; decrypted frames equal input: {same}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image")
    ap.add_argument("--resize", type=int)
    ap.add_argument("--roi-dir")
    ap.add_argument("--key", help='"c1,...,c6"')
    ap.add_argument("--rounds", type=int, default=roienc.ROUNDS)
    ap.add_argument("--video")
    ap.add_argument("--model")
    ap.add_argument("--max-frames", type=int, default=30)
    ap.add_argument("--out", default="demo_output")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    key = chaos.check_key([float(v) for v in a.key.split(",")] if a.key else chaos.DEMO_KEY)
    roi = load(a.image, a.resize)
    print(f"ROI {roi.shape}, key c {key.tolist()}, T = {a.rounds}, numba {roienc.HAVE_NUMBA}")
    single(roi, key, a.rounds, out)
    if a.roi_dir:
        batch(a.roi_dir, a.resize or 512, key, a.rounds)
    if a.video and a.model:
        video_test(a.video, a.model, key, a.rounds, a.max_frames, out)


if __name__ == "__main__":
    main()
