"""Self-checks for the Python port of the encryption core.

Usage::

    python python/run_tests.py [image_path]

The checks mirror those in ``tests/run_tests.m``: round-trip exactness,
ciphertext entropy, differential metrics (NPCR/UACI), adjacent-pixel
decorrelation, and key sensitivity. Entropy and NPCR are resolution
dependent and approach their theoretical optima as the patch grows, so the
thresholds below are deliberately loose enough for small test patches.
"""

from __future__ import annotations

import sys

import numpy as np

from roi_cipher import DEFAULT_ROUNDS, decrypt, encrypt, sha256_digest

THEORETICAL_NPCR = 99.6094
THEORETICAL_UACI = 33.4635


def entropy(data: np.ndarray) -> float:
    counts = np.bincount(np.asarray(data).ravel(), minlength=256).astype(float)
    probabilities = counts / counts.sum()
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log2(probabilities)).sum())


def horizontal_correlation(data: np.ndarray) -> float:
    plane = np.asarray(data)[:, :, 0].astype(float)
    return float(np.corrcoef(plane[:, :-1].ravel(), plane[:, 1:].ravel())[0, 1])


def load_patch(path: str | None, size: int = 256) -> np.ndarray:
    if path:
        from PIL import Image
        return np.array(Image.open(path).convert("RGB"))
    rows, columns = np.mgrid[0:size, 0:size]
    base = ((columns // 8 * 32 + rows // 8 * 16) % 256).astype(np.uint8)
    return np.dstack([base, np.roll(base, 5, 0), np.roll(base, 9, 1)])


def main() -> int:
    patch = load_patch(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"test patch: {patch.shape[0]}x{patch.shape[1]}x{patch.shape[2]}")

    cipher, digest = encrypt(patch, DEFAULT_ROUNDS)
    recovered = decrypt(cipher, digest, DEFAULT_ROUNDS)
    passed = True

    exact = np.array_equal(recovered, patch)
    print(f"[1] round-trip bit-exact   : {exact}")
    passed &= exact

    value = entropy(cipher)
    print(f"[2] ciphertext entropy     : {value:.4f}  (ideal 8.0000)")
    passed &= value > 7.99

    flipped = patch.copy()
    row, column = patch.shape[0] // 2, patch.shape[1] // 2
    flipped[row, column, 0] = (int(flipped[row, column, 0]) + 1) % 256
    cipher_flipped, _ = encrypt(flipped, DEFAULT_ROUNDS)

    npcr = float(np.mean(cipher != cipher_flipped) * 100)
    uaci = float(np.mean(
        np.abs(cipher.astype(float) - cipher_flipped.astype(float)) / 255) * 100)
    print(f"[3] NPCR                   : {npcr:.4f}%  (theory {THEORETICAL_NPCR})")
    print(f"    UACI                   : {uaci:.4f}%  (theory {THEORETICAL_UACI})")
    passed &= npcr > 99.4 and 33.0 < uaci < 34.0

    before = horizontal_correlation(patch)
    after = horizontal_correlation(cipher)
    print(f"[4] horizontal correlation : {before:.4f} -> {after:.4f}")
    passed &= abs(after) < 0.05

    # A one-pixel change avalanches the whole digest, so the realistic key
    # sensitivity test decrypts under the digest of a different ROI.
    other = ((patch.astype(int) + 37) % 256).astype(np.uint8)
    _, wrong_digest = encrypt(other, DEFAULT_ROUNDS)
    wrong = decrypt(cipher, wrong_digest, DEFAULT_ROUNDS)
    failed = not np.array_equal(wrong, patch)
    mae = float(np.abs(wrong.astype(int) - patch.astype(int)).mean())
    print(f"[5] wrong digest fails     : {failed}  (MAE {mae:.1f})")
    passed &= failed

    changed = int((sha256_digest(flipped) != digest).sum())
    print(f"[6] digest avalanche       : {changed}/32 bytes differ")
    passed &= changed > 24

    print("\nRESULT:", "all checks passed" if passed else "SOME CHECKS FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
