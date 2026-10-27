"""Metrics and damage models used in the experiments."""

import math

import numpy as np


def entropy(img):
    """Mean Shannon entropy of the channels, Eq. (12)."""
    out = []
    for c in range(img.shape[2]):
        p = np.bincount(img[:, :, c].ravel(), minlength=256) / img[:, :, c].size
        p = p[p > 0]
        out.append(-(p * np.log2(p)).sum())
    return float(np.mean(out))


def adjacent_correlation(img, pairs=3000, seed=0):
    """Eq. (11) on random adjacent pairs; {'H'|'V'|'D': [r_R, r_G, r_B]}."""
    rng = np.random.default_rng(seed)
    h, w, _ = img.shape
    out = {}
    for name, (dy, dx) in (("H", (0, 1)), ("V", (1, 0)), ("D", (1, 1))):
        ys, xs = rng.integers(0, h - dy, pairs), rng.integers(0, w - dx, pairs)
        out[name] = [float(np.corrcoef(img[ys, xs, c].astype(float),
                                       img[ys + dy, xs + dx, c].astype(float))[0, 1])
                     for c in range(img.shape[2])]
    return out


def npcr_uaci(c1, c2):
    """Eqs. (13)-(14), in percent."""
    a, b = c1.astype(np.float64), c2.astype(np.float64)
    return float(np.mean(a != b) * 100), float(np.mean(np.abs(a - b)) / 255 * 100)


def _gray(img):
    # MATLAB rgb2gray
    g = 0.298936021293775 * img[:, :, 0] + 0.587043074451121 * img[:, :, 1] \
        + 0.114020904255103 * img[:, :, 2]
    return np.clip(np.floor(g + 0.5), 0, 255).astype(np.uint8)


def psnr(ref, test, gray=True):
    """gray=True as in the robustness tests, gray=False (RGB) as in key sensitivity."""
    if gray:
        ref, test = _gray(ref), _gray(test)
    mse = float(np.mean((ref.astype(float) - test.astype(float)) ** 2))
    return math.inf if mse == 0 else 10 * math.log10(255 ** 2 / mse)


def ssim(ref, test):
    from skimage.metrics import structural_similarity
    return float(structural_similarity(_gray(test), _gray(ref), data_range=255,
                                       gaussian_weights=True, sigma=1.5,
                                       use_sample_covariance=False))


def crop_center(cipher, ratio):
    """Zero a centred block covering `ratio` of the image."""
    out = cipher.copy()
    h, w = out.shape[:2]
    ch, cw = int(h * math.sqrt(ratio)), int(w * math.sqrt(ratio))
    y0, x0 = int(h / 2 - ch / 2), int(w / 2 - cw / 2)
    out[y0:y0 + ch, x0:x0 + cw] = 0
    return out


def salt_and_pepper(cipher, density, seed=0):
    out = cipher.copy()
    u = np.random.default_rng(seed).random(out.shape)
    out[u < density / 2] = 0
    out[(u >= density / 2) & (u < density)] = 255
    return out
