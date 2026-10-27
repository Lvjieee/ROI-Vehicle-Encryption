"""Encrypt / decrypt one ROI.

roi: H x W x 3 uint8 (RGB). key_c: secret key c (six numbers).
digest: SHA-256 of the plain ROI, sent with the ciphertext (not secret).
"""

import numpy as np

from . import atp, ccgr, chaos, dftf

ROUNDS = 3


def _check(img):
    img = np.asarray(img)
    if img.ndim == 2:
        img = img[:, :, None]
    if img.ndim != 3 or img.dtype != np.uint8:
        raise ValueError("expected an H x W x C uint8 array")
    return np.ascontiguousarray(img)


def encrypt_roi(roi, key_c=chaos.DEMO_KEY, rounds=ROUNDS):
    roi = _check(roi)
    digest = chaos.roi_digest(roi)
    k1, k2 = chaos.keystreams(digest, key_c, roi.shape)
    return dftf.diffuse(atp.permute(roi, k1), k2, rounds), digest


def decrypt_roi(cipher, digest, key_c=chaos.DEMO_KEY, rounds=ROUNDS, recover=False):
    """recover=True runs CCGR when the damage test fires; undamaged input is
    always recovered bit-exactly."""
    cipher = _check(cipher)
    k1, k2 = chaos.keystreams(bytes(digest), key_c, cipher.shape)
    plain = atp.unpermute(dftf.undiffuse(cipher, k2, rounds), k1)
    if recover and ccgr.is_damaged(cipher):
        plain = ccgr.repair(plain, ccgr.damage_mask(cipher, k1, rounds))
    return plain
