"""DFTF diffusion, Eqs. (5)-(7).

The flattened tensor is split into wings L (floor(N/2)) and R. Each round:
    L = (L + [(R + K_R) mod 256][:len L]) mod 256
    R = (R + Pad((L + K_L) mod 256)) mod 256
then K_L and K_R are rotated by 13 and 17. Pad repeats the last element
when N is odd.
"""

import numpy as np


def driving_fields(k2, n):
    """Eq. (5), column by column, then flattened column-major."""
    k = np.asarray(k2, dtype=np.float64)
    if not np.all(np.isfinite(k)):
        raise FloatingPointError("keystream contains NaN or Inf")
    out = np.empty_like(k)
    for c in range(6):
        col = k[:, c]
        hi, lo = col.max(), col.min()
        if hi - lo <= np.spacing(max(abs(hi), abs(lo), 1.0)):
            raise ValueError("constant keystream column")
        norm = (2.0 * col - (hi + lo)) / (hi - lo)
        out[:, c] = np.mod(np.floor(np.mod(np.abs(norm), 1.0) * 1e10), 256)
    out = out.flatten(order="F")
    if out.size < n:
        raise ValueError("keystream too short")
    return out[: n // 2].astype(np.int64), out[n // 2: n].astype(np.int64)


def _pad(v, n):
    return v if v.size == n else np.append(v, v[-1])


def diffuse(tensor, k2, rounds=3):
    t = np.asarray(tensor, dtype=np.int64)
    x = t.flatten(order="F")
    n = x.size
    left, right = x[: n // 2], x[n // 2:]
    k_l, k_r = driving_fields(k2, n)
    for _ in range(rounds):
        left = (left + ((right + k_r) % 256)[: left.size]) % 256
        right = (right + _pad((left + k_l) % 256, right.size)) % 256
        k_l, k_r = np.roll(k_l, 13), np.roll(k_r, 17)
    return np.concatenate([left, right]).reshape(t.shape, order="F").astype(np.uint8)


def undiffuse(cipher, k2, rounds=3):
    t = np.asarray(cipher, dtype=np.int64)
    x = t.flatten(order="F")
    n = x.size
    left, right = x[: n // 2], x[n // 2:]
    k_l, k_r = driving_fields(k2, n)
    fields = []
    for _ in range(rounds):
        fields.append((k_l, k_r))
        k_l, k_r = np.roll(k_l, 13), np.roll(k_r, 17)
    for k_l, k_r in reversed(fields):
        right = (right - _pad((left + k_l) % 256, right.size)) % 256
        left = (left - ((right + k_r) % 256)[: left.size]) % 256
    return np.concatenate([left, right]).reshape(t.shape, order="F").astype(np.uint8)
