"""CCGR recovery, Eqs. (8)-(9).

1. damage test: more than 1.5 * 2N/256 cipher bytes equal to 0 or 255
2. mask: propagate those bytes through the DFTF wings, then inverse 3D-ATP
3-5. per-channel Laplace fill, then for each damaged channel c and each
   intact reference r: I_c = fill(I_c - I_r) + I_r, averaged over r.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from . import atp


def is_damaged(cipher):
    c = np.asarray(cipher)
    return np.count_nonzero((c == 0) | (c == 255)) > 1.5 * c.size * 2 / 256


def damage_mask(cipher, k1, rounds):
    c = np.asarray(cipher)
    m = ((c == 0) | (c == 255)).flatten(order="F")
    n = m.size
    m_l, m_r = m[: n // 2], m[n // 2:]
    for _ in range(rounds):
        m_r = m_r | (m_l if m_r.size == m_l.size else np.append(m_l, m_l[-1]))
        m_l = m_l | m_r[: m_l.size]
    m = np.concatenate([m_l, m_r]).reshape(c.shape, order="F")
    return atp.unpermute(m.astype(np.uint8), k1) > 0


def regionfill(image, mask):
    """Solve Laplace's equation inside mask, boundary values from outside."""
    img = np.asarray(image, dtype=np.float64).copy()
    ys, xs = np.nonzero(mask)
    n = ys.size
    if n == 0:
        return img
    h, w = img.shape
    idx = -np.ones((h, w), dtype=np.int64)
    idx[ys, xs] = np.arange(n)
    rows, cols, vals = [np.arange(n)], [np.arange(n)], [None]
    deg, rhs = np.zeros(n), np.zeros(n)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ny, nx = ys + dy, xs + dx
        ok = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        deg += ok
        p, q = np.nonzero(ok)[0], idx[ny[ok], nx[ok]]
        inner = q >= 0
        rows.append(p[inner]); cols.append(q[inner]); vals.append(-np.ones(inner.sum()))
        np.add.at(rhs, p[~inner], img[ny[ok][~inner], nx[ok][~inner]])
    vals[0] = deg
    a = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    img[ys, xs] = spla.spsolve(a, rhs)
    return img


def repair(decrypted, mask):
    x = np.asarray(decrypted, dtype=np.float64).copy()
    h, w, d = x.shape
    for c in range(d):
        if mask[:, :, c].any():
            x[:, :, c] = regionfill(x[:, :, c], mask[:, :, c])
    if d == 3:
        for c in range(d):
            m_c = mask[:, :, c]
            if not m_c.any():
                continue
            total, count = np.zeros((h, w)), np.zeros((h, w))
            for r in range(d):
                if r == c:
                    continue
                target = m_c & ~mask[:, :, r]
                if not target.any():
                    continue
                diff = regionfill(x[:, :, c] - x[:, :, r], m_c | mask[:, :, r])
                total[target] += x[:, :, r][target] + diff[target]
                count[target] += 1
            ch = x[:, :, c]
            ch[count > 0] = total[count > 0] / count[count > 0]
    # round half away from zero, as MATLAB does
    return np.clip(np.floor(x + 0.5), 0, 255).astype(np.uint8)
