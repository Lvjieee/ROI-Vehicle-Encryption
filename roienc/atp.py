"""3D-ATP permutation, Eq. (4).

Shifts along width, height and channel, each depending on the coordinates
of the fibre being shifted:
    M_Y[i,k] = floor(k_h[i] k_d[k] 1e5) mod W
    M_X[j,k] = floor(k_w[j] k_d[k] 1e5) mod H
    M_Z[i,j] = floor(k_h[i] k_w[j] 1e5) mod D
"""

import numpy as np


def shift_fields(k1, h, w, d):
    v = np.asarray(k1, dtype=np.float64).flatten(order="F")
    if v.size < h + w + d:
        raise ValueError("keystream too short")
    v = np.mod(v * 1e7, 1.0)
    k_h, k_w, k_d = v[:h], v[h:h + w], v[h + w:h + w + d]
    m_y = np.mod(np.floor(np.outer(k_h, k_d) * 1e5), w).astype(np.int64)
    m_x = np.mod(np.floor(np.outer(k_w, k_d) * 1e5), h).astype(np.int64)
    m_z = np.mod(np.floor(np.outer(k_h, k_w) * 1e5), d).astype(np.int64)
    return m_y, m_x, m_z


def _grid(h, w, d):
    return np.meshgrid(np.arange(h), np.arange(w), np.arange(d), indexing="ij")


def permute(roi, k1):
    img = np.asarray(roi)
    h, w, d = img.shape
    m_y, m_x, m_z = shift_fields(k1, h, w, d)
    X, Y, Z = _grid(h, w, d)
    s = img[X, (Y + m_y[X, Z]) % w, Z]
    s = s[(X + m_x[Y, Z]) % h, Y, Z]
    return s[X, Y, (Z + m_z[X, Y]) % d]


def unpermute(tensor, k1):
    img = np.asarray(tensor)
    h, w, d = img.shape
    m_y, m_x, m_z = shift_fields(k1, h, w, d)
    X, Y, Z = _grid(h, w, d)
    s = img[X, Y, (Z - m_z[X, Y]) % d]
    s = s[(X - m_x[Y, Z]) % h, Y, Z]
    return s[X, (Y - m_y[X, Z]) % w, Z]
