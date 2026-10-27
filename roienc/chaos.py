"""Key derivation and hyperchaotic keystream, Eqs. (1)-(3).

Floating-point operations follow the same order as matlab/, so both
versions give identical keystreams.
"""

import hashlib

import numpy as np

A, B, C, D, KAPPA = 36.0, 2.8, 28.6, 9.6, 1e-4   # Eq. (1)
STEP = 0.01
TRANSIENT = 5000
LIMIT = 1e12

DEMO_KEY = (0.01, 0.01, 0.01, 0.01, 0.01, 0.01)   # key c used in the paper


def check_key(key_c):
    key = np.asarray(key_c, dtype=np.float64).ravel()
    if key.size != 6 or not np.all(np.isfinite(key)):
        raise ValueError("key c must be six finite numbers")
    return key


def roi_digest(roi):
    """SHA-256 of the ROI bytes in column-major order (same as MATLAB roi(:))."""
    return hashlib.sha256(np.asarray(roi, np.uint8).flatten(order="F").tobytes()).digest()


def initial_state(digest, key_c):
    """Eqs. (2)-(3): big-endian 32-bit words of the digest, scaled to [0, 1],
    mixed and added to the secret key c."""
    if len(digest) != 32:
        raise ValueError("digest must be 32 bytes")
    b = np.frombuffer(digest, np.uint8).astype(np.float64)
    h = np.array([(b[i] * 2.0**24 + b[i + 1] * 2.0**16 + b[i + 2] * 2.0**8 + b[i + 3])
                  / (2.0**32 - 1) for i in range(0, 32, 4)])
    mix = np.mod([h[0] + h[1], h[2] + h[3], h[4] + h[5], h[6] + h[7],
                  ((h[0] + h[2]) + h[4]) + h[6], ((h[1] + h[3]) + h[5]) + h[7]], 1.0)
    return check_key(key_c) + mix


def _rk4_python(n, x0, h):
    def f(y):
        return (A * (y[1] - y[0]), C * y[1] - y[0] * y[2] + y[4], y[0] * y[1] - B * y[2],
                y[1] * y[2] - y[3], -KAPPA * y[1], y[0] * y[4] - D * y[5])

    out = np.empty((n, 6))
    x = [float(v) for v in x0]
    out[0] = x
    for i in range(1, n):
        k1 = f(x)
        k2 = f([x[j] + 0.5 * h * k1[j] for j in range(6)])
        k3 = f([x[j] + 0.5 * h * k2[j] for j in range(6)])
        k4 = f([x[j] + h * k3[j] for j in range(6)])
        x = [x[j] + h / 6.0 * (((k1[j] + 2.0 * k2[j]) + 2.0 * k3[j]) + k4[j]) for j in range(6)]
        if not all(abs(v) <= LIMIT for v in x):
            raise FloatingPointError(f"integration diverged at step {i}")
        out[i] = x
    return out


try:
    import numba

    @numba.njit(cache=True)
    def _rk4_numba(n, x0, h):
        out = np.empty((n, 6))
        x = x0.copy()
        out[0] = x
        k = np.empty((4, 6))
        y = np.empty(6)
        for i in range(1, n):
            for s in range(4):
                for j in range(6):
                    if s == 0:
                        y[j] = x[j]
                    elif s == 3:
                        y[j] = x[j] + h * k[2, j]
                    else:
                        y[j] = x[j] + 0.5 * h * k[s - 1, j]
                k[s, 0] = A * (y[1] - y[0])
                k[s, 1] = C * y[1] - y[0] * y[2] + y[4]
                k[s, 2] = y[0] * y[1] - B * y[2]
                k[s, 3] = y[1] * y[2] - y[3]
                k[s, 4] = -KAPPA * y[1]
                k[s, 5] = y[0] * y[4] - D * y[5]
            for j in range(6):
                x[j] = x[j] + h / 6.0 * (((k[0, j] + 2.0 * k[1, j]) + 2.0 * k[2, j]) + k[3, j])
                if not abs(x[j]) <= LIMIT:
                    raise FloatingPointError("integration diverged")
            out[i] = x
        return out

    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False


def integrate(x0, n, h=STEP):
    """n RK4 steps of Eq. (1); returns an (n, 6) trajectory."""
    x0 = np.asarray(x0, dtype=np.float64).ravel()
    return _rk4_numba(int(n), x0, float(h)) if HAVE_NUMBA else _rk4_python(int(n), x0, float(h))


def keystreams(digest, key_c, shape):
    """K1 (ceil(HWD/6) rows, for 3D-ATP) and K2 (ceil(HW/2) rows, for DFTF)."""
    h, w, d = shape
    r1, r2 = -(-h * w * d // 6), -(-h * w // 2)
    k = integrate(initial_state(digest, key_c), r1 + r2 + TRANSIENT)[TRANSIENT:]
    return k[:r1], k[r1:]
