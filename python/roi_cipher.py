"""Pure-Python/NumPy port of the MATLAB encryption core.

This module mirrors the MATLAB reference implementation in ``matlab/``
function by function:

===========================  =====================================
MATLAB                       Python
===========================  =====================================
hash.m                       :func:`sha256_digest`
generate_init.m              :func:`generate_init`
rossler.m                    :func:`rossler`
pool_required_length.m       :func:`pool_required_length`
build_shift_fields.m         :func:`build_shift_fields`
permutation.m                :func:`permutation`
de_permutation.m             :func:`de_permutation`
derive_diffusion_keys.m      :func:`derive_diffusion_keys`
myDiffusion.m                :func:`my_diffusion`
myDeDiffusion.m              :func:`my_de_diffusion`
encrypt.m                    :func:`encrypt`
decrypt.m                    :func:`decrypt`
===========================  =====================================

The MATLAB sources are unchanged and remain the reference used for the
results reported in the paper. This port exists so that detection and
encryption can run in a single Python environment.

MATLAB semantics that must be reproduced exactly:

* MATLAB arrays are column-major, so every flatten/reshape uses ``order='F'``.
* MATLAB indexes from 1; ``mod(i-1+s, n)+1`` becomes ``mod(i+s, n)`` here.
* ``hash(image, 'SHA-256')`` hashes the raw uint8 bytes in column-major order.
* ``circshift(v, k)`` shifts towards higher indices, i.e. ``np.roll(v, k)``.
"""

from __future__ import annotations

import hashlib

import numpy as np

__all__ = [
    "sha256_digest",
    "generate_init",
    "rossler",
    "pool_required_length",
    "build_shift_fields",
    "permutation",
    "de_permutation",
    "derive_diffusion_keys",
    "my_diffusion",
    "my_de_diffusion",
    "encrypt",
    "decrypt",
    "DEFAULT_BASE_STATE",
    "DEFAULT_ROUNDS",
    "TRANSIENT_STEPS",
    "STEP_SIZE",
]

DEFAULT_BASE_STATE = np.full(6, 0.01, dtype=np.float64)
DEFAULT_ROUNDS = 3
TRANSIENT_STEPS = 5000
STEP_SIZE = 0.01

_A, _B, _C, _D, _KK = 36.0, 2.8, 28.6, 9.6, 0.0001
_CLAMP = 1e6


# ---------------------------------------------------------------- hashing --

def sha256_digest(image: np.ndarray) -> np.ndarray:
    """SHA-256 over the ROI bytes, mirroring ``hash.m``.

    ``hash.m`` takes ``input_value(:)`` for uint8 input, i.e. the bytes in
    column-major order. Returns the 32 digest bytes as ``uint8``.
    """
    data = np.asarray(image, dtype=np.uint8).flatten(order="F").tobytes()
    return np.frombuffer(hashlib.sha256(data).digest(), dtype=np.uint8)


# ------------------------------------------------------- initial condition --

def generate_init(key_bytes: np.ndarray,
                  base_state: np.ndarray = DEFAULT_BASE_STATE) -> np.ndarray:
    """Map a 256-bit digest onto six finite states (``generate_init.m``).

    The digest is read as eight big-endian 32-bit words, normalised to
    ``[0, 1)`` and mixed; the result perturbs ``base_state``, the pre-shared
    secret key. Security rests on ``base_state``: the digest is only a
    per-image perturbation and need not be concealed.
    """
    key = np.asarray(key_bytes).ravel()
    if key.size != 32:
        raise ValueError(f"key_bytes must hold 32 values, got {key.size}")
    key = np.mod(np.floor(key.astype(np.float64)), 256).astype(np.uint8)

    base = np.asarray(base_state, dtype=np.float64).ravel()
    if base.size != 6:
        raise ValueError(f"base_state must hold 6 values, got {base.size}")

    words = np.empty(8, dtype=np.float64)
    for i in range(8):
        b = key[4 * i:4 * i + 4].astype(np.float64)
        words[i] = b[0] * 2 ** 24 + b[1] * 2 ** 16 + b[2] * 2 ** 8 + b[3]
    u = words / (2 ** 32 - 1)

    mix = np.array([
        (u[0] + u[1]) % 1.0,
        (u[2] + u[3]) % 1.0,
        (u[4] + u[5]) % 1.0,
        (u[6] + u[7]) % 1.0,
        u[0::2].sum() % 1.0,
        u[1::2].sum() % 1.0,
    ], dtype=np.float64)

    state = base + mix
    if not np.all(np.isfinite(state)):
        raise FloatingPointError("key mapping produced a non-finite state")
    return state


# --------------------------------------------------- hyperchaotic dynamics --

def _rhs(y: np.ndarray) -> np.ndarray:
    dx = np.empty(6, dtype=np.float64)
    dx[0] = _A * (y[1] - y[0])
    dx[1] = _C * y[1] - y[0] * y[2] + y[4]
    dx[2] = y[0] * y[1] - _B * y[2]
    dx[3] = y[1] * y[2] - y[3]
    dx[4] = -_KK * y[1]
    dx[5] = y[0] * y[4] - _D * y[5]
    return dx


def rossler(n_steps: int, initial_state: np.ndarray,
            step: float = STEP_SIZE,
            rng: np.random.Generator | None = None) -> np.ndarray:
    """Fixed-step RK4 integration returning an ``(n_steps, 6)`` trajectory.

    Mirrors ``rossler.m`` including its divergence guard: a non-finite or
    out-of-range step restarts from a damped, slightly perturbed state.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    if rng is None:
        rng = np.random.default_rng()

    y = np.zeros((n_steps, 6), dtype=np.float64)
    y[0] = np.asarray(initial_state, dtype=np.float64).ravel()
    h = float(step)

    for i in range(n_steps - 1):
        x = y[i]
        k1 = _rhs(x)
        k2 = _rhs(x + 0.5 * h * k1)
        k3 = _rhs(x + 0.5 * h * k2)
        k4 = _rhs(x + h * k3)
        nxt = x + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        if not np.all(np.isfinite(nxt)) or np.any(np.abs(nxt) > _CLAMP):
            nxt = x * 0.01 + rng.standard_normal(6) * 0.001

        y[i + 1] = nxt
    return y


def pool_required_length(height: int, width: int, depth: int) -> int:
    """Chaotic rows needed for one ROI (``pool_required_length.m``)."""
    permutation_rows = int(np.ceil(height * width * depth / 6))
    diffusion_rows = int(np.ceil(height * width / 2))
    return permutation_rows + diffusion_rows


def _key_stream(digest: np.ndarray, required_length: int,
                base_state: np.ndarray) -> np.ndarray:
    """Integrate the system and drop the transient, as ``encrypt.m`` does."""
    state = generate_init(digest, base_state)
    trajectory = rossler(required_length + TRANSIENT_STEPS, state)
    return trajectory[TRANSIENT_STEPS:]


# --------------------------------------------------------- 3D-ATP shifting --

def build_shift_fields(K: np.ndarray, h: int, w: int, d: int):
    """Three coordinate-dependent shift fields (``build_shift_fields.m``)."""
    values = np.asarray(K, dtype=np.float64).flatten(order="F")
    needed = h + w + d
    if values.size < needed:
        raise ValueError(
            f"permutation requires at least {needed} values, got {values.size}")
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("permutation key stream contains NaN or Inf")

    fractional = np.mod(values * 1e7, 1.0)
    k_h = fractional[:h]
    k_w = fractional[h:h + w]
    k_d = fractional[h + w:h + w + d]

    shift_y = np.mod(np.floor(np.outer(k_h, k_d) * 1e5), w).astype(np.int64)
    shift_x = np.mod(np.floor(np.outer(k_w, k_d) * 1e5), h).astype(np.int64)
    shift_z = np.mod(np.floor(np.outer(k_h, k_w) * 1e5), d).astype(np.int64)
    return shift_y, shift_x, shift_z


def _grids(h: int, w: int, d: int):
    return np.meshgrid(np.arange(h), np.arange(w), np.arange(d), indexing="ij")


def permutation(image: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Forward 3D-ATP: sequential Y, X, then Z fibre shifts."""
    img = np.asarray(image, dtype=np.float64)
    h, w, d = img.shape
    shift_y, shift_x, shift_z = build_shift_fields(K, h, w, d)
    X, Y, Z = _grids(h, w, d)

    stage_y = img[X, np.mod(Y + shift_y[X, Z], w), Z]
    stage_x = stage_y[np.mod(X + shift_x[Y, Z], h), Y, Z]
    return stage_x[X, Y, np.mod(Z + shift_z[X, Y], d)]


def de_permutation(image: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Inverse 3D-ATP: the same fields applied in reverse order Z, X, Y."""
    img = np.asarray(image, dtype=np.float64)
    h, w, d = img.shape
    shift_y, shift_x, shift_z = build_shift_fields(K, h, w, d)
    X, Y, Z = _grids(h, w, d)

    stage_z = img[X, Y, np.mod(Z - shift_z[X, Y], d)]
    stage_x = stage_z[np.mod(X - shift_x[Y, Z], h), Y, Z]
    return stage_x[X, np.mod(Y - shift_y[X, Z], w), Z]


# ------------------------------------------------------------- DFTF layer --

def derive_diffusion_keys(K: np.ndarray, total_length: int):
    """Map chaotic values onto GF(2^8) driving fields.

    Mirrors ``derive_diffusion_keys.m``.
    """
    K = np.asarray(K, dtype=np.float64)
    if K.ndim != 2 or K.shape[1] != 6:
        raise ValueError("K must have six columns")
    if not np.all(np.isfinite(K)):
        raise FloatingPointError("diffusion key stream contains NaN or Inf")

    mapped = np.zeros_like(K)
    for channel in range(6):
        column = K[:, channel]
        maximum, minimum = column.max(), column.min()
        span = maximum - minimum
        if span <= np.spacing(max(abs(maximum), abs(minimum), 1.0)):
            raise ValueError(f"chaotic key column {channel + 1} is constant")
        normalized = (2.0 * column - (maximum + minimum)) / span
        fractional = np.mod(np.abs(normalized), 1.0)
        mapped[:, channel] = np.mod(np.floor(fractional * 1e10), 256)

    raw = mapped.flatten(order="F")
    if raw.size < total_length:
        raise ValueError(
            f"diffusion requires {total_length} bytes, got {raw.size}")

    noise = raw[:total_length]
    left_length = total_length // 2
    return noise[:left_length].copy(), noise[left_length:].copy()


def _extend_injection(injection_left: np.ndarray,
                      right_length: int, left_length: int) -> np.ndarray:
    """Reconcile wing lengths when N is odd, as ``myDiffusion.m`` does."""
    if right_length <= left_length:
        return injection_left
    extended = np.zeros(right_length, dtype=np.float64)
    extended[:left_length] = injection_left
    extended[-1] = injection_left[-1]
    return extended


def my_diffusion(image: np.ndarray, K: np.ndarray,
                 rounds: int) -> np.ndarray:
    """Forward DFTF: two-wing mutual injection over GF(2^8)."""
    img = np.asarray(image, dtype=np.float64)
    h, w, d = img.shape
    sequence = img.flatten(order="F")
    total_length = sequence.size
    left_length = total_length // 2
    right_length = total_length - left_length

    left = sequence[:left_length].copy()
    right = sequence[left_length:].copy()
    key_left, key_right = derive_diffusion_keys(K, total_length)

    for _ in range(rounds):
        injection_right = np.mod(right + key_right, 256)
        left = np.mod(left + injection_right[:left_length], 256)

        injection_left = np.mod(left + key_left, 256)
        injection_left = _extend_injection(
            injection_left, right_length, left_length)
        right = np.mod(right + injection_left, 256)

        key_left = np.roll(key_left, 13)
        key_right = np.roll(key_right, 17)

    joined = np.concatenate([left, right])
    return joined.reshape((h, w, d), order="F").astype(np.uint8)


def my_de_diffusion(cipher: np.ndarray, K: np.ndarray,
                    rounds: int) -> np.ndarray:
    """Inverse DFTF: unwind the rounds, replacing additions by subtractions."""
    img = np.asarray(cipher, dtype=np.float64)
    h, w, d = img.shape
    sequence = img.flatten(order="F")
    total_length = sequence.size
    left_length = total_length // 2
    right_length = total_length - left_length

    left = sequence[:left_length].copy()
    right = sequence[left_length:].copy()
    key_left, key_right = derive_diffusion_keys(K, total_length)

    # Pre-stack the driving fields in forward order so the reverse loop can
    # retrieve the exact field used at each forward round.
    lefts, rights = [], []
    for _ in range(rounds):
        lefts.append(key_left.copy())
        rights.append(key_right.copy())
        key_left = np.roll(key_left, 13)
        key_right = np.roll(key_right, 17)

    for index in range(rounds - 1, -1, -1):
        k_left, k_right = lefts[index], rights[index]

        injection_left = np.mod(left + k_left, 256)
        injection_left = _extend_injection(
            injection_left, right_length, left_length)
        right = np.mod(right - injection_left, 256)

        injection_right = np.mod(right + k_right, 256)
        left = np.mod(left - injection_right[:left_length], 256)

    joined = np.concatenate([left, right])
    return joined.reshape((h, w, d), order="F").astype(np.uint8)


# ------------------------------------------------------------- public API --

def _as_roi(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image)
    if img.ndim == 2:
        img = img[:, :, None]
    if img.ndim != 3:
        raise ValueError("ROI must be 2-D or 3-D")
    return np.ascontiguousarray(img, dtype=np.uint8)


def encrypt(image: np.ndarray, rounds: int = DEFAULT_ROUNDS,
            base_state: np.ndarray = DEFAULT_BASE_STATE):
    """Encrypt one ROI (``encrypt.m``, SHA mode).

    Returns ``(cipher, digest)``. The digest travels with the ciphertext as
    non-secret side information: it cannot be recomputed from the ciphertext,
    yet it is useless to an adversary who does not hold ``base_state``.
    """
    roi = _as_roi(image)
    h, w, d = roi.shape

    permutation_rows = int(np.ceil(h * w * d / 6))
    required_length = pool_required_length(h, w, d)

    digest = sha256_digest(roi)
    key_slice = _key_stream(digest, required_length, base_state)

    scrambled = permutation(roi, key_slice[:permutation_rows])
    cipher = my_diffusion(scrambled, key_slice[permutation_rows:], rounds)
    return cipher, digest


def decrypt(cipher: np.ndarray, digest: np.ndarray,
            rounds: int = DEFAULT_ROUNDS,
            base_state: np.ndarray = DEFAULT_BASE_STATE) -> np.ndarray:
    """Decrypt one ROI from the ciphertext and the transmitted digest."""
    img = _as_roi(cipher)
    h, w, d = img.shape

    permutation_rows = int(np.ceil(h * w * d / 6))
    required_length = pool_required_length(h, w, d)

    key_slice = _key_stream(np.asarray(digest), required_length, base_state)

    undiffused = my_de_diffusion(img, key_slice[permutation_rows:], rounds)
    recovered = de_permutation(undiffused, key_slice[:permutation_rows])
    return np.mod(recovered, 256).astype(np.uint8)
