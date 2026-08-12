# Recoverable ROI Vehicle Video Encryption

Research implementation of frame-wise vehicle ROI encryption for Internet
of Vehicles experiments. YOLO detects vehicle regions, the 3D-ATP
permutation and DFTF transform encrypt them, and the decryptor reconstructs
the ROIs from lossless encrypted frames and explicit metadata.

Two equivalent implementations of the encryption core are provided. The
MATLAB code in `matlab/` is the reference used for the reported results; the
pure-Python port in `python/` implements the same algorithm so that
detection and encryption can run in a single environment without the MATLAB
Engine.

> **Research prototype only.** This repository is not production
> cryptographic software and should not be used to protect real operational
> or personal data.

## What this repository implements

- Frame-wise vehicle detection with Ultralytics YOLO.
- ROI-only encryption using a deterministic secret-key-derived chaotic pool.
- Lossless encrypted PNG frames suitable for exact decryption.
- JSON metadata containing frame indices, bounding boxes, ROI dimensions,
  and the pool start index used for each ROI.
- Exact standard decryption and an optional experimental CCGR recovery mode.
- Per-frame detection, MATLAB-call, and total timing records.

The algorithm processes video as a sequence of image frames. It is not a
compressed-domain H.264/H.265 cipher.

## Repository layout

```text
matlab/                 MATLAB encryption, decryption, and key-pool code
python/roi_cipher.py    Pure-Python port of the same encryption core
scripts/video_encrypt.py
scripts/video_decrypt.py
scripts/roi_video_common.py
```

The original development folder, datasets, model weights, paper figures,
temporary scripts, tests, and generated results are intentionally not included.

## Requirements

- Python 3.11 or another version supported by your MATLAB release.
- MATLAB R2023b or newer.
- MATLAB Engine API for Python.
- MATLAB Image Processing Toolbox only when `--enable-recovery` is used.
- A compatible Ultralytics YOLO weights file fine-tuned for the target
  vehicle classes.

Install Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Install the MATLAB Engine API using the instructions for the locally
installed MATLAB release. Verify the installation with:

```bash
python -c "import matlab.engine; print('MATLAB Engine available')"
```

### Windows / PyCharm

On the validated Windows machine, select this interpreter in PyCharm:

```text
C:\Users\86136\AppData\Local\Programs\Python\Python311\python.exe
```

First mark the project as trusted, then run:

```bash
python scripts/check_environment.py
```

To avoid entering command-line arguments in an IDE, run
`scripts/run_pipeline.py` once. It creates an ignored local `config.json`
from `config.example.json`. Edit the paths, key, and mode, then run
`run_pipeline.py` again.

Do not run `video_decrypt.py` before encryption. It requires the
`metadata.json` and lossless PNG frames produced by `video_encrypt.py`.

On localized Windows systems, if MATLAB Engine raises a UTF-8 decoding
error, set the user environment variable below and restart PyCharm:

```text
MWLOCALE_TRANSLATED=OFF
```

## Secret pool key

Encryption and decryption must use the same 256-bit pool key. Generate a
random development key:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Provide it through `--key-hex` or, preferably, the `ROI_POOL_KEY`
environment variable. Do not commit a real key to Git.

The key pool is an optimization and is not itself a substitute for a secret
key. The repository stores only a short non-secret key fingerprint in
`metadata.json` to detect accidental wrong-key use. Each encryption run also
creates a random public session nonce, derives a fresh session pool key, and
allocates non-overlapping pool segments to ROIs. The keyed preferred segment
for an ROI is bound to its frame index, ROI index, dimensions, and plaintext
bytes through HMAC-SHA-256; the final collision-free index is stored as
metadata for decryption.

## Encrypt a video

```bash
python scripts/video_encrypt.py \
  --input path/to/input.mp4 \
  --model path/to/yolo26_best.pt \
  --output-dir runs/encrypted \
  --pool-size 1000000 \
  --rounds 3 \
  --classes 0 1 2 \
  --preview
```

`--input` may also point to a directory of image frames. The decryptable
artifacts are:

```text
runs/encrypted/frames/*.png
runs/encrypted/metadata.json
```

`encrypted_preview.mp4` is optional and lossy. It is for visual inspection
only and **must not** be used as decryption input.

For the 30-frame timing experiment:

```bash
python scripts/video_encrypt.py \
  --input path/to/MVI_39821 \
  --model path/to/yolo26_best.pt \
  --output-dir runs/benchmark_30 \
  --max-frames 30 \
  --imgsz 960
```

The console and metadata contain detection, encryption-call, and total
latencies. MATLAB's internal encryption time is recorded separately from
Python-to-MATLAB transfer overhead.

## Decrypt

```bash
python scripts/video_decrypt.py \
  --input-dir runs/encrypted \
  --output-dir runs/recovered \
  --preview
```

The decryptor does not run YOLO again. It uses the exact bounding boxes and
pool indices stored during encryption, avoiding detector-jitter mismatch.

For deliberately damaged ciphertext experiments only:

```bash
python scripts/video_decrypt.py \
  --input-dir runs/damaged \
  --output-dir runs/recovered_damaged \
  --enable-recovery
```

## MATLAB-only demo

From MATLAB:

```matlab
cd matlab
demo
```

## Pure-Python encryption core

`python/roi_cipher.py` ports the encryption core to NumPy, function by
function, so the cipher can be used without the MATLAB Engine:

```python
import numpy as np
from roi_cipher import encrypt, decrypt

roi = np.asarray(...)                      # H x W x 3, uint8
cipher, digest = encrypt(roi, rounds=3)
plain = decrypt(cipher, digest, rounds=3)  # bit-exact
assert np.array_equal(plain, roi)
```

The secret key is the `base_state` argument, pre-shared out of band and
never derived from image data. The returned digest is *not* secret: it
cannot be recomputed from the ciphertext, so it is transmitted with the ROI
metadata, and it gives an adversary no advantage without `base_state`.

### Choosing between the two implementations

The hyperchaotic system is integrated with fixed-step RK4 in double
precision and is highly sensitive to its initial state, so floating-point
rounding differences between MATLAB and NumPy can yield different key
streams for the same input. Each implementation is internally consistent
and round-trips bit-exactly, but ciphertexts are **not guaranteed to be
byte-identical across the two**. Encrypt and decrypt with the same
implementation end to end. The MATLAB code remains the reference for the
reported experimental figures.

## Important implementation decisions

1. SHA mode hashes the complete `uint8` ROI once. It does not hash the three
   colour channels separately and add the digests.
2. Digest bytes are mapped to initial states through explicit unsigned
   integer arithmetic. Arbitrary bytes are never reinterpreted as IEEE-754
   floats.
3. Chaotic integration is deterministic. Numerical divergence raises an
   error; it is never repaired using random noise.
4. Pool-mode encryption returns and stores the exact start index used by
   the ROI. It does not substitute the next free index.
5. Pool segments are allocated without overlap within an encryption run,
   avoiding accidental chaotic-keystream reuse.
6. Overlapping detector boxes are merged before encryption, so one encrypted
   ROI cannot overwrite another.
7. Standard decryption disables heuristic damage recovery to preserve exact
   pixels.

## Known research limitations

- The current DFTF source preserves the transform used by the existing
  experiments. Its fixed-key, cross-position avalanche property must be
  evaluated independently from the plaintext-dependent pool-index
  selection. Do not attribute an NPCR result caused by a changed key stream
  solely to the diffusion transform.
- CCGR damage detection is an experimental heuristic for zeroing and
  salt-and-pepper corruption; it is not authenticated integrity protection.
- Precomputing a large chaotic pool is CPU- and memory-intensive.
- Metadata integrity is not authenticated in this research version.
- Lossy codecs modify ciphertext pixels. Only the PNG frames and metadata
  are decryptable.

## Data and weights

UA-DETRAC and trained YOLO weights are not bundled. Obtain the dataset under
its own terms and supply the weight path with `--model`. Before publishing
weights, verify the dataset, model, and framework licensing conditions.

## Citation and license

Update `CITATION.cff` with the final paper DOI and repository URL before
release. Select and add a software license before making the repository
public; no license has been chosen automatically.
