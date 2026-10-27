# Recoverable ROI Encryption for Vehicle Privacy in the IoV

Code for *Recoverable ROI Encryption for Vehicle Privacy in the Internet of
Vehicles*. YOLO26s finds vehicles, each vehicle region is encrypted with
3D-ATP permutation and DFTF diffusion driven by a 6-D hyperchaotic system,
and CCGR repairs ciphertext damaged in transit.

Research code, not audited cryptographic software.

## Run

```bash
pip install -r requirements.txt
python demo.py                                   # synthetic ROI
python demo.py --image car.png --resize 512      # your ROI, paper setting
python demo.py --roi-dir rois/                   # mean/std over many ROIs
python demo.py --image car.png --video MVI_39821/ --model yolo26_best.pt
```

`demo.py` prints lossless recovery, entropy, correlation, NPCR/UACI, key
sensitivity, and PSNR/SSIM after cropping and noise with and without CCGR,
and saves `demo_output/demo.png`.

## Layout

```text
roienc/chaos.py    digest, initial state, RK4 keystream     Eqs. (1)-(3)
roienc/atp.py      3D-ATP                                    Eq. (4)
roienc/dftf.py     DFTF                                      Eqs. (5)-(7)
roienc/ccgr.py     CCGR                                      Eqs. (8)-(9)
roienc/cipher.py   encrypt_roi / decrypt_roi
roienc/video.py    detection and video encryption/decryption
roienc/metrics.py  evaluation metrics and damage models
matlab/            the same cipher in MATLAB (encrypt.m, decrypt.m, demo.m)
```

## API

```python
from roienc import encrypt_roi, decrypt_roi

key_c = (0.01, 0.01, 0.01, 0.01, 0.01, 0.01)    # secret key c
cipher, digest = encrypt_roi(roi, key_c)          # roi: H x W x 3 uint8 RGB
plain = decrypt_roi(cipher, digest, key_c)        # bit-exact
fixed = decrypt_roi(damaged, digest, key_c, recover=True)   # CCGR
```

Only `key_c` is secret. `digest` (SHA-256 of the ROI) is sent with the
ciphertext and bounding box. The paper's experiments use the key above.

## Video

```bash
python -m roienc encrypt --input video.mp4 --model yolo26_best.pt --out runs/enc --preview
python -m roienc decrypt --input runs/enc --out runs/dec [--recover]
```

`--input` may be a video or a folder of frames. Key: `--key "c1,...,c6"` or
`ROIENC_KEY`. Decrypt from `runs/enc/frames/*.png` and `metadata.json`;
the mp4 preview is lossy. Detection settings: confidence 0.1, size 960,
classes 0-2, boxes under 10 px dropped, overlapping boxes merged.

## MATLAB

`matlab/` gives byte-identical ciphertexts to the Python code for the same
ROI and key. CCGR needs `regionfill` (Image Processing Toolbox).

## Requirements

Python 3.10+, numpy, scipy, opencv-python, pillow, scikit-image, matplotlib.
numba is optional (much faster key generation). ultralytics and torch are
needed only for video. UA-DETRAC and the YOLO26s weights are not included.
