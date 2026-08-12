# Changes from the original development folder

The desktop development folder was not modified or deleted. This public
package is a separate clean implementation copy.

## Protocol and reproducibility fixes

- Replaced three-channel digest addition with one SHA-256 digest over the
  complete `uint8` ROI in MATLAB SHA mode.
- Replaced digest-to-IEEE-single reinterpretation with explicit,
  endian-independent unsigned-integer mapping.
- Removed random numerical recovery from the chaotic integrator. Divergence
  now raises a reproducible error.
- Replaced a mutable MATLAB global variable with controlled persistent key
  pool storage.
- Pool encryption returns the start index actually used by the ROI.
- Added pool range validation.
- Added a secret base key, a public random session nonce, and a derived
  per-run pool key.
- Added HMAC-SHA-256 plaintext-bound preferred pool indices and a
  non-overlapping interval allocator to avoid segment reuse within a run.
- Made the forward and inverse 3D-ATP implementations share exactly the same
  shift construction.
- Separated exact decryption from optional heuristic CCGR recovery.

## Video-interface fixes

- Replaced hard-coded personal paths with command-line arguments.
- Added MATLAB source-path registration after engine startup.
- Crops every ROI from the untouched plaintext frame.
- Merges overlapping detection boxes before encryption.
- Saves decryptable ciphertext as lossless PNG frames.
- Saves exact bounding boxes and pool indices in JSON metadata.
- Added a video decryption entry point that does not rerun YOLO.
- Treats generated MP4 files as lossy previews only.
- Separately records detector latency, Python/MATLAB call latency, and
  MATLAB-internal encryption latency.

## Deliberately not changed

The two-wing DFTF arithmetic was preserved so this package remains comparable
with the existing experiments. The included diagnostic shows that its
fixed-key cross-position avalanche is limited. Changing that transform would
require rerunning the paper's entropy, correlation, NPCR/UACI, timing, and
robustness experiments and updating the mathematical description.

## Consequences for the paper

Results produced with the old digest-to-float mapping are not bit-for-bit
reproducible with this corrected mapping. Before submission, rerun the
reported experiments with the public package or explicitly identify the
archived implementation used to produce the tables and figures.
