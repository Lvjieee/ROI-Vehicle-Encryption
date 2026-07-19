"""Check the local Python and MATLAB Engine environment without arguments."""

from __future__ import annotations

import importlib
import os
import platform
import sys


REQUIRED_MODULES = {
    "cv2": "opencv-python",
    "numpy": "numpy",
    "matlab.engine": "MATLAB Engine for Python",
    "ultralytics": "ultralytics",
    "torch": "torch",
}


def main() -> int:
    print(f"Python executable : {sys.executable}")
    print(f"Python version    : {sys.version.split()[0]}")
    print(f"Operating system  : {platform.platform()}")
    print()

    failed = False
    for module_name, package_name in REQUIRED_MODULES.items():
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "available")
            print(f"[OK] {package_name:<26} {version}")
        except Exception as exc:
            failed = True
            print(f"[MISSING] {package_name:<21} {type(exc).__name__}: {exc}")

    print()
    if sys.version_info[:2] != (3, 11):
        print(
            "[WARNING] This project was validated with Python 3.11. "
            "Select the installed Python 3.11 interpreter in the IDE."
        )
    if os.name == "nt":
        translated = os.environ.get("MWLOCALE_TRANSLATED")
        print(f"MWLOCALE_TRANSLATED: {translated or '<not set>'}")
        if translated != "OFF":
            print(
                "[TIP] If MATLAB Engine raises UnicodeDecodeError, set the "
                "Windows environment variable MWLOCALE_TRANSLATED=OFF and "
                "restart the IDE."
            )

    if failed:
        print("\nEnvironment check failed.")
        return 1
    print("\nEnvironment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
