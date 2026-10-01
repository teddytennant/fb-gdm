"""Preload NixOS libstdc++ before JAX native extensions."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

_PRELOADED = False


def preload() -> None:
    """Load libstdc++, libgcc_s, and libz with RTLD_GLOBAL if present."""
    global _PRELOADED
    if _PRELOADED:
        return
    roots: list[str] = []
    for key in ("NIX_LD_LIBRARY_PATH", "LD_LIBRARY_PATH"):
        roots.extend(p for p in os.environ.get(key, "").split(":") if p)
    roots.append("/run/current-system/sw/share/nix-ld/lib")
    for name in ("libstdc++.so.6", "libgcc_s.so.1", "libz.so.1"):
        for root in roots:
            path = Path(root) / name
            if path.exists():
                ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
                break
    _PRELOADED = True
