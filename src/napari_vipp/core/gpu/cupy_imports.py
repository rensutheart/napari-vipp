"""Narrow lazy-import helpers for optional CuPyX modules."""

from __future__ import annotations

import importlib
import re
import warnings
from types import ModuleType

_CUPYX_JIT_RAWKERNEL_EXPERIMENTAL_WARNING = (
    "cupyx.jit.rawkernel is experimental. The interface can change in the future."
)


def import_cupyx_signal_module() -> ModuleType:
    """Import CuPyX signal without exposing its internal API-stability notice.

    CuPy 14.1.1 emits this exact warning while importing two private signal
    kernels.  VIPP does not call ``cupyx.jit.rawkernel`` itself.  Keep the
    filter local to this lazy import so every other warning remains visible.
    """

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(rf"^{re.escape(_CUPYX_JIT_RAWKERNEL_EXPERIMENTAL_WARNING)}$"),
            category=FutureWarning,
            module=r"^cupyx\.jit\._interface$",
        )
        return importlib.import_module("cupyx.scipy.signal")


__all__ = ["import_cupyx_signal_module"]
