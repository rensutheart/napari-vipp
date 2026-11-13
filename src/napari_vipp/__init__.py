"""napari-vipp plugin."""

from napari_vipp._runtime_lease import register_runtime as _register_runtime

_runtime_use_lease = _register_runtime()
del _register_runtime

try:
    from importlib.metadata import version
except ImportError:  # pragma: no cover
    from importlib_metadata import version

try:
    __version__ = version("napari-vipp")
except Exception:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = ["__version__"]
