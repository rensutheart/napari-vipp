"""napari-vipp plugin."""

try:
    from importlib.metadata import version
except ImportError:  # pragma: no cover
    from importlib_metadata import version

try:
    __version__ = version("napari-vipp")
except Exception:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = ["__version__"]
