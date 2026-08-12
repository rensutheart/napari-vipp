"""Launch VIPP with immediate, branded startup feedback."""

from __future__ import annotations

from napari_vipp.launcher import main

if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
