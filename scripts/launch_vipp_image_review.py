"""Launch one synthetic review workflow in a new, independent VIPP session."""

from __future__ import annotations

import argparse

from launch_vipp_intensity_workflow import main as launch_workflow

EXAMPLES = (
    "review-channels-2d",
    "review-mask-3d",
    "review-labels-time-series",
    "review-rgb-index-3d",
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "example", nargs="?", choices=EXAMPLES, default="review-mask-3d"
    )
    args = parser.parse_args(argv)
    launch_workflow([args.example, "review"])


if __name__ == "__main__":
    main()
