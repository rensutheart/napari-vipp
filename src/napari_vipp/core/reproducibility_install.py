"""Offline installation links derived from recorded versions, never supplied URLs."""

from __future__ import annotations

from collections.abc import Mapping

from packaging.version import InvalidVersion, Version

from .updates import RELEASES_URL, current_release_url

INSTALLATION_GUIDE_URL = (
    "https://rensutheart.github.io/vipp-mkdocs/stable/"
    "getting-started/installation/"
)


def installation_guidance(environment: Mapping | None, *, recorded: bool) -> dict:
    """Never substitute the exporter's version for an unknown historical run.

    A version-tag link is a navigation aid, not evidence that a release exists,
    includes unpublished edits, or recreates every recorded dependency.
    """
    environment = environment if isinstance(environment, Mapping) else {}
    source = environment.get("run" if recorded else "export")
    source = source if isinstance(source, Mapping) else {}
    packages = source.get("packages")
    packages = packages if isinstance(packages, Mapping) else {}
    raw_version = packages.get("napari-vipp")
    parsed = None
    if isinstance(raw_version, str) and len(raw_version) <= 100:
        try:
            parsed = Version(raw_version)
        except InvalidVersion:
            pass
    if parsed is None or parsed == Version("0.0.0"):
        return {
            "version": "", "label": "VIPP downloads", "url": RELEASES_URL,
            "note": "The VIPP version was not recorded. Ask the author which "
            "version to use.",
        }
    version = str(parsed)
    if parsed.is_devrelease or parsed.local:
        return {
            "version": version, "label": "VIPP downloads", "url": RELEASES_URL,
            "note": f"VIPP {version} is a development build. Ask the author for "
            "that build; a public release may not match.",
        }
    return {
        "version": version,
        "label": f"VIPP {version} release and installers",
        "url": current_release_url(version),
        "note": "A release installer sets up VIPP, not an exact copy of every "
        "recorded dependency. Unpublished changes may require the author's build.",
    }
