"""Release discovery policy; no network, package installation, or Qt imports."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote

from packaging.version import InvalidVersion, Version

RELEASES_URL = "https://github.com/rensutheart/napari-vipp/releases"
RELEASES_API = (
    "https://api.github.com/repos/rensutheart/napari-vipp/releases?per_page=100"
)
MAX_RESPONSE_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class PublishedRelease:
    version: Version
    tag: str
    prerelease: bool
    assets: tuple[str, ...]

    @property
    def notes_url(self) -> str:
        return f"{RELEASES_URL}/tag/{quote(self.tag, safe='')}"

    def asset_url(self, name: str) -> str:
        if name not in self.assets:
            raise ValueError("That asset is not part of the published release.")
        return (
            f"{RELEASES_URL}/download/{quote(self.tag, safe='')}/{quote(name, safe='')}"
        )

    def cache_record(self) -> dict:
        return {
            "tag_name": self.tag,
            "draft": False,
            "prerelease": self.prerelease,
            "published_at": "cached",
            "assets": [
                {
                    "name": name,
                    "state": "uploaded",
                    "browser_download_url": self.asset_url(name),
                }
                for name in self.assets
            ],
        }


def current_release_url(version: str) -> str:
    try:
        parsed = Version(version)
    except InvalidVersion:
        return RELEASES_URL
    if parsed.is_devrelease or parsed.local:
        return RELEASES_URL
    return f"{RELEASES_URL}/tag/v{parsed}"


def parse_releases(payload: object) -> tuple[PublishedRelease, ...]:
    """Ignore drafts, malformed versions, and non-official download URLs."""
    if not isinstance(payload, list):
        raise ValueError("GitHub did not return a release list.")
    releases = []
    for entry in payload:
        if not isinstance(entry, dict) or entry.get("draft") is not False:
            continue
        tag = entry.get("tag_name")
        if not isinstance(tag, str) or not tag.startswith("v") or len(tag) > 80:
            continue
        try:
            version = Version(tag[1:])
        except InvalidVersion:
            continue
        if (
            tag != f"v{version}"
            or version.is_devrelease
            or version.local
            or not entry.get("published_at")
        ):
            continue
        assets = []
        prefix = f"{RELEASES_URL}/download/{quote(tag, safe='')}/"
        for asset in (
            entry.get("assets", []) if isinstance(entry.get("assets"), list) else []
        ):
            if not isinstance(asset, dict):
                continue
            name = asset.get("name")
            if (
                isinstance(name, str)
                and 0 < len(name) < 200
                and "/" not in name
                and "\\" not in name
                and asset.get("state") == "uploaded"
                and asset.get("browser_download_url") == prefix + quote(name, safe="")
            ):
                assets.append(name)
        releases.append(
            PublishedRelease(
                version,
                tag,
                bool(entry.get("prerelease")) or version.is_prerelease,
                tuple(assets),
            )
        )
    return tuple(sorted(releases, key=lambda release: release.version, reverse=True))


def newest_release(
    releases: tuple[PublishedRelease, ...],
    *,
    include_prereleases: bool,
) -> PublishedRelease | None:
    eligible = [r for r in releases if include_prereleases or not r.prerelease]
    return max(eligible, key=lambda r: r.version, default=None)


def installer_asset(
    release: PublishedRelease,
    system: str,
    machine: str,
) -> tuple[str, str] | None:
    """Offer only a matching published installer with its checksum sidecar."""
    version = str(release.version)
    if system == "Windows" and machine.lower() in {"amd64", "x86_64"}:
        base = f"VIPP-Setup-{version}-Windows-x86_64"
        checksum = f"SHA256SUMS-Windows-{version}.txt"
        extension = ".exe"
    elif system == "Darwin" and machine.lower() in {
        "arm64",
        "aarch64",
        "x86_64",
        "amd64",
    }:
        arch = "arm64" if machine.lower() in {"arm64", "aarch64"} else "x86_64"
        base = f"VIPP-{version}-macOS-{arch}"
        checksum = f"SHA256SUMS-macOS-{arch}-{version}.txt"
        extension = ".pkg"
    else:
        return None
    if checksum not in release.assets:
        return None
    for name in (base + extension, base + "-UNSIGNED" + extension):
        if name in release.assets:
            return name, checksum
    return None
