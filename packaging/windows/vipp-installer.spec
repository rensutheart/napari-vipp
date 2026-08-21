"""Reviewed PyInstaller configuration for the standalone Windows setup EXE."""

from __future__ import annotations

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

ROOT = Path(SPEC).resolve().parents[2]
WHEEL = Path(os.environ["VIPP_INSTALLER_WHEEL"]).resolve()
PAYLOAD_MANIFEST = Path(os.environ["VIPP_INSTALLER_PAYLOAD_MANIFEST"]).resolve()
ICON = Path(os.environ["VIPP_INSTALLER_ICON"]).resolve()
SPLASH_IMAGE = Path(os.environ["VIPP_INSTALLER_SPLASH"]).resolve()
INSTALLER_LOGO = Path(os.environ["VIPP_INSTALLER_LOGO"]).resolve()
VERSION_INFO = Path(os.environ["VIPP_INSTALLER_VERSION_INFO"]).resolve()
LICENSE_DIRECTORY = Path(os.environ["VIPP_INSTALLER_LICENSE_DIRECTORY"]).resolve()
EXE_NAME = os.environ["VIPP_INSTALLER_EXE_NAME"]

datas = [
    (str(WHEEL), "installer_payload"),
    (str(PAYLOAD_MANIFEST), "installer_payload"),
    (str(ROOT / "LICENSE"), "licenses/napari-vipp"),
    (str(ROOT / "NOTICE"), "licenses/napari-vipp"),
    (str(LICENSE_DIRECTORY), "installer_licenses"),
    (str(INSTALLER_LOGO), "installer_branding"),
    (
        str(
            ROOT
            / "src"
            / "napari_vipp"
            / "compute_policies"
            / "phase1-gpu-public-v9.json"
        ),
        "napari_vipp/compute_policies",
    ),
]

analysis = Analysis(
    [str(ROOT / "packaging" / "windows" / "vipp_installer_entry.py")],
    pathex=[str(ROOT / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=collect_submodules("napari_vipp.installer"),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "cupy",
        "dask",
        "napari",
        "numpy",
        "PyQt5",
        "PyQt6",
        "scipy",
        "setuptools",
        "skimage",
        "wheel",
    ],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(analysis.pure)
splash = Splash(
    str(SPLASH_IMAGE),
    binaries=analysis.binaries,
    datas=analysis.datas,
    text_pos=(360, 390),
    text_size=11,
    text_color="#d8e7f3",
    text_default="Starting VIPP Setup…",
    minify_script=True,
    always_on_top=True,
)

exe = EXE(
    pyz,
    analysis.scripts,
    splash,
    splash.binaries,
    analysis.binaries,
    analysis.datas,
    [],
    name=EXE_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch="x86_64",
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ICON),
    version=str(VERSION_INFO),
    uac_admin=False,
)
