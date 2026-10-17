"""Verify an unpacked private VIPP wheel and bind actual offline pass receipts.

Pure inventory/hash work only: no install, package import, Qt, provider or
authentication activity. The wheel is built separately by the release owner.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import re
import sys
import zipfile
from pathlib import Path

_HEX = re.compile(r"[0-9a-f]{64}\Z")


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path):
    value = json.loads(Path(path).read_text("utf-8"))
    if type(value) is not dict:
        raise ValueError("A qualification document must be a JSON object.")
    return value


def _package_directory(root):
    choices = [
        path
        for path in (root / "napari_vipp", root / "src/napari_vipp")
        if (path / "__init__.py").is_file()
    ]
    if len(choices) != 1:
        raise ValueError("Exactly one private napari_vipp package is required.")
    package = choices[0]
    if package.is_symlink() or not package.resolve().is_relative_to(root):
        raise ValueError("A redirected private package is not qualified.")
    return package


def inventory(root):
    """Inventory every package module/asset, excluding only runtime byte caches."""
    root = Path(root).resolve(strict=True)
    package = _package_directory(root)
    modules, assets = {}, {}
    for path in sorted(package.rglob("*")):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(package):
            raise ValueError("A redirected package file is not qualified.")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        (modules if path.suffix == ".py" else assets)[relative] = file_sha256(path)
    return package, modules, assets


def _wheel_inventory(wheel):
    result = {}
    with zipfile.ZipFile(wheel) as archive:
        for entry in archive.infolist():
            path = Path(entry.filename)
            if not entry.filename.startswith("napari_vipp/") or entry.is_dir():
                continue
            if ".." in path.parts or path.is_absolute():
                raise ValueError("A wheel contains an invalid package path.")
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            if entry.filename in result:
                raise ValueError("A wheel contains duplicate package members.")
            result[entry.filename] = hashlib.sha256(archive.read(entry)).hexdigest()
    if "napari_vipp/__init__.py" not in result:
        raise ValueError("The wheel does not contain the expected package.")
    return result


def build_receipt(
    wheel,
    qualified_source,
    check_receipts,
    *,
    source_head=None,
    applied_source_inventory=None,
):
    root, wheel = (
        Path(qualified_source).resolve(strict=True),
        Path(wheel).resolve(strict=True),
    )
    package, modules, assets = inventory(root)
    prefix = package.parent.relative_to(root).as_posix()
    prefix = "" if prefix == "." else prefix + "/"
    wheel_files = {
        prefix + key: value for key, value in _wheel_inventory(wheel).items()
    }
    if wheel_files != modules | assets:
        raise ValueError(
            "Private package inventory/bytes do not exactly match the wheel."
        )
    checks = []
    for name, path in check_receipts:
        path = Path(path).resolve(strict=True)
        document = _json(path)
        if not name or document.get("status") != "passed":
            raise ValueError(
                "Every offline check must have an actual passed JSON receipt."
            )
        checks.append(
            {
                "name": name,
                "status": "passed",
                "receipt_path": str(path),
                "receipt_sha256": file_sha256(path),
                "receipt_document": document,
            }
        )
    if not checks:
        raise ValueError("At least one actual offline pass receipt is required.")
    result = {
        "kind": "napari-vipp-private-wheel-qualification",
        "version": 1,
        "status": "passed",
        "qualified_source": str(root),
        "package_directory": package.relative_to(root).as_posix(),
        "wheel_path": str(wheel),
        "wheel_sha256": file_sha256(wheel),
        "module_file_sha256": modules,
        "asset_file_sha256": assets,
        "checks": checks,
    }
    if source_head is not None:
        if not re.fullmatch(r"[0-9a-f]{40,64}", source_head):
            raise ValueError("The source head must be an exact hexadecimal revision.")
        result["source_head"] = source_head
    if applied_source_inventory is not None:
        path = Path(applied_source_inventory).resolve(strict=True)
        result["applied_source_inventory"] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "document": _json(path),
        }
    return result


def verify_receipt(receipt_path, qualified_source):
    """Recheck the complete byte inventory and original offline/wheel receipts."""
    receipt = _json(receipt_path)
    root = Path(qualified_source).resolve(strict=True)
    if (
        receipt.get("kind") != "napari-vipp-private-wheel-qualification"
        or receipt.get("version") != 1
        or receipt.get("status") != "passed"
        or receipt.get("qualified_source") != str(root)
    ):
        raise ValueError(
            "A passed qualification for this exact private root is required."
        )
    package, modules, assets = inventory(root)
    if (
        receipt.get("package_directory") != package.relative_to(root).as_posix()
        or receipt.get("module_file_sha256") != modules
        or receipt.get("asset_file_sha256") != assets
    ):
        raise ValueError(
            "The complete qualified module/asset inventory or bytes changed."
        )
    wheel_path, wheel_hash = receipt.get("wheel_path"), receipt.get("wheel_sha256")
    if type(wheel_hash) is not str or not _HEX.fullmatch(wheel_hash):
        raise ValueError("The qualification must identify its exact built wheel.")
    if type(wheel_path) is not str or file_sha256(wheel_path) != wheel_hash:
        raise ValueError("The qualified wheel identity changed or is unavailable.")
    checks = receipt.get("checks")
    if type(checks) is not list or not checks:
        raise ValueError("Actual offline pass receipts are required.")
    check_pairs = []
    for check in checks:
        if (
            type(check) is not dict
            or check.get("status") != "passed"
            or type(check.get("name")) is not str
            or not check["name"]
            or type(check.get("receipt_path")) is not str
            or file_sha256(check["receipt_path"]) != check.get("receipt_sha256")
            or _json(check["receipt_path"]) != check.get("receipt_document")
        ):
            raise ValueError("An actual offline pass receipt changed or is invalid.")
        check_pairs.append((check["name"], check["receipt_path"]))
    rebuilt = build_receipt(
        wheel_path,
        root,
        check_pairs,
        source_head=receipt.get("source_head"),
        applied_source_inventory=receipt.get("applied_source_inventory", {}).get(
            "path"
        ),
    )
    if rebuilt != receipt:
        raise ValueError(
            "Qualification receipt fields do not match their verified evidence."
        )
    return receipt


def assert_private_imports(qualified_source):
    package = _package_directory(Path(qualified_source).resolve(strict=True))
    for name, module in tuple(sys.modules.items()):
        if name != "napari_vipp" and not name.startswith("napari_vipp."):
            continue
        origin = getattr(module, "__file__", None)
        if origin is None or not Path(origin).resolve().is_relative_to(package):
            raise ValueError(
                "A VIPP module was imported outside the qualified package."
            )


class _VerifiedSourceOnlyLoader(importlib.machinery.SourceFileLoader):
    """Compile the qualified source bytes; never consult or execute cached pyc."""

    def __init__(self, fullname, filename, expected_sha256):
        super().__init__(fullname, str(filename))
        self.expected_sha256 = expected_sha256

    def get_code(self, fullname):
        source = Path(self.path).read_bytes()
        if hashlib.sha256(source).hexdigest() != self.expected_sha256:
            raise ImportError("Qualified module source bytes changed before import.")
        return compile(source, self.path, "exec", dont_inherit=True)


class PrivatePackageImportGuard(importlib.abc.MetaPathFinder):
    """Reject foreign VIPP modules before their bodies can execute."""

    def __init__(self, qualified_source, *, module_file_sha256=None):
        self.root = Path(qualified_source).resolve(strict=True)
        self.package, actual_modules, _assets = inventory(self.root)
        self.modules = (
            actual_modules if module_file_sha256 is None else module_file_sha256
        )
        if self.modules != actual_modules:
            raise ImportError("The verified module inventory changed before import.")

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "napari_vipp" and not fullname.startswith("napari_vipp."):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if (
            spec is None
            or spec.origin is None
            or Path(spec.origin).suffix != ".py"
            or not (Path(spec.origin).resolve().is_relative_to(self.package))
        ):
            raise ImportError("A VIPP import is outside the qualified private package.")
        origin = Path(spec.origin).resolve()
        expected = self.modules.get(origin.relative_to(self.root).as_posix())
        if expected is None:
            raise ImportError(
                "A VIPP module is outside the qualified source inventory."
            )
        return importlib.util.spec_from_file_location(
            fullname,
            origin,
            loader=_VerifiedSourceOnlyLoader(fullname, origin, expected),
            submodule_search_locations=[]
            if spec.submodule_search_locations is not None
            else None,
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--qualified-source", type=Path, required=True)
    parser.add_argument("--check-receipt", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-head")
    parser.add_argument("--applied-source-inventory", type=Path)
    args = parser.parse_args(argv)
    checks = []
    for value in args.check_receipt:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            parser.error("Each --check-receipt must be NAME=PATH_TO_PASSED_JSON.")
        checks.append((name, Path(path)))
    receipt = build_receipt(
        args.wheel,
        args.qualified_source,
        checks,
        source_head=args.source_head,
        applied_source_inventory=args.applied_source_inventory,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    verify_receipt(args.output, args.qualified_source)
    print(
        f"Qualified {len(receipt['module_file_sha256'])} modules and "
        f"{len(receipt['asset_file_sha256'])} assets; no UI/model activity."
    )


if __name__ == "__main__":
    main()
