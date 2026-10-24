"""Bounded file exchanges for changed packages, never whole-prefix deletion.

All paths are prefix-relative and independently validated on recovery. Backups
and discarded new files are renamed, not copied. Every rollback conflict is
checked before the first restoration so foreign content is always preserved.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath

from napari_vipp.installer import inplace as safety

MAX_FILES = 200_000
MAX_BYTES = 32 * 1024**3


def direct(path: Path, *, file=False) -> None:
    safety.direct(path, file=file)
    for part in (path, *path.parents):
        info = part.lstat()
        safety.require(
            not getattr(info, "st_file_attributes", 0) & 0x441000,
            "Offline/recall content cannot enter package recovery.",
        )


def manifests(rows) -> dict[str, tuple[str, int]]:
    safety.require(
        isinstance(rows, list) and len(rows) <= MAX_FILES,
        "Invalid package transaction manifest.",
    )
    result, total = {}, 0
    folded = set()
    for row in rows:
        safety.require(
            isinstance(row, dict) and set(row) == {"path", "sha256", "size"},
            "Invalid changed-file record.",
        )
        name, sha, size = row["path"], row["sha256"], row["size"]
        safety.require(isinstance(name, str), "Invalid changed-file name.")
        path = PurePosixPath(name)
        safety.require(
            str(path) == name
            and 3 <= len(path.parts) <= 128
            and len(name) <= 32767
            and path.parts[:2] == ("Lib", "site-packages")
            and all(part not in {".", "..", ""} for part in path.parts)
            and not any(c in name for c in "\\:\0")
            and all(
                not p.endswith((".", " "))
                and p.split(".")[0].casefold() not in safety.reuse._RESERVED
                and not any(ord(c) < 32 or c in '<>"|?*' for c in p)
                for p in path.parts
            )
            and name.casefold() not in folded
            and isinstance(sha, str)
            and len(sha) == 64
            and all(c in "0123456789abcdef" for c in sha)
            and type(size) is int
            and 0 <= size <= MAX_BYTES,
            "Unsafe changed-package file scope.",
        )
        result[name] = sha, size
        folded.add(name.casefold())
        total += size
    safety.require(total <= MAX_BYTES, "Changed package batch exceeds its size bound.")
    return result


def matches(path: Path, expected: tuple[str, int]) -> bool:
    if not os.path.lexists(path):
        return False
    direct(path, file=True)
    return path.stat().st_size == expected[1] and safety.reuse._sha(path) == expected[0]


def _absent(path: Path) -> None:
    safety.require(
        not os.path.lexists(path), "A transaction destination already exists."
    )
    parent = path.parent
    while not os.path.lexists(parent):
        parent = parent.parent
    direct(parent)


def _parents(path: Path, boundary: Path) -> None:
    safety.require(
        path.is_relative_to(boundary), "Transaction path escapes its boundary."
    )
    pending, parent = [], path.parent
    while parent != boundary and not os.path.lexists(parent):
        pending.append(parent)
        parent = parent.parent
    direct(parent)
    for directory in reversed(pending):
        directory.mkdir(exist_ok=False)
    direct(path.parent)


def _move(source: Path, destination: Path, expected, boundary: Path) -> None:
    safety.require(matches(source, expected), "Changed package bytes no longer match.")
    _absent(destination)
    _parents(destination, boundary)
    # No replace of existing files is authorized. The same-target lock plus
    # offline prefix/maintenance guard owns this namespace, not arbitrary users.
    if os.name == "nt":
        os.rename(source, destination)  # Windows refuses existing destination.
    else:
        # Unix is used only by source fixture tests; check again immediately.
        _absent(destination)
        os.rename(source, destination)


def preflight(prefix: Path, old: dict, new: dict) -> None:
    """Check the entire original touched-file union before any exchange."""
    direct(prefix)
    for name, value in old.items():
        safety.require(
            matches(prefix / name, value), "An installed changed file differs."
        )
    for name in new:
        if name not in old:
            _absent(prefix / name)


def check_bytecode(prefix: Path, old: dict, new: dict, checkpoint=None) -> None:
    """Refuse late caches before touching source; -B alone still reads caches.

    Inspect only caches adjacent to touched Python sources, including newly added
    sources. Unknown cache tags and optimisation variants are not exempt. A
    refusal preserves the live cache and aborts the untouched transaction.
    """
    direct(prefix)
    known = {name.casefold(): value for name, value in old.items()}
    sources = {
        name.casefold(): name
        for name in old.keys() | new.keys()
        if Path(name).suffix.casefold() == ".py"
    }
    directories = {}
    for name in sources.values():
        source = prefix / name
        adjacent = source.with_suffix(".pyc")
        if os.path.lexists(adjacent):
            key = adjacent.relative_to(prefix).as_posix().casefold()
            safety.require(
                key in known and matches(adjacent, known[key]),
                "Python bytecode changed before exclusion; retry the review.",
            )
        directories.setdefault(source.parent / "__pycache__", set()).add(
            source.stem.casefold()
        )
    visited = 0
    for directory, stems in directories.items():
        if checkpoint:
            checkpoint()
        if not os.path.lexists(directory):
            continue
        direct(directory)
        for cache in directory.iterdir():
            visited += 1
            safety.require(visited <= MAX_FILES, "Too many adjacent cache entries.")
            if checkpoint:
                checkpoint()
            folded = cache.name.casefold()
            pieces = folded.split(".")
            if folded.endswith(".pyc") and any(
                ".".join(pieces[:index]) in stems for index in range(1, len(pieces))
            ):
                key = cache.relative_to(prefix).as_posix().casefold()
                safety.require(
                    key in known and matches(cache, known[key]),
                    "Python bytecode appeared or changed before exclusion; "
                    "the original files are preserved. Review and try again.",
                )


def exchange(
    prefix: Path, workspace: Path, old: dict, new: dict, checkpoint=None, progress=None
) -> None:
    preflight(prefix, old, new)
    for name, value in new.items():
        safety.require(
            matches(workspace / "stage" / name, value), "A staged changed file differs."
        )
    for index, (name, value) in enumerate(old.items(), 1):
        if checkpoint:
            checkpoint()
        _move(prefix / name, workspace / "backup" / name, value, workspace)
        if progress:
            progress("backup", index, len(old))
    for index, (name, value) in enumerate(new.items(), 1):
        if checkpoint:
            checkpoint()
        _move(workspace / "stage" / name, prefix / name, value, prefix)
        if progress:
            progress("install", index, len(new))


def restore(prefix: Path, workspace: Path, old: dict, new: dict) -> None:
    """Idempotent at every rename boundary, without unlinking live payloads."""
    direct(prefix)
    # Validate all backups, live files and prior discarded files BEFORE moving.
    for name, value in old.items():
        backup, live = workspace / "backup" / name, prefix / name
        if os.path.lexists(backup):
            safety.require(
                matches(backup, value), "A rollback backup changed; preserved."
            )
            if os.path.lexists(live):
                safety.require(
                    name in new and matches(live, new[name]),
                    "A live rollback conflict was preserved.",
                )
        else:
            safety.require(matches(live, value), "An old changed file is missing.")
    for name, value in new.items():
        live, discard = prefix / name, workspace / "discard" / name
        original = name in old and not os.path.lexists(workspace / "backup" / name)
        if os.path.lexists(live):
            safety.require(
                matches(live, old[name] if original else value),
                "A new changed file has foreign content; preserved.",
            )
        if os.path.lexists(discard):
            safety.require(
                matches(discard, value) and (original or not live.exists()),
                "A discarded-file conflict was preserved.",
            )
    for name, value in new.items():
        live = prefix / name
        original = name in old and not os.path.lexists(workspace / "backup" / name)
        if os.path.lexists(live) and not original:
            _move(live, workspace / "discard" / name, value, workspace)
    for name, value in old.items():
        backup = workspace / "backup" / name
        if os.path.lexists(backup):
            _move(backup, prefix / name, value, prefix)
        safety.require(
            matches(prefix / name, value), "Package rollback did not complete."
        )


def cleanup(workspace: Path, old: dict, new: dict) -> None:
    """Remove only exact files in the owned workspace; no recursive deletion."""
    if not os.path.lexists(workspace):
        return
    direct(workspace)
    for directory, expected in (("backup", old), ("stage", new), ("discard", new)):
        root = workspace / directory
        if not os.path.lexists(root):
            continue
        direct(root)
        pending, files, dirs = [root], [], []
        while pending:
            parent = pending.pop()
            safety.require(
                len(files) + len(dirs) <= MAX_FILES * 4,
                "Workspace cleanup exceeds its bound.",
            )
            direct(parent)
            for child in parent.iterdir():
                safety.require(
                    len(files) + len(dirs) < MAX_FILES * 4,
                    "Workspace cleanup exceeds its entry bound.",
                )
                direct(child)
                if child.is_dir():
                    dirs.append(child)
                    pending.append(child)
                else:
                    name = child.relative_to(root).as_posix()
                    safety.require(
                        name in expected and matches(child, expected[name]),
                        "Unknown backup content was preserved.",
                    )
                    files.append((child, expected[name]))
        for path, value in files:
            safety.require(matches(path, value), "Backup changed during cleanup.")
            path.unlink()
        for directory_path in sorted(dirs, key=lambda p: len(p.parts), reverse=True):
            direct(directory_path)
            directory_path.rmdir()
        root.rmdir()
    workspace.rmdir()  # Unexpected content blocks; never recursively removed.


def remove_empty_payload_directories(prefix: Path, paths: dict) -> None:
    """Retire empty touched metadata directories, never nonempty namespaces."""
    boundary = prefix / "Lib" / "site-packages"
    parents = set()
    for name in paths:
        parent = (prefix / name).parent
        while parent != boundary:
            safety.require(parent.is_relative_to(boundary), "Unsafe payload directory.")
            parents.add(parent)
            parent = parent.parent
    for parent in sorted(parents, key=lambda p: len(p.parts), reverse=True):
        if not os.path.lexists(parent):
            continue
        direct(parent)
        if not any(parent.iterdir()):
            parent.rmdir()
