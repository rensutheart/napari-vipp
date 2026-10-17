"""Private, read-only output destinations for assistant-authored Save nodes.

The model is allowed to request a writer, never to choose a filesystem path.
An explicit local user choice is bound after response validation. These helpers
do not create a directory, reserve a filename, or write scientific outputs.
The ordinary executor/writer must retain its no-overwrite publication policy;
a preflight check alone cannot close a filesystem check/use race.
"""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit
from urllib.request import url2pathname

OUTPUT_DIRECTORY_PLACEHOLDER = "USER_OUTPUT_DIRECTORY"
_LOCAL_KEY = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,39}\Z")
_SAME_FOLDER_ANSWERS = frozenset(
    {
        "same folder as input",
        "the same folder as input",
        "use the same folder as input",
        "use same folder as input",
        "same folder as the input",
        "use the same folder as the input",
        "same directory as input",
        "use the same directory as input",
    }
)
_EXTENSIONS = {
    "ome-zarr": ".ome.zarr",
    "ome-zarr-0.5": ".ome.zarr",
    "ome-tiff": ".ome.tif",
    "imagej-tiff": ".tif",
    "tiff": ".tif",
    "npy": ".npy",
    "png": ".png",
    "jpeg": ".jpg",
    "bmp": ".bmp",
    "gif": ".gif",
    "webp": ".webp",
    "tga": ".tga",
    "pnm": ".pnm",
    "obj": ".obj",
    "3mf": ".3mf",
}


class OutputDestinationError(ValueError):
    """The user must choose or review a local output destination."""


def _is_link(value: os.stat_result) -> bool:
    return stat.S_ISLNK(value.st_mode) or bool(
        getattr(value, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _identity(path: Path) -> tuple[int, int, int]:
    value = path.lstat()
    if _is_link(value):
        raise OutputDestinationError(
            "Choose a physical folder or source, not a symbolic link or junction."
        )
    return value.st_dev, value.st_ino, stat.S_IFMT(value.st_mode)


def _absolute_local_path(value: str | Path) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise OutputDestinationError("Choose an output folder first.")
    text = str(value).strip()
    if any(ord(character) < 32 for character in text):
        raise OutputDestinationError("The folder or source path contains control text.")
    if text.startswith("file:"):
        parsed = urlsplit(text)
        if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
            raise OutputDestinationError("Choose a local physical folder or source.")
        if parsed.query or parsed.fragment:
            raise OutputDestinationError("The source must identify a physical path.")
        text = url2pathname(parsed.path)
    elif "://" in text:
        raise OutputDestinationError("Choose a local physical folder or source.")
    path = Path(text).expanduser()
    if not path.is_absolute():
        raise OutputDestinationError("Choose a full output folder path.")
    # Inspect the authored path before resolve can hide a symlink or junction.
    try:
        for ancestor in reversed((path, *path.parents)):
            _identity(ancestor)
        resolved = path.resolve(strict=True)
        _identity(resolved)
    except (OSError, RuntimeError) as exc:
        raise OutputDestinationError(
            "The selected folder or source is no longer available. Choose it again."
        ) from exc
    return resolved


def _source_paths(values: Sequence[str | Path]) -> tuple[Path, ...]:
    if isinstance(values, (str, bytes, Path)):
        raise TypeError("source_paths must be a sequence of physical source paths.")
    sources: list[Path] = []
    for value in values:
        path = _absolute_local_path(value)
        if not path.is_file() and not path.is_dir():
            raise OutputDestinationError("The input must be a physical file or store.")
        if path not in sources:
            sources.append(path)
    return tuple(sources)


@dataclass(frozen=True, slots=True)
class OutputDestination:
    """User-owned local choice; never serialize this record into a model request.

    ``source_paths`` must come from the application's private, verified source
    bindings, not model text, display names, sample identifiers, or live layers.
    ``same_input_folder`` requires exactly one regular file or OME-Zarr root.
    Directory/source identities are captured without reading image contents.
    """

    directory: str = field(repr=False)
    mode: Literal["directory", "same_input_folder"] = "directory"
    source_paths: tuple[str, ...] = field(default=(), repr=False)
    _directory_identity: tuple[int, int, int] = field(init=False, repr=False)
    _source_identities: tuple[tuple[int, int, int], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.mode not in {"directory", "same_input_folder"}:
            raise OutputDestinationError("Choose an output folder or the input folder.")
        directory = _absolute_local_path(self.directory)
        if not directory.is_dir():
            raise OutputDestinationError(
                "The output location must be an existing folder."
            )
        sources = _source_paths(self.source_paths)
        if self.mode == "same_input_folder":
            source = _single_physical_source(sources)
            if directory != source.parent:
                raise OutputDestinationError(
                    "The selected folder is not the input folder."
                )
        for source in sources:
            if source.is_dir() and (directory == source or source in directory.parents):
                raise OutputDestinationError(
                    "Outputs cannot be saved inside the original input store."
                )
        object.__setattr__(self, "directory", str(directory))
        object.__setattr__(self, "source_paths", tuple(map(str, sources)))
        object.__setattr__(self, "_directory_identity", _identity(directory))
        object.__setattr__(
            self, "_source_identities", tuple(_identity(path) for path in sources)
        )

    @classmethod
    def from_directory(
        cls, path: str | Path, *, source_paths: Sequence[str | Path] = ()
    ) -> OutputDestination:
        sources = _source_paths(source_paths)
        return cls(str(path), source_paths=tuple(map(str, sources)))

    @classmethod
    def same_folder_as_input(
        cls, source_paths: Sequence[str | Path]
    ) -> OutputDestination:
        paths = _source_paths(source_paths)
        source = _single_physical_source(paths)
        return cls(str(source.parent), "same_input_folder", tuple(map(str, paths)))


def _single_physical_source(sources: tuple[Path, ...]) -> Path:
    if len(sources) != 1:
        raise OutputDestinationError(
            "The input folder is not unambiguous. Choose an output folder yourself."
        )
    source = sources[0]
    if source.is_file():
        return source
    if (
        source.is_dir()
        and source.name.lower().endswith(".zarr")
        and ((source / ".zgroup").is_file() or (source / "zarr.json").is_file())
    ):
        return source
    raise OutputDestinationError(
        "Use an input file or an OME-Zarr store, or choose an output folder yourself."
    )


def is_same_input_folder_answer(answer: str) -> bool:
    """Recognize only an explicit local destination choice, never infer consent."""
    if not isinstance(answer, str):
        return False
    normalized = " ".join(answer.strip().casefold().split()).rstrip(".! ")
    return normalized in _SAME_FOLDER_ANSWERS


def resolve_output_destination(
    answer: str, *, source_paths: Sequence[str | Path] = ()
) -> OutputDestination:
    """Resolve user-entered folder text locally without transmitting that text."""
    if is_same_input_folder_answer(answer):
        return OutputDestination.same_folder_as_input(source_paths)
    return OutputDestination.from_directory(answer, source_paths=source_paths)


def recheck_output_destination(destination: OutputDestination) -> None:
    """Fail closed if a selected physical folder/source was replaced or moved."""
    if type(destination) is not OutputDestination:
        raise TypeError("An explicit OutputDestination is required.")
    directory = _absolute_local_path(destination.directory)
    if _identity(directory) != destination._directory_identity:
        raise OutputDestinationError("The output folder changed. Choose it again.")
    for text, expected in zip(
        destination.source_paths, destination._source_identities, strict=True
    ):
        source = _absolute_local_path(text)
        if _identity(source) != expected:
            raise OutputDestinationError("The input source changed. Review it again.")


def validate_output_target(destination: OutputDestination, target: str | Path) -> Path:
    """Preflight a new direct child; the writer still must publish exclusively."""
    recheck_output_destination(destination)
    directory = Path(destination.directory)
    path = Path(target)
    # Do not resolve a candidate: that could conceal an existing symlink.
    if (
        not path.is_absolute()
        or path.parent != directory
        or path.name in {"", ".", ".."}
    ):
        raise OutputDestinationError("Outputs must stay in the chosen output folder.")
    if path in tuple(Path(value) for value in destination.source_paths):
        raise OutputDestinationError("An output must not replace its original input.")
    try:
        path.lstat()
    except FileNotFoundError:
        return path
    except OSError as exc:
        raise OutputDestinationError("The output filename cannot be checked.") from exc
    raise OutputDestinationError(
        "An output with this name already exists. Choose another folder or node name; "
        "existing files will not be overwritten."
    )


def output_path_for_node(
    destination: OutputDestination, node_key: str, format: str
) -> Path:
    """Use a bounded proposal key, never a model-authored filename or path."""
    if not isinstance(node_key, str) or _LOCAL_KEY.fullmatch(node_key) is None:
        raise OutputDestinationError("The output node has an invalid local name.")
    if format not in _EXTENSIONS:
        raise OutputDestinationError("Choose a concrete supported output format.")
    # The prefix also excludes Windows device names such as CON, NUL and COM1.
    target = Path(destination.directory) / f"vipp-{node_key}{_EXTENSIONS[format]}"
    return validate_output_target(destination, target)


def bind_output_parameters(
    parameters: Mapping[str, object],
    destination: OutputDestination,
    node_key: str,
) -> dict[str, object]:
    """Replace one exact placeholder with the user-approved local filename.

    Call this only after normal proposal validation. Enabled writes and a
    concrete image/mesh format must be visible in the accepted draft; the
    assistant cannot opt into overwriting or supply a different local path.
    Adding this node does not constitute permission to execute it.
    """
    if not isinstance(parameters, Mapping):
        raise OutputDestinationError("The output node parameters must be an object.")
    if parameters.get("path") != OUTPUT_DIRECTORY_PLACEHOLDER:
        raise OutputDestinationError("The assistant cannot choose a local output path.")
    if parameters.get("enabled", "off") != "on":
        raise OutputDestinationError("The draft must explicitly enable its Save node.")
    if parameters.get("overwrite", "no") != "no":
        raise OutputDestinationError("Assistant output nodes cannot overwrite files.")
    format = parameters.get("format", "auto")
    if not isinstance(format, str):
        raise OutputDestinationError("Choose a concrete supported output format.")
    target = output_path_for_node(destination, node_key, format)
    return {**parameters, "path": str(target), "overwrite": "no"}


__all__ = [
    "OUTPUT_DIRECTORY_PLACEHOLDER",
    "OutputDestination",
    "OutputDestinationError",
    "bind_output_parameters",
    "is_same_input_folder_answer",
    "output_path_for_node",
    "recheck_output_destination",
    "resolve_output_destination",
    "validate_output_target",
]
