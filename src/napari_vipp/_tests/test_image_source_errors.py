from __future__ import annotations

from pathlib import Path

import pytest

from napari_vipp.core.file_sources import load_frozen_file_source_snapshot
from napari_vipp.core.io.errors import (
    ImageSourceError,
    ImageSourceErrorCode,
    as_image_source_error,
)
from napari_vipp.core.io.microscope import OptionalMicroscopeReaderError
from napari_vipp.core.progress import OperationCancelled
from napari_vipp.core.source_identity import SourceChangedError
from napari_vipp.ui import file_sources as ui_file_sources
from napari_vipp.ui.file_sources import SourceFileLoadSpec, SourceFileLoadWorker


@pytest.mark.parametrize(
    ("error", "source_path", "code"),
    (
        (
            OptionalMicroscopeReaderError(
                "Reading .czi requires an optional dependency.",
                suffix=".czi",
                format_name="zeiss-czi",
                module_name="czifile",
                install_command='pip install "napari-vipp[czi]"',
            ),
            "image.czi",
            ImageSourceErrorCode.OPTIONAL_DEPENDENCY,
        ),
        (
            RuntimeError("Bio-Formats could not start the Java JVM."),
            "image.vsi",
            ImageSourceErrorCode.JAVA_BIOFORMATS_READINESS,
        ),
        (
            FileNotFoundError(2, "missing", "frame_t_0.ets"),
            "image.vsi",
            ImageSourceErrorCode.MISSING_COMPANION,
        ),
        (
            EOFError("Source is truncated."),
            "image.oir",
            ImageSourceErrorCode.CORRUPT_SOURCE,
        ),
        (
            ValueError("Axis metadata rank does not match the array."),
            "image.oib",
            ImageSourceErrorCode.CONTRACT_MISMATCH,
        ),
        (
            SourceChangedError("Source changed during execution."),
            "image.nd2",
            ImageSourceErrorCode.SOURCE_CHANGED,
        ),
        (
            OperationCancelled("Source loading cancelled."),
            "image.lif",
            ImageSourceErrorCode.CANCELLED,
        ),
        (
            MemoryError("Unable to allocate the source array."),
            "image.ims",
            ImageSourceErrorCode.MEMORY_PREFLIGHT,
        ),
    ),
)
def test_image_source_error_classification(
    error: Exception,
    source_path: str,
    code: ImageSourceErrorCode,
) -> None:
    classified = as_image_source_error(
        error,
        path=source_path,
        stage="open",
        item=3,
    )

    assert classified.code is code
    assert classified.stage == "open"
    assert classified.path == source_path
    assert classified.item == "3"
    assert str(error).strip() in classified.display_text
    assert classified.remediation
    assert classified.to_dict()["code"] == code.value


def test_image_source_error_sanitizes_display_text() -> None:
    error = ImageSourceError(
        ImageSourceErrorCode.READ_FAILED,
        "unsafe\nreader\x00detail",
        remediation="Retry.\r\n",
    )

    assert error.detail == "unsafe reader detail"
    assert error.display_text == "unsafe reader detail Retry."

    no_remediation = ImageSourceError(
        ImageSourceErrorCode.READ_FAILED,
        "plain failure",
    )
    assert no_remediation.remediation == ""
    assert no_remediation.display_text == "plain failure"


def test_bare_jvm_failure_is_java_bioformats_readiness() -> None:
    class JVMNotFoundException(RuntimeError):
        pass

    classified = as_image_source_error(
        JVMNotFoundException("No JVM shared library was found."),
        path="sample.vsi",
    )

    assert classified.code is ImageSourceErrorCode.JAVA_BIOFORMATS_READINESS


def test_core_loader_preserves_original_type_and_phase_context(tmp_path: Path) -> None:
    source = tmp_path / "source.oib"
    source.write_bytes(b"reader fixture")

    def mismatched_reader(path, *, series_index=0):
        raise ValueError("Axis metadata rank does not match the array.")

    with pytest.raises(ValueError) as caught:
        load_frozen_file_source_snapshot(
            source,
            2,
            reader=mismatched_reader,
        )

    classified = as_image_source_error(caught.value)
    assert classified.code is ImageSourceErrorCode.CONTRACT_MISMATCH
    assert classified.stage == "open"
    assert classified.path == str(source.resolve())
    assert classified.format == "oib"
    assert classified.item == "2"


def test_core_loader_classifies_cancellation_before_hashing(tmp_path: Path) -> None:
    source = tmp_path / "source.nd2"
    source.write_bytes(b"reader fixture")

    with pytest.raises(OperationCancelled) as caught:
        load_frozen_file_source_snapshot(
            source,
            0,
            reader=lambda *_args, **_kwargs: None,
            cancel_callback=lambda: True,
        )

    classified = as_image_source_error(caught.value)
    assert classified.code is ImageSourceErrorCode.CANCELLED
    assert classified.stage == "source-validation"
    assert classified.path == str(source.resolve())


def test_source_worker_keeps_legacy_text_and_structured_error(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_load(*_args, **_kwargs):
        raise RuntimeError("Bio-Formats could not initialize the Java JVM.")

    monkeypatch.setattr(
        ui_file_sources,
        "load_frozen_file_source_snapshot",
        fail_load,
    )
    spec = SourceFileLoadSpec(
        node_id="input",
        path="sample.vsi",
        series_index=1,
        cache_key=("sample.vsi", 1),
    )
    worker = SourceFileLoadWorker(7, (spec,), reader=lambda *_args, **_kwargs: None)
    observed = []
    worker.signals.finished.connect(observed.append)

    worker.run()

    assert len(observed) == 1
    result = observed[0]
    assert isinstance(result.error, str)
    assert "Bio-Formats could not initialize" in result.error
    assert result.node_id == "input"
    assert result.source_error is not None
    assert result.source_error.code is ImageSourceErrorCode.JAVA_BIOFORMATS_READINESS
    assert result.source_error.path == "sample.vsi"
    assert result.source_error.item == "1"
