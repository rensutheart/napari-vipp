from __future__ import annotations

import warnings

from napari_vipp.core.gpu import cupy_imports

_RAWKERNEL_NOTICE = (
    "cupyx.jit.rawkernel is experimental. The interface can change in the future."
)


def test_signal_import_hides_only_exact_upstream_future_warning(monkeypatch):
    expected_module = object()

    def load(name: str):
        assert name == "cupyx.scipy.signal"
        warnings.warn_explicit(
            _RAWKERNEL_NOTICE,
            FutureWarning,
            "cupyx/jit/_interface.py",
            247,
            module="cupyx.jit._interface",
        )
        warnings.warn_explicit(
            "A different CuPyX compatibility warning.",
            FutureWarning,
            "cupyx/jit/_interface.py",
            248,
            module="cupyx.jit._interface",
        )
        warnings.warn_explicit(
            _RAWKERNEL_NOTICE,
            UserWarning,
            "cupyx/jit/_interface.py",
            249,
            module="cupyx.jit._interface",
        )
        warnings.warn_explicit(
            _RAWKERNEL_NOTICE,
            FutureWarning,
            "another/provider.py",
            250,
            module="another.provider",
        )
        return expected_module

    monkeypatch.setattr(cupy_imports.importlib, "import_module", load)

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        result = cupy_imports.import_cupyx_signal_module()

    assert result is expected_module
    assert [(record.category, str(record.message)) for record in records] == [
        (FutureWarning, "A different CuPyX compatibility warning."),
        (UserWarning, _RAWKERNEL_NOTICE),
        (FutureWarning, _RAWKERNEL_NOTICE),
    ]


def test_signal_import_propagates_failures(monkeypatch):
    failure = ImportError("simulated CuPyX import failure")

    def fail(_name: str):
        raise failure

    monkeypatch.setattr(cupy_imports.importlib, "import_module", fail)

    try:
        cupy_imports.import_cupyx_signal_module()
    except ImportError as exc:
        assert exc is failure
    else:  # pragma: no cover - explicit failure message is clearer than pytest.raises
        raise AssertionError("The CuPyX import failure was hidden.")
