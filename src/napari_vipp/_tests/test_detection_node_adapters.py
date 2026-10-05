"""Runtime state injection and declared two-port detection adapter seams."""

import numpy as np
import pytest

from napari_vipp.core import detection
from napari_vipp.core.detection_nodes import (
    DETECTION_RUNTIME_KEYWORDS,
    detection_node_specs,
    find_peaks_node,
    template_match_node,
)


def test_template_adapter_returns_only_declared_arrays_and_forwards_runtime(
    monkeypatch,
):
    inputs, states, progress = [object(), object()], [object(), object()], object()
    outputs = (np.ones((2, 3)), np.ones((2, 3), dtype=bool))
    seen = {}

    def capture(*args, **kwargs):
        seen.update(args=args, kwargs=kwargs)
        return outputs

    monkeypatch.setattr(detection, "template_match_arrays", capture)
    assert (
        template_match_node(inputs, input_states=states, progress=progress) is outputs
    )
    assert seen["args"] == tuple(inputs)
    assert seen["kwargs"] == dict(
        search_state=states[0], template_state=states[1], progress_context=progress
    )


def test_peaks_adapter_passes_explicit_optional_mask_and_parameters(monkeypatch):
    inputs, states, progress = [object(), object()], [object(), object()], object()
    seen = {}

    def capture(*args, **kwargs):
        seen.update(args=args, kwargs=kwargs)
        return "table"

    monkeypatch.setattr(detection, "find_peaks", capture)
    assert (
        find_peaks_node(
            inputs,
            input_states=states,
            use_mask=True,
            minimum_value=0.7,
            progress=progress,
        )
        == "table"
    )
    assert seen["args"] == tuple(inputs)
    assert seen["kwargs"]["minimum_value"] == 0.7
    assert seen["kwargs"]["mask_state"] is states[1]
    assert "use_mask" not in seen["kwargs"]
    assert find_peaks_node(inputs[:1], input_states=states[:1]) == "table"
    assert seen["args"] == (inputs[0], None)
    assert seen["kwargs"]["mask_state"] is None


@pytest.mark.parametrize("use_mask, count", [(False, 2), (True, 1), ("yes", 2)])
def test_mask_port_is_never_inferred_or_silently_ignored(use_mask, count):
    with pytest.raises(ValueError, match="Connect Image"):
        find_peaks_node(
            [object()] * count, input_states=[object()] * count, use_mask=use_mask
        )


def test_specs_are_manual_two_output_match_and_table_first_peaks():
    match, peaks = detection_node_specs()
    assert match.execution_policy == peaks.execution_policy == "manual"
    assert tuple(port.name for port in match.output_ports) == ("scores", "valid_mask")
    assert peaks.output_type == "table"
    assert dict(DETECTION_RUNTIME_KEYWORDS) == {
        "template_match": frozenset({"input_states"}),
        "find_peaks": frozenset({"input_states"}),
    }
