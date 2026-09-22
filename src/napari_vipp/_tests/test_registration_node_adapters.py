"""Graph registration adapters never trust display names as source identity."""

from dataclasses import replace

import numpy as np
import pytest

from napari_vipp.core import registration
from napari_vipp.core.metadata import SourceMetadata, image_state_from_array
from napari_vipp.core.registration_nodes import (
    apply_transform_node,
    estimate_registration_node,
)


def _named_image():
    image = np.ones((16, 20), dtype=np.float32)
    image.setflags(write=False)
    state = image_state_from_array(
        image, layer_metadata={"axes": "YX"}, source_name="Same visible image name"
    )
    return image, state


@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("adapter", ["estimate", "apply"])
def test_registration_adapters_reject_named_only_or_missing_cached_identity(
    adapter, missing, monkeypatch
):
    image, state = _named_image()
    assert state.source_name and not state.source.source_uuid and not state.source.uri
    if missing:
        state = None

    def forbidden(*_args, **_kwargs):
        pytest.fail("The numerical kernel must not run without a source identity.")

    monkeypatch.setattr(registration, "estimate_registration", forbidden)
    monkeypatch.setattr(registration, "apply_transform", forbidden)
    with pytest.raises(ValueError, match="Refresh the source image.*recalculate"):
        if adapter == "estimate":
            estimate_registration_node(
                [image], input_states=[state], mode="Time series"
            )
        else:
            apply_transform_node([image, object()], input_states=[state, object()])


def test_pairwise_adapter_checks_reference_identity_before_estimation(monkeypatch):
    image, anonymous = _named_image()
    identified = replace(anonymous, source=SourceMetadata(source_uuid="moving-frame"))
    calls = []
    monkeypatch.setattr(
        registration,
        "estimate_registration",
        lambda *args, **kwargs: calls.append(args),
    )
    with pytest.raises(ValueError, match="no registration source identity.*Refresh"):
        estimate_registration_node(
            [image, image], input_states=[identified, anonymous], mode="Two images"
        )
    assert not calls


@pytest.mark.parametrize(
    "source",
    [
        SourceMetadata(source_uuid="known-frame"),
        SourceMetadata(uri="sample:known-image"),
    ],
    ids=["uuid", "uri"],
)
def test_estimate_adapter_preserves_identified_states_and_forwards_controls(
    source, monkeypatch
):
    image, state = _named_image()
    moving = replace(state, source=source)
    reference = replace(state, source=SourceMetadata(source_uuid="reference-frame"))
    expected = (object(), object())
    captured = {}
    progress = object()

    def capture(*args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        return expected

    monkeypatch.setattr(registration, "estimate_registration", capture)
    result = estimate_registration_node(
        [image, image],
        input_states=[moving, reference],
        progress=progress,
        mode="Two images",
        model="Rigid",
        channel=1,
        reference_channel=0,
    )
    assert result is expected
    assert captured["args"][0] is image and captured["args"][1] is image
    assert captured["kwargs"]["moving_state"] is moving
    assert captured["kwargs"]["reference_state"] is reference
    assert captured["kwargs"]["progress_context"] is progress
    assert captured["kwargs"]["model"] == "Rigid"
    assert captured["kwargs"]["channel"] == 1
    assert moving.source is source


@pytest.mark.parametrize(
    "source",
    [
        SourceMetadata(source_uuid="known-frame"),
        SourceMetadata(uri="sample:known-image"),
    ],
    ids=["uuid", "uri"],
)
def test_apply_adapter_checks_only_the_image_and_preserves_transform_payload(
    source, monkeypatch
):
    image, state = _named_image()
    state = replace(state, source=source)
    transform, transform_state, progress = object(), object(), object()
    expected = (object(), object())
    captured = {}

    def capture(*args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        return expected

    monkeypatch.setattr(registration, "apply_transform", capture)
    result = apply_transform_node(
        [image, transform],
        input_states=[state, transform_state],
        progress=progress,
        interpolation="Nearest",
        outside_value=0.0,
    )
    assert result is expected
    assert captured["args"][0] is image and captured["args"][1] is transform
    assert captured["kwargs"]["image_state"] is state
    assert captured["kwargs"]["progress_context"] is progress
    assert captured["kwargs"]["interpolation"] == "Nearest"
    assert state.source is source
