"""The review helper selects a safe bundled workflow, not an existing session."""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def launcher(monkeypatch):
    scripts = Path(__file__).resolve().parents[3] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "launch_vipp_image_review_test", scripts / "launch_vipp_image_review.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "example",
    (
        "review-channels-2d",
        "review-mask-3d",
        "review-labels-time-series",
        "review-rgb-index-3d",
    ),
)
def test_review_launcher_delegates_only_registered_example(
    launcher, monkeypatch, example
):
    requested = []
    monkeypatch.setattr(launcher, "launch_workflow", requested.append)
    launcher.main([example])
    assert requested == [[example, "review"]]


def test_review_launcher_defaults_to_mask_volume(launcher, monkeypatch):
    requested = []
    monkeypatch.setattr(launcher, "launch_workflow", requested.append)
    launcher.main([])
    assert requested == [["review-mask-3d", "review"]]


def test_review_launcher_rejects_external_or_unknown_workflow(launcher, monkeypatch):
    requested = []
    monkeypatch.setattr(launcher, "launch_workflow", requested.append)
    with pytest.raises(SystemExit):
        launcher.main(["unreviewed-file.json"])
    assert requested == []
