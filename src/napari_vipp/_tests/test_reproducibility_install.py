"""Version guidance never confuses current export software with a past run."""

import pytest

from napari_vipp.core.reproducibility_install import installation_guidance
from napari_vipp.core.updates import RELEASES_URL


def _environment(run="0.15.0a2", export="0.16.0"):
    return {
        "run": {"packages": {"napari-vipp": run}},
        "export": {"packages": {"napari-vipp": export}},
    }


def test_recorded_version_is_used_instead_of_export_version():
    guidance = installation_guidance(_environment(), recorded=True)
    assert guidance["version"] == "0.15.0a2"
    assert guidance["url"] == RELEASES_URL + "/tag/v0.15.0a2"
    assert "0.15.0a2" in guidance["label"]
    assert "Unpublished changes" in guidance["note"]
    recipe = installation_guidance(_environment(), recorded=False)
    assert recipe["version"] == "0.16.0"


@pytest.mark.parametrize("raw", [None, "", "garbage", "0.0.0", "https://bad.test", {},
                                 '0.15\" onclick=\"bad()', "a" * 200])
def test_missing_or_invalid_run_version_never_uses_exporter(raw):
    guidance = installation_guidance(_environment(run=raw), recorded=True)
    assert guidance["version"] == ""
    assert guidance["url"] == RELEASES_URL
    assert "Ask the author" in guidance["note"]


@pytest.mark.parametrize(
    "environment", [None, {}, {"run": []}, {"run": {"packages": []}}]
)
def test_incomplete_environment_records_are_safe(environment):
    assert installation_guidance(environment, recorded=True)["url"] == RELEASES_URL


@pytest.mark.parametrize("version", ["0.15.0.dev1", "0.15.0a2+local"])
def test_development_build_does_not_invent_release_tag(version):
    guidance = installation_guidance(_environment(run=version), recorded=True)
    assert guidance["url"] == RELEASES_URL
    assert "development build" in guidance["note"]
