"""The example chooser's learning copy is complete and presentation-only."""

from dataclasses import FrozenInstanceError, fields

import pytest

from napari_vipp.ui.example_guidance import (
    EXAMPLE_GUIDANCE_BY_ID,
    ExampleGuidance,
)
from napari_vipp.ui.examples import EXAMPLE_WORKFLOWS, ExampleWorkflowSpec


def test_every_bundled_example_has_curated_guidance():
    assert len(EXAMPLE_WORKFLOWS) == 22
    assert set(EXAMPLE_GUIDANCE_BY_ID) == {spec.id for spec in EXAMPLE_WORKFLOWS}
    for spec in EXAMPLE_WORKFLOWS:
        assert spec.guidance is EXAMPLE_GUIDANCE_BY_ID[spec.id]
        for name in ("purpose", "data", "try_this", "caution"):
            assert getattr(spec.guidance, name).strip()
        for name in ("explore", "results"):
            bullets = getattr(spec.guidance, name)
            assert isinstance(bullets, tuple)
            assert bullets
            assert all(isinstance(bullet, str) and bullet.strip() for bullet in bullets)
        assert 3 <= len(spec.guidance.explore) <= 4
        assert all(
            isinstance(getattr(spec.guidance, field.name), str)
            for field in fields(ExampleGuidance)
            if field.name not in {"explore", "results"}
        )


def test_guidance_and_catalogue_are_immutable():
    guidance = EXAMPLE_GUIDANCE_BY_ID["label-cleanup"]
    with pytest.raises(FrozenInstanceError):
        guidance.purpose = "Changed"
    with pytest.raises(FrozenInstanceError):
        guidance.explore += ("Changed",)
    with pytest.raises(TypeError):
        guidance.explore[0] = "Changed"
    with pytest.raises(FrozenInstanceError):
        guidance.results += ("Changed",)
    with pytest.raises(TypeError):
        guidance.results[0] = "Changed"
    with pytest.raises(TypeError):
        EXAMPLE_GUIDANCE_BY_ID["label-cleanup"] = guidance


def test_legacy_and_custom_specs_can_omit_guidance():
    spec = ExampleWorkflowSpec(
        "custom", "Category", "Title", "custom.json", (), "Description", True
    )
    assert spec.generated_batch_demo is True
    assert spec.guidance is None
    guidance = ExampleGuidance(
        "Purpose", "Data", ("Explore",), ("Results",), "Try", "Note"
    )
    described = ExampleWorkflowSpec(
        "custom",
        "Category",
        "Title",
        "custom.json",
        (),
        "Description",
        False,
        guidance,
    )
    assert described.guidance is guidance
    for name in (
        "method_name",
        "method_description",
        "method_meaning",
        "paper_authors",
        "paper_journal",
        "paper_url",
        "chooser_category",
    ):
        assert getattr(guidance, name) == ""


def test_only_five_interface_fixtures_override_the_chooser_category():
    developer_ids = {
        "exhaustive-inspector",
        "graph-authoring",
        "responsive-crop",
        "safe-node-bypass",
        "general-node-bypass",
    }
    assert {
        spec.id for spec in EXAMPLE_WORKFLOWS if spec.guidance.chooser_category
    } == developer_ids
    for spec in EXAMPLE_WORKFLOWS:
        if spec.id in developer_ids:
            assert spec.guidance.chooser_category == "Developer & testing workflows"
            assert spec.category != spec.guidance.chooser_category


def test_racc_guidance_explains_the_method_and_links_the_original_paper():
    guidance = EXAMPLE_GUIDANCE_BY_ID["racc-colocalization"]
    assert guidance.method_name == (
        "Regression adjusted colocalisation colour mapping (RACC)"
    )
    assert "Deming regression" in guidance.method_description
    assert "qualitative" in guidance.method_meaning
    assert "not a probability" in guidance.method_meaning
    assert guidance.paper_authors == "Theart, Loos & Niesler (2019)"
    assert guidance.paper_journal == "PLOS ONE 14(11): e0225141."
    assert guidance.paper_url == "https://doi.org/10.1371/journal.pone.0225141"
    explore = " ".join(guidance.explore)
    assert "Magma" in explore
    assert "43,970.51" in explore
    assert "48,073.03" in explore
    assert "30,000" in explore
    assert "Theta" in explore
    assert "45° to 60°" in guidance.try_this


def test_overlap_guidance_distinguishes_regions_from_organelles_and_roi():
    guidance = EXAMPLE_GUIDANCE_BY_ID["colocalization-overlap"]
    assert "20 to 1 voxel" in guidance.try_this
    assert "not the original organelles" in guidance.caution
    assert "not restricted to the ROI" in guidance.caution
    assert not guidance.method_name
    assert not guidance.paper_url


def test_batch_guidance_explains_the_working_folder_and_saved_files():
    guidance = EXAMPLE_GUIDANCE_BY_ID["batch-provenance"]
    assert "Choose a folder" in guidance.try_this
    assert "creates a working folder" in guidance.caution
    assert "writes result files" in guidance.caution


def test_guidance_does_not_repeat_removed_download_reassurances():
    for guidance in EXAMPLE_GUIDANCE_BY_ID.values():
        text = " ".join(
            getattr(guidance, field.name)
            for field in fields(ExampleGuidance)
            if field.name not in {"explore", "results"}
        ).lower()
        text += " " + " ".join(guidance.explore).lower()
        text += " " + " ".join(guidance.results).lower()
        assert "included with vipp" not in text
        assert "no download needed" not in text


def test_explore_bullets_are_short_plain_text_without_list_markup():
    for guidance in EXAMPLE_GUIDANCE_BY_ID.values():
        assert len(guidance.explore) == len(set(guidance.explore))
        for bullet in guidance.explore:
            assert bullet == bullet.strip()
            assert "\n" not in bullet
            assert not bullet.startswith(("-", "•", "<li>"))
            assert len(bullet.split()) <= 20


def test_results_use_individual_plain_text_items_including_single_outputs():
    assert EXAMPLE_GUIDANCE_BY_ID["label-cleanup"].results == ("Cleaned object labels",)
    assert EXAMPLE_GUIDANCE_BY_ID["racc-colocalization"].results == (
        "Whole-image RACC map",
        "ROI-restricted RACC map",
    )
    assert EXAMPLE_GUIDANCE_BY_ID["colocalization-overlap"].results == (
        "Overlap overlays",
        "Whole-image and ROI metric tables",
        "Boolean masks",
        "Labelled regions",
        "Object measurements",
    )
    for guidance in EXAMPLE_GUIDANCE_BY_ID.values():
        assert len(guidance.results) == len(set(guidance.results))
        for bullet in guidance.results:
            assert bullet == bullet.strip()
            assert "\n" not in bullet
            assert "·" not in bullet
            assert not bullet.startswith(("-", "•", "<li>"))
