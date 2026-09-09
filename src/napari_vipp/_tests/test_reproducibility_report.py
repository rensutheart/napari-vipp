import pytest

from napari_vipp.core.reproducibility_report import (
    render_reproducibility_readme,
    render_reproducibility_report,
)


def test_recorded_package_explains_the_two_opening_modes():
    data = {
        "package_kind": "recorded_batch_run",
        "reproduction": {"available": True},
    }
    for text in (
        render_reproducibility_report(data),
        render_reproducibility_readme(data),
    ):
        assert "Choose Reproduce original run" in text
        assert "Use workflow on new data" in text
        assert "Check batch" in text


def test_run_version_override_is_a_visible_qualification_not_a_success_claim():
    report = render_reproducibility_report(
        {
            "package_kind": "recorded_batch_run",
            "reproduction_check": {
                "matched_count": 3,
                "mismatch_count": 0,
                "recorded_vipp_version": "0.14.0",
                "current_vipp_version": "0.15.0a2",
                "version_override_used": True,
            },
        }
    )
    assert "3 inputs matched the earlier run" in report
    assert "author explicitly accepted" in report
    assert "not a verified same-version reproduction" in report
    assert "checks do not guarantee identical results" in report


def test_recipe_is_not_presented_as_completed_run():
    report = render_reproducibility_report({"summary": {"nodes": 0}})
    assert "Workflow recipe" in report
    assert "does not establish" in report
    assert "Run results" not in report
    assert "<strong>0</strong>" in report
    assert "Outputs saved" not in report


def test_run_reports_failure_and_absent_evidence_truthfully():
    report = render_reproducibility_report(
        {
            "package_kind": "recorded_batch_run",
            "summary": {"failed": 1, "saved_outputs": 0},
            "items": [
                {"id": "sample", "status": "failed", "message": "Missing calibration"}
            ],
        }
    )
    assert "Recorded batch run" in report
    assert "Missing calibration" in report
    assert "<strong>0</strong>" in report
    assert "No run-time environment was recorded" in report
    assert "Completed" not in report


def test_all_dynamic_values_are_escaped_and_document_is_offline():
    bad = '</style><script>alert(1)</script><img src="https://bad.test">'
    report = render_reproducibility_report(
        {
            "title": bad,
            "notes": bad,
            "created_at": bad,
            "workflow": [{"id": bad, "parameters": {bad: bad}}],
            "sources": [{"name": bad, "selector": bad}],
            "files": [{"path": bad, "description": bad}],
            "privacy": {bad: bad},
            "omissions": [bad],
            "limitations": [bad],
        }
    )
    assert bad not in report
    assert "<script" not in report
    assert "<img" not in report
    assert "&lt;script&gt;" in report
    assert "Content-Security-Policy" in report
    assert "default-src 'none'" in report


def test_long_record_table_states_its_limit():
    report = render_reproducibility_report(
        {
            "sources": [{"name": f"source-{i}"} for i in range(101)],
        }
    )
    assert "first 100 inputs of 101" in report
    assert "source-100" not in report
    assert "report.json" in report


def test_recipe_readme_explains_missing_data_without_batch_instructions():
    readme = render_reproducibility_readme({})
    assert "not a run record" in readme
    assert "No source images or result files" in readme
    assert "python batch-runner.py" not in readme
    assert "Check batch" not in readme
    assert "Source node" in readme
    assert "Save outputs" in readme


def test_batch_readme_keeps_runner_optional_and_no_resume_truthful():
    readme = render_reproducibility_readme({"package_kind": "recorded_batch_run"})
    assert "python batch-runner.py --help" in readme
    assert "workflow.json does not update that script" in readme.replace("\n", " ")
    assert "cannot resume a batch" in readme
    assert readme.index("Choose Check batch") < readme.index("python batch-runner.py")


def test_reused_outputs_and_pending_items_do_not_look_newly_calculated():
    report = render_reproducibility_report(
        {
            "package_kind": "recorded_batch_run",
            "summary": {"saved_outputs": 0, "reused_outputs": 3, "pending": 1},
            "items": [
                {
                    "name": "sample",
                    "status": "completed",
                    "outputs": [
                        {
                            "tag": "Labels",
                            "status": "completed",
                            "provenance_status": "verified_reused",
                            "format": "npy",
                        }
                    ],
                }
            ],
        }
    )
    assert "Outputs reused" in report
    assert "Pending" in report
    assert "verified reuse from earlier run" in report


@pytest.mark.parametrize("message", [None, ""])
def test_empty_item_message_has_no_dash_but_keeps_output_details(message):
    report = render_reproducibility_report(
        {
            "package_kind": "recorded_batch_run",
            "items": [
                {
                    "name": "sample",
                    "status": "completed",
                    "message": message,
                    "outputs": [
                        {"tag": "Labels", "status": "completed", "format": "npy"}
                    ],
                }
            ],
        }
    )
    results = report.split("<h2>Run results</h2>", 1)[1].split("</section>", 1)[0]
    assert "—" not in results
    assert "Labels · completed · npy" in results
    assert "<td><p>Labels" in results


def test_item_failure_message_remains_visible_and_escaped():
    report = render_reproducibility_report(
        {
            "package_kind": "recorded_batch_run",
            "items": [
                {
                    "name": "sample",
                    "status": "failed",
                    "message": 'Missing <calibration> for "sample"',
                }
            ],
        }
    )
    assert "Missing &lt;calibration&gt; for &quot;sample&quot;" in report
    assert "<td>failed</td>" in report


def _environment():
    return {
        "run": {"packages": {"napari-vipp": "0.22.0"}},
        "export": {"packages": {"napari-vipp": "0.23.0"}},
    }


@pytest.mark.parametrize("markdown", [False, True])
def test_recorded_repeat_steps_are_gui_first_and_link_the_recorded_release(markdown):
    data = {"package_kind": "recorded_batch_run", "environment": _environment()}
    if markdown:
        document = render_reproducibility_readme(data)
        steps = document.split("## How to repeat the analysis", 1)[1].split(
            "## Optional command-line use", 1
        )[0]
        assert "[workflow.json](workflow.json)" in steps
    else:
        document = render_reproducibility_report(data)
        steps = document.split("<h2>How to repeat the analysis</h2>", 1)[1].split(
            "</section>", 1
        )[0]
        assert '<a href="workflow.json">workflow.json</a>' in steps
    expected_order = [
        "Extract the ZIP",
        "Install VIPP",
        "choose Open",
        "batch workspace opens automatically",
        "In Setup",
        "input folder for each source",
        "new output folder",
        "Choose Check batch",
        "then choose Run",
    ]
    assert [steps.index(text) for text in expected_order] == sorted(
        steps.index(text) for text in expected_order
    )
    assert "releases/tag/v0.22.0" in steps
    assert "releases/tag/v0.23.0" not in steps
    assert "installer for your operating system" in steps
    for technical in ("selectors", "hashes", "batch-config.json", "Python", "README"):
        assert technical not in steps
    assert "Identical results are not guaranteed" in steps


@pytest.mark.parametrize("markdown", [False, True])
def test_recipe_repeat_steps_use_sources_and_save_outputs_not_batch(markdown):
    data = {"package_kind": "workflow_recipe", "environment": _environment()}
    document = (
        render_reproducibility_readme(data)
        if markdown
        else render_reproducibility_report(data)
    )
    steps = (
        document.split("## How to repeat the analysis", 1)[1].split(
            "## Optional command-line use", 1
        )[0]
        if markdown
        else document.split("<h2>How to repeat the analysis</h2>", 1)[1].split(
            "</section>", 1
        )[0]
    )
    assert "releases/tag/v0.23.0" in steps
    assert "Source node" in steps
    assert "Save outputs" in steps
    assert "choose Calculate all" in steps
    assert "batch" not in steps.lower()


def test_installation_does_not_substitute_export_version_for_missing_run_version():
    report = render_reproducibility_report(
        {
            "package_kind": "recorded_batch_run",
            "environment": {"export": _environment()["export"]},
        }
    )
    steps = report.split("<h2>How to repeat the analysis</h2>", 1)[1].split(
        "</section>", 1
    )[0]
    assert "releases/tag/v0.23.0" not in steps
    assert "author" in steps.lower()


def test_technical_identities_and_software_follow_basic_instructions():
    report = render_reproducibility_report(
        {"sources": [{"name": "input.tif", "sha256": "a" * 64}]}
    )
    assert (
        report.index("How to repeat the analysis")
        < report.index("Package contents")
        < report.index("Technical input records")
        < report.index("Software and execution evidence")
    )
    assert "a" * 64 in report


@pytest.mark.parametrize("fixed_sources", [[], ["reference"]])
def test_recipe_with_attached_batch_uses_workspace_and_conditional_fixed_reference_note(
    fixed_sources,
):
    data = {
        "package_kind": "workflow_recipe",
        "environment": _environment(),
        "batch": {
            "embedded_workspace": True,
            "requires_full_check": True,
            "collection_source_ids": ["sample"],
            "fixed_source_ids": fixed_sources,
        },
    }
    for document in (
        render_reproducibility_report(data),
        render_reproducibility_readme(data),
    ):
        assert "batch workspace opens automatically" in document
        assert "input folder for each source" in document
        assert "Choose Check batch" in document
        assert "choose Run batch" in document
        assert "choose Calculate all" not in document
        assert "python batch-runner.py --help" in document
        assert ("fixed reference files" in document) is bool(fixed_sources)
        assert "releases/tag/v0.23.0" in document


def test_missing_data_note_is_stated_once_in_repeat_instructions():
    readme = render_reproducibility_readme({"environment": _environment()})
    assert readme.count("obtain the input data separately") == 1
    assert readme.count("No source images or result files") == 1


def test_omission_section_explains_notes_and_separate_data_sharing():
    report = render_reproducibility_report({})
    section = report.split("<h2>Deliberately left out</h2>", 1)[1].split(
        "</section>", 1
    )[0]
    assert (
        "Explanatory notes saved in the workflow are included in workflow.json"
        in section
    )
    assert (
        "Source images, result files, meshes/tables, thumbnails and previews "
        "are deliberately not read or included."
    ) in section
    assert "Share the input data and any results separately" in section
    assert "Intermediate results are not included" in section


@pytest.mark.parametrize("anonymous", [False, True])
def test_sharing_review_uses_plain_language_instead_of_internal_flags(anonymous):
    report = render_reproducibility_report(
        {
            "privacy": {
                "paths_redacted": True,
                "anonymise_filenames": anonymous,
                "original_records_included": False,
            },
            "changes": ["Folder locations replaced."],
        }
    )
    section = report.split("<h2>Sharing review</h2>", 1)[1].split("</section>", 1)[0]
    expected = (
        "Filenames have been replaced with anonymous names."
        if anonymous
        else "Original filenames are included."
    )
    assert expected in section
    assert "author-supplied notes" in section
    assert "Changes made for sharing" in section
    assert "Folder locations replaced." in section
    for jargon in (
        "paths redacted",
        "anonymise filenames:",
        "original records included",
        "True",
        "False",
    ):
        assert jargon not in section
