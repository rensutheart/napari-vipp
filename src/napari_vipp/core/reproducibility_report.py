"""Offline, human-readable views of already-sanitized package evidence.

This renderer makes no filesystem or network reads. Escaping is still required:
sharing review and HTML safety are separate from path redaction.
"""

# HTML/CSS templates keep complete elements together for maintainability.
# ruff: noqa: E501

from __future__ import annotations

import html
import json
from collections.abc import Mapping
from datetime import UTC, datetime

from napari_vipp.core.reproducibility_install import (
    INSTALLATION_GUIDE_URL,
    installation_guidance,
)


def _text(value) -> str:
    if value is None:
        return "Not recorded"
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    return html.escape(str(value), quote=True)


def _timestamp(value) -> str:
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if stamp.tzinfo is not None:
            return stamp.astimezone(UTC).strftime("%d %b %Y, %H:%M:%S UTC")
    except (TypeError, ValueError):
        pass
    return _text(value)


def _details(title, value) -> str:
    if not value:
        return ""
    # Qt treats <pre> as non-wrapping even when browser CSS requests wrapping.
    # Explicit line breaks retain readable structure without forcing the
    # preview's entire table wider than the window. Exact JSON remains in the
    # corresponding package member, available through Package contents.
    body = _text(value).replace("\n", "<br>")
    return (
        f"<details><summary>{_text(title)}</summary>"
        '<div style="white-space:normal;font-family:monospace;font-size:12px">'
        f"{body}</div></details>"
    )


def _list(values) -> str:
    return "<ul>" + "".join(f"<li>{_text(v)}</li>" for v in values) + "</ul>"


def _output_summary(outputs) -> str:
    """Readable outcomes; lengthy identities stay in the structured evidence."""
    if not outputs:
        return ""
    rows = []
    for output in outputs:
        if not isinstance(output, Mapping):
            rows.append(_text(output))
            continue
        label = output.get("tag") or output.get("node_id") or "Output"
        status = output.get("status") or "Not recorded"
        if output.get("provenance_status") == "verified_reused":
            status = "verified reuse from earlier run"
        fmt = output.get("format")
        rows.append(
            f"{_text(label)} · {_text(status)}" + (f" · {_text(fmt)}" if fmt else "")
        )
    return "<p>" + "<br>".join(rows) + "</p>"


def _table(headers, rows) -> str:
    return (
        '<div class="table-scroll"><table><thead><tr>'
        + "".join(f"<th>{_text(h)}</th>" for h in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>" for row in rows
        )
        + "</tbody></table></div>"
    )


def _section(title, body) -> str:
    return f"<section><h2>{_text(title)}</h2>{body}</section>"


def _limited(records, name) -> tuple[list, str]:
    records = list(records or [])
    note = (
        f'<p class="muted">Showing the first 100 {name} of {len(records):,}. '
        "The complete record is in report.json.</p>"
        if len(records) > 100
        else ""
    )
    return records[:100], note


def _markdown_text(value: object) -> str:
    value = html.escape(str(value), quote=False)
    for character in ("\\", "`", "*", "_", "[", "]"):
        value = value.replace(character, "\\" + character)
    return value


def _installation_step(data: dict, *, markdown: bool = False) -> str:
    guidance = installation_guidance(
        data.get("environment") or {},
        recorded=data.get("package_kind") == "recorded_batch_run",
    )
    link = (
        f"[{_markdown_text(guidance['label'])}]({guidance['url']})"
        if markdown
        else f'<a href="{_text(guidance["url"])}">{_text(guidance["label"])}</a>'
    )
    step = f"Install VIPP using {link}; choose the installer for your operating system."
    # Unknown/development builds cannot be presented as an exact public release.
    if "/releases/tag/" not in guidance["url"]:
        step += " " + (
            _markdown_text(guidance["note"]) if markdown else _text(guidance["note"])
        )
    return step


def _has_batch_workspace(data: dict) -> bool:
    batch = data.get("batch")
    return data.get("package_kind") == "recorded_batch_run" or (
        isinstance(batch, Mapping) and bool(batch.get("embedded_workspace"))
    )


def _repeat_steps(data: dict, *, markdown: bool = False) -> list[str]:
    """Keep the report and README on the same short, GUI-first path."""
    workflow_link = (
        "[workflow.json](workflow.json)"
        if markdown
        else '<a href="workflow.json">workflow.json</a>'
    )
    steps = [
        "Extract the ZIP into a folder.",
        _installation_step(data, markdown=markdown),
        f"In VIPP, choose Open and select the packaged {workflow_link}.",
    ]
    if _has_batch_workspace(data):
        if (data.get("reproduction") or {}).get("available"):
            steps[2] += (
                " Choose Reproduce original run to check the original inputs, or "
                "Use workflow on new data to analyse another dataset."
            )
        steps[2] += " The batch workspace opens automatically."
        locations = (
            "In Setup, choose the input folder for each source and a new output folder."
        )
        batch = data.get("batch")
        if isinstance(batch, Mapping) and batch.get("fixed_source_ids"):
            locations += " Select fixed reference files in their Image Source nodes."
        steps += [
            locations,
            "Choose Check batch, resolve any reported issues, then choose Run batch.",
        ]
    else:
        steps += [
            "Select each Source node and choose its input data. Set new destinations for Save outputs.",
            "Review the workflow settings, then choose Calculate all.",
        ]
    return steps


def _repeat_note(data: dict) -> str:
    note = "No source images or result files are included; obtain the input data separately. Identical results are not guaranteed."
    if data.get("package_kind") == "recorded_batch_run":
        note += " Start a new batch; the shared records cannot resume a batch."
    return note


def _advanced_runner(data: dict, *, markdown: bool = False) -> str:
    command = (
        "python batch-runner.py --help"
        if _has_batch_workspace(data)
        else "python runner.py --help"
    )
    command = f"`{command}`" if markdown else f"<code>{command}</code>"
    return (
        f"Python is optional. Command-line users can run {command} for options. "
        "runner.py embeds its own workflow: editing workflow.json does not update "
        "that script. Regenerate it in VIPP after changing the workflow."
    )


def render_reproducibility_report(report_data: dict) -> str:
    """Render a standalone document; missing measurements never become zero."""
    data = report_data
    recorded = data.get("package_kind") == "recorded_batch_run"
    kind = "Recorded batch run" if recorded else "Workflow recipe"
    title = data.get("title") or "VIPP analysis"
    summary = data.get("summary") or {}
    sections = []
    intro = (
        "This report describes the archived batch settings and recorded outcomes. "
        "It does not substitute the workflow currently open in VIPP."
        if recorded
        else "This package describes a saved analysis recipe. It does not establish "
        "that the workflow ran successfully or produced particular results."
    )
    metrics = []
    keys = [("nodes", "Workflow nodes")]
    if recorded:
        keys += [
            ("items", "Batch items"),
            ("completed", "Completed"),
            ("failed", "Failed"),
            ("skipped", "Skipped"),
            ("partial", "Partial"),
            ("cancelled", "Cancelled"),
            ("pending", "Pending"),
            ("running", "In progress"),
            ("saved_outputs", "Outputs saved"),
            ("reused_outputs", "Outputs reused"),
        ]
    for key, label in keys:
        if key in summary and summary[key] is not None:
            if (
                key
                in {
                    "skipped",
                    "partial",
                    "cancelled",
                    "pending",
                    "running",
                    "reused_outputs",
                }
                and not summary[key]
            ):
                continue
            metrics.append(
                f'<td class="metric"><strong>{_text(summary[key])}</strong>'
                f"<br>{_text(label)}</td>"
            )
    overview = f"<p>{intro}</p>"
    if metrics:
        # Small table also renders correctly in Qt's limited rich-text engine.
        overview += (
            '<div class="table-scroll"><table class="metrics"><tr>'
            + "".join(metrics)
            + "</tr></table></div>"
        )
    if recorded:
        duration = summary.get("duration_seconds")
        if isinstance(duration, (int, float)):
            overview += (
                f'<p class="muted">Recorded duration: {duration:,.1f} seconds.</p>'
            )
        for key, label in [("started_at", "Started"), ("finished_at", "Finished")]:
            if summary.get(key):
                overview += f'<p class="muted">{label}: {_timestamp(summary[key])}</p>'
    if data.get("notes"):
        overview += f'<div class="note"><h3>Author’s notes</h3><p class="preserve">{_text(data["notes"])}</p></div>'
    sections.append(_section("At a glance", overview))

    audit = data.get("reproduction_check")
    if audit:
        body = (
            f"<p>{_text(audit.get('matched_count'))} inputs matched the earlier run; "
            f"{_text(audit.get('mismatch_count'))} did not match.</p>"
            f"<p>Earlier run: VIPP {_text(audit.get('recorded_vipp_version'))}. "
            f"This run: VIPP {_text(audit.get('current_vipp_version'))}.</p>"
        )
        if audit.get("version_override_used"):
            body += (
                '<p class="callout">The author explicitly accepted a different or '
                "unverifiable VIPP version. This was not a verified same-version "
                "reproduction.</p>"
            )
        body += "<p>These checks do not guarantee identical results.</p>"
        sections.append(_section("Reproduction checks for this run", body))

    if recorded:
        items, note = _limited(data.get("items"), "items")
        rows = []
        for item in items:
            outcomes = _text(item.get("message")) if item.get("message") else ""
            outcomes += _output_summary(item.get("outputs"))
            rows.append(
                [
                    _text(item.get("name") or item.get("id")),
                    _text(item.get("status")),
                    outcomes,
                ]
            )
        body = (
            _table(["Item", "Outcome", "Details"], rows)
            if rows
            else "<p>No item outcomes were recorded.</p>"
        )
        sections.append(_section("Run results", body + note))

    sections.append(
        _section(
            "How to repeat the analysis",
            "<ol>"
            + "".join(f"<li>{step}</li>" for step in _repeat_steps(data))
            + "</ol>"
            + f'<p class="callout">{_text(_repeat_note(data))}</p>',
        )
    )

    nodes, note = _limited(data.get("workflow"), "nodes")
    rows = [
        [
            _text(n.get("title") or n.get("id")),
            _text(n.get("operation")),
            _details("Parameters", n.get("parameters")) or "—",
            _text(n.get("execution_mode")),
        ]
        for n in nodes
    ]
    sections.append(
        _section(
            "Workflow",
            (
                _table(["Node", "Operation", "Settings", "Execution"], rows)
                if rows
                else "<p>The portable workflow file contains the analysis recipe.</p>"
            )
            + note,
        )
    )

    sources, note = _limited(data.get("sources"), "inputs")
    rows = [
        [
            _text(s.get("name") or s.get("id")),
            _text(s.get("selector") or "Not recorded"),
            _text(s.get("sha256") or "Not recorded"),
        ]
        for s in sources
    ]
    input_evidence = _section(
        "Technical input records",
        (
            _table(["Input", "Selection", "Recorded SHA-256"], rows)
            if rows
            else "<p>No file input records were available. Live-layer or embedded inputs must be supplied separately.</p>"
        )
        + note
        + '<p class="muted">Hashes, when present, are recorded evidence. Export does not reopen the images or verify their current contents.</p>',
    )

    files = data.get("files") or []
    sections.append(
        _section(
            "Package contents",
            _table(
                ["File", "Purpose"],
                [[_text(f.get("path")), _text(f.get("description"))] for f in files],
            ),
        )
    )
    sections.append(
        _section(
            "Deliberately left out",
            "<p>Explanatory notes saved in the workflow are included in workflow.json.</p>"
            + _list(
                data.get("omissions")
                or [
                    "Source images, result files, meshes/tables, thumbnails and previews are deliberately not read or included.",
                ]
            )
            + "<p>Share the input data and any results separately when needed. Intermediate results are not included either.</p>",
        )
    )
    privacy = data.get("privacy") or {}
    review = "<p>Folder locations are hidden. Filenames, scientific settings, labels and author-supplied notes can still identify a project: review the package before sharing.</p>"
    if isinstance(privacy, Mapping):
        if privacy.get("anonymise_filenames") is True:
            review += "<p>Filenames have been replaced with anonymous names.</p>"
        elif privacy.get("anonymise_filenames") is False:
            review += "<p>Original filenames are included.</p>"
    review += _details("Changes made for sharing", data.get("changes"))
    sections.append(_section("Sharing review", review))
    sections.append(
        _section("Optional command-line use", f"<p>{_advanced_runner(data)}</p>")
    )
    sections.append(input_evidence)
    env = data.get("environment") or {}
    guidance = installation_guidance(env, recorded=recorded)
    sections.append(
        _section(
            "Software and execution evidence",
            (
                "<p>Run-time records and export-time software are separate. Version lists describe an environment; they are not an installation lockfile.</p>"
                + f"<p>{_text(guidance['note'])}</p>"
                + f'<p>Other installation options: <a href="{_text(INSTALLATION_GUIDE_URL)}">VIPP installation guide</a>.</p>'
                + _details("Recorded run environment", env.get("run"))
                + (
                    "<p>No run-time environment was recorded.</p>"
                    if not env.get("run")
                    else ""
                )
                + _details("Environment used to export this package", env.get("export"))
            ),
        )
    )
    sections.append(
        _section(
            "Limitations",
            _list(
                data.get("limitations")
                or [
                    "The package alone cannot reproduce an analysis without its source data.",
                ]
            ),
        )
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>{_text(title)} — VIPP analysis record</title>
<style>
body{{margin:0;background:#eef3f6;color:#203344;font:16px/1.55 'Segoe UI',system-ui,sans-serif}}
main{{max-width:1080px;margin:32px auto;padding:0 24px 40px}}
.hero{{background:#153b4c;color:white;border-top:6px solid #1cc4ca;border-radius:10px}}
.hero td{{color:white;padding:30px 34px;border:0}}
h1{{font-size:30px;line-height:1.2;margin:14px 0}} h2{{font-size:21px;margin:0 0 14px}}h3{{font-size:17px;margin:0}}
.eyebrow{{letter-spacing:.12em;font-size:13px;font-weight:700;color:#80e1e5}}.badge{{font-size:14px;color:#c0eef1}}
section{{background:white;margin-top:18px;padding:26px 30px;border:1px solid #dce5eb;border-radius:8px}}
p{{margin:8px 0 14px}}li{{margin:8px 0}}.muted{{color:#586d7c;font-size:14px}}
.table-scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:12px;text-align:left;vertical-align:top;border-bottom:1px solid #e3e9ee;overflow-wrap:anywhere}}th{{background:#edf5f7;color:#244d60}}
.metrics td{{background:#f0f8f9;border:4px solid white;min-width:68px;font-size:13px}}.metric strong{{font-size:26px;color:#0c6475}}
.callout,.note{{background:#f0f7fa;border-left:4px solid #28b7c8;padding:14px 18px}}
details{{margin:8px 0}}summary{{cursor:pointer;color:#12647d;font-weight:600}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.5 Consolas,monospace;background:#f4f7f9;padding:12px}}.preserve{{white-space:pre-wrap}}
footer{{padding:24px 0;color:#586d7c;font-size:13px}}
@media(max-width:650px){{main{{padding:0 12px}}.hero td,section{{padding:20px}}h1{{font-size:25px}}}}
@media print{{body{{background:white}}main{{margin:0;padding:0}}section{{break-inside:avoid}}}}
</style></head><body><main>
<table class="hero" width="100%" cellspacing="0" cellpadding="24"><tr><td bgcolor="#153b4c" style="color:white">
<div class="eyebrow">VIPP · ANALYSIS RECORD</div><h1>{_text(title)}</h1>
<div class="badge">{kind} · Exported {_timestamp(data.get("created_at"))}</div></td></tr></table>
{"".join(sections)}
<footer>Generated locally by VIPP. No data are uploaded by this export.</footer>
</main></body></html>"""


def render_reproducibility_readme(report_data: dict) -> str:
    """Portable, deliberately short first-read instructions without shell paths."""
    recorded = report_data.get("package_kind") == "recorded_batch_run"
    kind = (
        "an archived batch run" if recorded else "a workflow recipe (not a run record)"
    )
    guidance = installation_guidance(
        report_data.get("environment") or {}, recorded=recorded
    )
    steps = "\n".join(
        f"{index}. {step}"
        for index, step in enumerate(_repeat_steps(report_data, markdown=True), start=1)
    )
    return (
        "# VIPP reproducibility package\n\n"
        f"This package contains {kind}. Open **report.html** for the readable "
        "summary; **report.json** contains its complete structured record.\n\n"
        "## How to repeat the analysis\n\n"
        f"{steps}\n\n"
        f"{_repeat_note(report_data)}\n\n"
        "## Optional command-line use\n\n"
        f"{_advanced_runner(report_data, markdown=True)}\n\n"
        "## Technical records\n\n"
        "report.html includes input identities, software versions and evidence "
        "limitations; report.json and environment.json retain the structured details.\n\n"
        f"{_markdown_text(guidance['note'])}\n\n"
        f"Other installation options: [VIPP installation guide]({INSTALLATION_GUIDE_URL}).\n\n"
        "## Before sharing\n\n"
        "Review filenames, scientific labels, parameters and author notes. Directory "
        "locations are hidden, but scientific context may still identify a project. "
        "Nothing is uploaded automatically.\n"
    )
