"""Capture bounded, offline native evidence for the assistant settings dialog.

Each theme runs in a fresh process, with a detached, read-only synthetic source.
Provider discovery, sign-in, context capture, planning, Add and Run are forbidden.
This is local presentation evidence, not authentication or scientific validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--theme", choices=("dark", "light"))
    focus = parser.add_mutually_exclusive_group()
    focus.add_argument(
        "--background-only",
        action="store_true",
        help="Capture only one Advanced card-surface fixture per fresh theme.",
    )
    focus.add_argument(
        "--repair-only",
        action="store_true",
        help="Capture only repair availability, activity and narrow action fixtures.",
    )
    parser.add_argument(
        "--scene",
        action="append",
        help="Run only this named scene from the chosen fixture group; repeatable.",
    )
    return parser


def _fresh_theme(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    state = output / "state"
    state.mkdir(exist_ok=True)
    os.environ["NAPARI_VIPP_PIPELINE_TIMING_HISTORY_PATH"] = str(state / "timings.json")
    os.environ["VIPP_DISABLE_UPDATE_CHECKS"] = "1"

    import numpy as np
    from napari._qt.qt_resources import get_stylesheet
    from napari.resources._icons import _theme_path
    from qtpy.QtCore import QCoreApplication, QDir, QEvent, QPoint, QSettings
    from qtpy.QtGui import QFont, QFontInfo
    from qtpy.QtTest import QTest
    from qtpy.QtWidgets import QApplication, QPushButton, QWidget

    from napari_vipp.core import workflow_proposal_codex, workflow_proposal_model
    from napari_vipp.core.metadata import AxisMetadata, image_state_from_array
    from napari_vipp.core.pipeline import PrototypePipeline
    from napari_vipp.core.snapshots import GraphSnapshot
    from napari_vipp.ui import workflow_assistant

    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(state))
    app = QApplication([])
    app.setQuitOnLastWindowClosed(False)
    QDir.addSearchPath(f"theme_{args.theme}", str(_theme_path(args.theme)))
    app.setStyleSheet(get_stylesheet(args.theme))
    counts = {
        name: 0
        for name in (
            "connection_discovery",
            "auth",
            "repair",
            "model_discovery",
            "planning",
            "context_capture",
            "assert_current",
            "add",
            "run",
        )
    }

    def forbidden(name):
        def call(*_args, **_kwargs):
            counts[name] += 1
            raise AssertionError(f"Offline settings cannot call {name}")

        return call

    workflow_proposal_codex.discover_codex_runtime = forbidden("connection_discovery")
    workflow_proposal_codex.start_codex_login = forbidden("auth")
    workflow_proposal_codex.repair_codex_profile = forbidden("repair")
    workflow_proposal_codex.generate_codex_proposal = forbidden("planning")
    workflow_proposal_model.discover_models = forbidden("model_discovery")
    workflow_proposal_model.generate_proposal = forbidden("planning")
    workflow_assistant.WorkflowAssistantDialog._refresh_codex_connection = (
        lambda *_args: None
    )
    source = np.arange(24 * 32, dtype=np.float32).reshape(24, 32).copy()
    source.flags.writeable = False
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    pipeline.outputs["input"] = source
    pipeline.output_states["input"] = image_state_from_array(
        source,
        axes=(AxisMetadata("y", "space", "um"), AxisMetadata("x", "space", "um")),
    )
    snapshot = GraphSnapshot.from_pipeline(pipeline)
    source_before = hashlib.sha256(source.tobytes()).hexdigest()
    source_paths = (
        Path(workflow_assistant.__file__).resolve(),
        Path(workflow_assistant.__file__)
        .resolve()
        .with_name("workflow_assistant_style.py"),
    )
    fingerprints = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_paths
    }
    parent = QWidget()
    parent.ensurePolished()
    assistant = workflow_assistant.WorkflowAssistantDialog(
        parent,
        sources=lambda: [("input", "Detached synthetic image")],
        capture=forbidden("context_capture"),
        assert_current=forbidden("assert_current"),
        apply=forbidden("add"),
        run=forbidden("run"),
    )
    assistant._theme_probe.ensurePolished()
    assistant._refresh_appearance()
    base_font = QFont(assistant.font())
    receipt = {
        "theme": args.theme,
        "platform": app.platformName(),
        "scope": (
            "Offline native settings only; no auth/provider/scientific qualification"
        ),
        "fresh_process_per_theme": True,
        "background_only": args.background_only,
        "repair_only": args.repair_only,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256_before": source_before,
        "ui_sources_sha256_before": fingerprints,
        "counts": counts,
        "states": [],
        "errors": [],
        "limitations": [
            "Does not exercise or resolve the separately recorded "
            "warm-process global-theme-switch limitation.",
            "Large-font scenes use a scoped settings-descendant 16pt stylesheet; "
            "not an OS-wide font or accessibility-setting qualification.",
        ],
    }

    def flush():
        app.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        app.processEvents()

    def rectangle(widget, ancestor):
        rect = widget.rect()
        rect.moveTopLeft(widget.mapTo(ancestor, QPoint(0, 0)))
        return rect

    def rect_values(rect):
        return [rect.x(), rect.y(), rect.width(), rect.height()]

    def passive():
        assert all(value == 0 for value in counts.values()), counts
        assert GraphSnapshot.from_pipeline(pipeline) == snapshot
        assert hashlib.sha256(source.tobytes()).hexdigest() == source_before
        assert not source.flags.writeable
        assert assistant._context is assistant._owner is assistant._prepared is None
        assert assistant.provider_combo.currentData() == "codex"

    def background_samples(settings):
        """Compare rendered blank container areas with their raised card gutter."""
        card = settings.advanced_panel.parentWidget()
        details_layout = settings.connection_details.layout()
        details_heading = details_layout.itemAt(0).widget().geometry()
        details_text = settings.connection_label.geometry()
        options_first = settings.codex_model_edit.geometry()
        options_second = settings.reasoning_combo.geometry()
        probes = (
            (
                settings.advanced_panel,
                QPoint(settings.advanced_panel.width() // 2, 1),
                True,
            ),
            (
                settings.codex_options,
                QPoint(
                    settings.codex_options.width() // 2,
                    (options_first.bottom() + options_second.top()) // 2,
                ),
                True,
            ),
            (
                settings.connection_details,
                QPoint(
                    settings.connection_details.width() // 2,
                    (details_heading.bottom() + details_text.top()) // 2,
                ),
                True,
            ),
            (
                settings.codex_model_edit,
                QPoint(
                    settings.codex_model_edit.width() - 8,
                    settings.codex_model_edit.height() // 2,
                ),
                False,
            ),
            (settings.timeout_spin, QPoint(6, 6), False),
        )
        samples = []
        for container, local_point, continuous in probes:
            body_point = container.mapTo(settings.body, local_point)
            settings.scroll.ensureVisible(body_point.x(), body_point.y(), 0, 24)
            flush()
            point = container.mapTo(settings, local_point)
            gutter = card.mapTo(settings, QPoint(6, 0))
            gutter.setY(point.y())
            viewport = rectangle(settings.scroll.viewport(), settings)
            assert viewport.contains(point) and viewport.contains(gutter), (
                f"Background probe outside visible viewport: {container.objectName()}"
            )
            rendered = settings.grab()
            scale = rendered.devicePixelRatioF()
            image = rendered.toImage()

            def colour(location, rendered_image=image, device_scale=scale):
                return rendered_image.pixelColor(
                    round(location.x() * device_scale),
                    round(location.y() * device_scale),
                ).name()

            sample = {
                "container": container.objectName() or type(container).__name__,
                "point": [point.x(), point.y()],
                "gutter_point": [gutter.x(), gutter.y()],
                "surface": colour(point),
                "card_gutter": colour(gutter),
                "expected_continuous": continuous,
            }
            assert (sample["surface"] == sample["card_gutter"]) is continuous, sample
            samples.append(sample)
        settings.scroll.verticalScrollBar().setValue(0)
        flush()
        return samples

    scenes = (
        [
            ("signed-out", "codex", "signed-out", False, 600, False),
            ("connected", "codex", "connected", False, 600, False),
            ("profile-blocked", "codex", "profile", False, 600, False),
            ("advanced", "codex", "profile", True, 600, False),
            ("local", "local", "signed-out", False, 600, False),
            ("example", "example", "signed-out", False, 600, False),
            ("signing-in", "codex", "signin", False, 600, False),
            ("checking", "codex", "status", False, 600, False),
            ("narrow-large-font", "codex", "signed-out", True, 420, True),
            ("local-narrow-large-font", "local", "signed-out", True, 420, True),
        ]
        if args.theme == "dark"
        else [
            ("signed-out", "codex", "signed-out", False, 600, False),
            ("profile-blocked", "codex", "profile", False, 600, False),
            ("advanced-large-font", "codex", "connected", True, 600, True),
        ]
    )
    if args.background_only:
        scenes = [("advanced-background", "codex", "connected", True, 600, False)]
    elif args.repair_only:
        scenes = [("repair-available", "codex", "repair-available", False, 600, False)]
        if args.theme == "dark":
            scenes.extend(
                [
                    ("repairing", "codex", "repair", False, 600, False),
                    (
                        "repair-narrow-large-font",
                        "codex",
                        "repair-available",
                        False,
                        420,
                        True,
                    ),
                ]
            )
    if args.scene:
        selected = set(args.scene)
        unknown = selected - {scene[0] for scene in scenes}
        if unknown:
            raise SystemExit(f"Unknown {args.theme} fixture scenes: {sorted(unknown)}")
        scenes = [scene for scene in scenes if scene[0] in selected]
    for name, provider, connection, advanced, width, large_font in scenes:
        settings = None
        scene = {"name": name, "requested_width": width, "large_font": large_font}
        try:
            font = QFont(base_font)
            if large_font:
                font.setPointSize(16)
            assistant.setFont(font)
            assistant._codex_authenticated = connection == "connected"
            assistant._codex_connection_issue = (
                "profile" if connection in {"profile", "repair-available"} else ""
            )
            assistant._codex_profile_repairable = connection == "repair-available"
            assistant._connection_busy = connection in {"signin", "status", "repair"}
            assistant._connection_kind = (
                connection if assistant._connection_busy else ""
            )
            tone = "error" if connection in {"profile", "repair-available"} else "info"
            text = (
                "The VIPP Codex profile was modified; no request was sent."
                if connection in {"profile", "repair-available"}
                else "Offline fixture status; no actual connection check was performed."
            )
            assistant._set_connection_status(tone, text)
            settings = workflow_assistant._AssistantSettingsDialog(assistant)
            settings.setWindowTitle(
                "Workflow assistant settings — OFFLINE visual fixture"
            )
            if large_font:
                settings.setStyleSheet(
                    settings.styleSheet()
                    + "\nQDialog#WorkflowAssistantSettings QWidget { font-size: 16pt; }"
                )
            assistant._settings_dialog = settings
            settings.provider_combo.setCurrentIndex(
                settings.provider_combo.findData(provider)
            )
            settings.advanced_toggle.setChecked(advanced)
            settings.resize(width, 460)
            settings.show()
            settings._refresh_connection_state()
            QTest.qWait(160)
            flush()
            settings._fit_content()
            QTest.qWait(80)
            flush()
            if large_font:
                assert settings.signin_button.font().pointSizeF() == 16
                effective_points = QFontInfo(settings.signin_button.font()).pointSizeF()
                assert abs(effective_points - 16) <= 0.75, effective_points
                assert settings.signin_button.fontMetrics().height() >= 20
            passive()
            scene["horizontal_scroll_maximum"] = (
                settings.scroll.horizontalScrollBar().maximum()
            )
            assert scene["horizontal_scroll_maximum"] == 0, (
                "Settings content extends beyond the viewport: "
                f"horizontal scroll maximum {scene['horizontal_scroll_maximum']}"
            )
            footer = rectangle(settings._footer, settings)
            assert settings.rect().contains(footer), (
                f"Footer outside dialog: {rect_values(footer)}"
            )
            button_rows = []
            for button in settings.findChildren(QPushButton):
                if not button.isVisibleTo(settings):
                    continue
                rect = rectangle(button, settings)
                hint = button.sizeHint()
                row = {
                    "text": button.text(),
                    "rectangle": rect_values(rect),
                    "size_hint": [hint.width(), hint.height()],
                    "enabled": button.isEnabled(),
                }
                button_rows.append(row)
                assert button.width() >= hint.width(), f"Button text clipped: {row}"
                assert button.height() >= hint.height(), f"Button height clipped: {row}"
                assert 0 <= rect.x() and rect.right() < settings.width(), (
                    f"Button outside width: {row}"
                )
            for button in (settings.accept_button, settings.cancel_button):
                assert settings.rect().contains(rectangle(button, settings)), (
                    "Footer action clipped"
                )
            if connection in {"profile", "repair-available"}:
                assert not settings.signin_button.isEnabled()
                assert settings.signin_button.isHidden()
                assert settings.connection_details_button.isVisibleTo(settings)
                assert settings.connection_repair_button.isVisibleTo(settings)
                assert settings.connection_repair_button.isEnabled() is (
                    connection == "repair-available"
                )
                assert (
                    "Choose Repair connection setup"
                    if connection == "repair-available"
                    else "cannot safely repair"
                ) in settings.connection_hint.text()
            if connection in {"signin", "status", "repair"}:
                spinner = {
                    "signin": settings.signin_button,
                    "status": settings.connection_refresh_button,
                    "repair": settings.connection_repair_button,
                }[connection]
                start_angle = spinner._spinner_angle
                QTest.qWait(100)
                assert (
                    spinner._spinner_timer.isActive()
                    and spinner._spinner_angle != start_angle
                )
                assert (
                    not settings.signin_button.isEnabled()
                    and not settings.accept_button.isEnabled()
                )
            last_bottom = max(
                item.widget().geometry().bottom()
                for index in range(settings.body.layout().count())
                if (item := settings.body.layout().itemAt(index)).widget() is not None
                and item.widget().isVisibleTo(settings)
            )
            scroll_maximum = settings.scroll.verticalScrollBar().maximum()
            gap = settings.scroll.viewport().height() - last_bottom
            if not scroll_maximum:
                assert gap <= 48, f"Unnecessary blank content gap: {gap}"
            scene.update(
                {
                    "logical_size": [settings.width(), settings.height()],
                    "footer": rect_values(footer),
                    "buttons": button_rows,
                    "vertical_scroll_maximum": scroll_maximum,
                    "blank_content_gap": gap,
                    "connection_title": settings.connection_title.text(),
                    "connection_tone": settings.codex_group.property("connectionTone"),
                    "font_point_size": settings.font().pointSizeF(),
                    "effective_button_font_point_size": QFontInfo(
                        settings.signin_button.font()
                    ).pointSizeF(),
                    "effective_button_font_height": (
                        settings.signin_button.fontMetrics().height()
                    ),
                    "advanced_visible": settings.advanced_panel.isVisibleTo(settings),
                }
            )
            if advanced:
                fields = [settings.attempts_spin, settings.timeout_spin]
                if provider == "codex":
                    fields = [
                        settings.codex_model_edit,
                        settings.reasoning_combo,
                        *fields,
                    ]
                reachability = []
                for field in fields:
                    settings.scroll.ensureWidgetVisible(field)
                    flush()
                    # Qt may prioritize a spin box's input-method cursor instead
                    # of the full control. Prove full-field reachability through
                    # the ordinary scrollbar as a user would, not cursor focus.
                    field_rect = rectangle(field, settings.scroll.viewport())
                    viewport = settings.scroll.viewport().rect()
                    displacement = max(0, field_rect.bottom() - viewport.bottom())
                    displacement -= max(0, viewport.top() - field_rect.top())
                    scrollbar = settings.scroll.verticalScrollBar()
                    scrollbar.setValue(scrollbar.value() + displacement)
                    flush()
                    field_rect = rectangle(field, settings.scroll.viewport())
                    assert settings.scroll.viewport().rect().contains(field_rect), (
                        f"Advanced field unreachable: {field_rect}"
                    )
                    reachability.append(rect_values(field_rect))
                scene["advanced_field_reachability"] = reachability
                settings.scroll.verticalScrollBar().setValue(0)
                flush()
                if provider == "codex":
                    scene["background_samples"] = background_samples(settings)
            screenshot = output / f"{args.theme}-{name}.png"
            assert settings.grab().save(str(screenshot), "PNG")
            scene["screenshot"] = str(screenshot)
            scene["screenshot_sha256"] = hashlib.sha256(
                screenshot.read_bytes()
            ).hexdigest()
            scene["passed"] = True
        except Exception:
            scene["passed"] = False
            scene["error"] = traceback.format_exc()
            receipt["errors"].append({"scene": name, "error": scene["error"]})
            if settings is not None:
                screenshot = output / f"{args.theme}-{name}-failure.png"
                if settings.grab().save(str(screenshot), "PNG"):
                    scene["screenshot"] = str(screenshot)
        finally:
            receipt["states"].append(scene)
            if settings is not None:
                settings.reject()
                assistant._settings_dialog = None
                settings.deleteLater()
                flush()
    assistant.close()
    parent.close()
    flush()
    receipt["source_sha256_after"] = hashlib.sha256(source.tobytes()).hexdigest()
    receipt["source_unchanged"] = receipt["source_sha256_after"] == source_before
    receipt["source_readonly"] = not source.flags.writeable
    receipt["graph_unchanged"] = GraphSnapshot.from_pipeline(pipeline) == snapshot
    receipt["ui_sources_sha256_after"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_paths
    }
    receipt["ui_sources_unchanged"] = receipt["ui_sources_sha256_after"] == fingerprints
    receipt["passed"] = (
        not receipt["errors"]
        and all(value == 0 for value in counts.values())
        and receipt["source_unchanged"]
        and receipt["source_readonly"]
        and receipt["graph_unchanged"]
        and receipt["ui_sources_unchanged"]
    )
    (output / "native-settings-receipt.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "theme": args.theme,
                "passed": receipt["passed"],
                "states": len(receipt["states"]),
                "errors": receipt["errors"],
                "counts": counts,
            }
        )
    )
    return 0 if receipt["passed"] else 1


def main():
    args = _parser().parse_args()
    if args.theme:
        return _fresh_theme(args)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    runs = []
    for theme in ("dark", "light"):
        target = output / theme
        if target.exists():
            raise SystemExit(f"Refusing to overwrite prior evidence: {target}")
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--output",
                str(target),
                "--theme",
                theme,
                *(["--background-only"] if args.background_only else []),
                *(["--repair-only"] if args.repair_only else []),
                *[
                    argument
                    for scene in args.scene or []
                    for argument in ("--scene", scene)
                ],
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            env=os.environ.copy(),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        runs.append(
            {
                "theme": theme,
                "returncode": result.returncode,
                "stdout": result.stdout[-12000:],
                "stderr": result.stderr[-8000:],
            }
        )
        print(result.stdout[-12000:])
        if result.returncode:
            print(result.stderr[-8000:], file=sys.stderr)
    reports = [
        json.loads(
            (output / theme / "native-settings-receipt.json").read_text(
                encoding="utf-8"
            )
        )
        for theme in ("dark", "light")
        if (output / theme / "native-settings-receipt.json").exists()
    ]
    receipt = {
        "runs": runs,
        "states": sum(len(report["states"]) for report in reports),
        "passed": len(reports) == 2 and all(report["passed"] for report in reports),
    }
    (output / "native-settings-summary.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
