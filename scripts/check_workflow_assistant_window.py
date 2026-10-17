"""Render the native staged assistant offline, without a viewer or inference.

Uses one detached synthetic cached source, mock connection status and inert
Add/Run spies. No VippWidget, calculation, credentials or live session is used.
The output is local UI evidence, not scientific or provider qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--responsive", action="store_true")
    parser.add_argument("--initiative-label", action="store_true")
    parser.add_argument("--return-navigation", action="store_true")
    parser.add_argument("--plan-review", action="store_true")
    parser.add_argument("--plan-recovery", action="store_true")
    parser.add_argument("--json-details", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    state = output / "state"
    state.mkdir(exist_ok=True)
    os.environ["NAPARI_VIPP_PIPELINE_TIMING_HISTORY_PATH"] = str(state / "timings.json")
    os.environ["VIPP_DISABLE_UPDATE_CHECKS"] = "1"

    import numpy as np
    from napari._qt.qt_resources import get_stylesheet
    from napari.resources._icons import _theme_path
    from qtpy.QtCore import QCoreApplication, QDir, QEvent, QSettings, Qt, QTimer
    from qtpy.QtGui import (
        QFontInfo,
        QPalette,
        QTextCursor,
        QTextDocument,
        QTextFrameFormat,
        QTextLength,
        QTextTable,
    )
    from qtpy.QtTest import QSignalSpy, QTest
    from qtpy.QtWidgets import (
        QAbstractScrollArea,
        QApplication,
        QCheckBox,
        QLabel,
        QPlainTextEdit,
        QPushButton,
        QStyle,
        QStyleOptionSlider,
        QWidget,
    )

    from napari_vipp.core import (
        workflow_proposal_codex,
        workflow_proposal_model,
        workflow_proposals,
    )
    from napari_vipp.core.metadata import AxisMetadata, image_state_from_array
    from napari_vipp.core.pipeline import PrototypePipeline
    from napari_vipp.core.workflow_proposals import (
        ProposalDialogue,
        capture_proposal_context,
        reference_proposal,
    )
    from napari_vipp.ui import workflow_assistant, workflow_assistant_details

    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(state))
    app = QApplication.instance() or QApplication([])
    app.setQuitOnLastWindowClosed(False)
    counts = {
        "mock_connection_checks": 0,
        "auth_calls": 0,
        "model_calls": 0,
        "model_discovery_calls": 0,
        "add_spy_calls": 0,
        "run_spy_calls": 0,
        "context_captures": 0,
        "spinner_animation_checks": 0,
        "question_reachability_checks": 0,
        "disclosure_retention_checks": 0,
        "initiative_choice_checks": 0,
        "initiative_retention_checks": 0,
        "goal_reachability_checks": 0,
        "blank_revision_stub_calls": 0,
        "deferred_preview_checks": 0,
        "question_answer_response_checks": 0,
        "chat_bubble_geometry_checks": 0,
        "multi_turn_chat_checks": 0,
        "assistant_outline_checks": 0,
        "quoted_question_outline_checks": 0,
        "render_authority_checks": 0,
        "persistent_action_checks": 0,
        "responsive_input_reachability_checks": 0,
        "responsive_authority_checks": 0,
        "recovery_stub_calls": 0,
    }

    def forbidden(name):
        def call(*_args, **_kwargs):
            counts[name] += 1
            raise AssertionError(f"Offline UI evidence cannot call {name}")

        return call

    def ready():
        counts["mock_connection_checks"] += 1
        return SimpleNamespace(
            supported=True, authenticated=True, detail="Offline mock connection"
        )

    workflow_proposal_codex.discover_codex_runtime = ready
    workflow_proposal_codex.start_codex_login = forbidden("auth_calls")
    workflow_proposal_codex.generate_codex_proposal = forbidden("model_calls")
    workflow_proposal_model.generate_proposal = forbidden("model_calls")
    workflow_proposal_model.discover_models = forbidden("model_discovery_calls")

    source = np.linspace(0, 1, 24 * 32, dtype=np.float32).reshape(24, 32).copy()
    source.flags.writeable = False
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    pipeline.outputs["input"] = source
    pipeline.output_states["input"] = image_state_from_array(
        source,
        axes=(AxisMetadata("y", "space", "um"), AxisMetadata("x", "space", "um")),
    )
    owner = object()

    def capture(source_id, *, include_images=False):
        counts["context_captures"] += 1
        assert source_id == "input" and include_images is False
        return capture_proposal_context(pipeline, source_id), owner

    def assert_current(context, token):
        assert token is owner
        context.assert_current(pipeline)

    def apply(prepared, token):
        counts["add_spy_calls"] += 1
        assert token is owner
        return prepared.added_node_ids

    def run(_ids):
        counts["run_spy_calls"] += 1

    ui_path = Path(workflow_assistant.__file__).resolve()
    policy_path = Path(workflow_proposals.__file__).resolve()
    ui_sources = (
        ui_path,
        ui_path.with_name("workflow_assistant_style.py"),
        ui_path.with_name("toolbar_controls.py"),
        ui_path.with_name("workflow_assistant_details.py"),
    )

    def ui_fingerprints():
        return {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in ui_sources
        }

    receipt = {
        "platform": app.platformName(),
        "scope": "Offline native UI only; no scientific/provider qualification",
        "profile": (
            "json-details"
            if args.json_details
            else "plan-recovery"
            if args.plan_recovery
            else "plan-review"
            if args.plan_review
            else "return-navigation"
            if args.return_navigation
            else "initiative-label"
            if args.initiative_label
            else "responsive"
            if args.responsive
            else "presentation"
        ),
        "qualification_limitations": [
            "This bounded native capture does not exercise or resolve the separately "
            "recorded warm-process QApplication global-theme-switch limitation."
        ],
        "fixture_notes": [],
        "ui_source_sha256_before": hashlib.sha256(ui_path.read_bytes()).hexdigest(),
        "ui_sources_sha256_before": ui_fingerprints(),
        "planning_policy_sha256_before": hashlib.sha256(
            policy_path.read_bytes()
        ).hexdigest(),
        "source_sha256_before": hashlib.sha256(source.tobytes()).hexdigest(),
        "states": [],
        "counts": counts,
        "initiative_preview_checks": [],
        "initiative_retention_checks": [],
        "blank_revision_checks": [],
        "question_answer_checks": [],
        "multi_turn_chat_checks": [],
        "responsive_checks": [],
        "initiative_label_checks": [],
        "return_navigation_checks": [],
        "plan_review_checks": [],
        "plan_recovery_checks": [],
        "json_details_checks": [],
        "json_font_growth_checks": [],
    }
    errors = []

    def flush():
        app.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        app.processEvents()

    def assert_spinner_animation(button):
        angle = button._spinner_angle
        QTest.qWait(180)
        assert button._spinner_timer.isActive(), "Pending spinner timer stopped"
        assert button._spinner_angle != angle, (
            f"Spinner did not advance from {angle}; visible={button.isVisible()}"
        )
        counts["spinner_animation_checks"] += 1

    def render_authority_snapshot(dialog):
        return (
            id(dialog._context),
            id(dialog._prepared),
            id(dialog._owner),
            dialog._request_objective,
            dialog.request_edit.toPlainText(),
            dialog.json_edit.toPlainText(),
            dialog._generation,
            dialog._draft_authorized,
            dialog._current_draft_index,
            dialog.draft_combo.currentIndex(),
            dialog.apply_button.isEnabled(),
            dialog.run_button.isEnabled(),
            dialog.consent_checkbox.isChecked(),
            tuple(dialog._added_ids),
            counts["context_captures"],
            counts["auth_calls"],
            counts["model_calls"],
            counts["add_spy_calls"],
            counts["run_spy_calls"],
        )

    def chat_bubble_geometry(dialog):
        document = dialog.transcript_edit.document()
        layout = document.documentLayout()
        root = document.rootFrame()
        root_rect = layout.frameBoundingRect(root)
        root_format = root.frameFormat()
        content_left = root_rect.left() + root_format.leftMargin()
        content_right = root_rect.right() - root_format.rightMargin()
        content_width = content_right - content_left
        assistant_surface, assistant_border, quote_surface, quote_border = (
            workflow_assistant.assistant_chat_surfaces(dialog._theme_probe.palette())
        )
        result = []
        previous_rectangle = None
        for frame in root.childFrames():
            if not isinstance(frame, QTextTable):
                continue
            cursor = frame.firstCursorPosition()
            cursor.setPosition(
                frame.lastCursorPosition().position(), QTextCursor.KeepAnchor
            )
            text = cursor.selectedText().replace("\u2029", "\n")
            speaker = text.lstrip().split("\n", 1)[0]
            user = speaker.startswith("You")
            rectangle = layout.frameBoundingRect(frame)
            table_format = frame.format()
            expected_percent = 74 if user else 82
            # Qt reports table content bounds inset by cell padding. Normalize
            # that inset, as the focused Qt geometry tests do, before comparing
            # the bubbles' alignment edges with the document viewport.
            inset = (
                document.documentMargin()
                + table_format.cellPadding()
                + table_format.border()
            )
            raw_left_gutter = rectangle.left() - root_rect.left()
            raw_right_gutter = root_rect.right() - rectangle.right()
            left_gutter = raw_left_gutter - inset
            right_gutter = raw_right_gutter + inset
            geometry = {
                "speaker": speaker,
                "role": "user" if user else "assistant",
                "width_percent": table_format.width().rawValue(),
                "percentage_width": table_format.width().type()
                == QTextLength.PercentageLength,
                "alignment": int(table_format.alignment()),
                "horizontal_alignment": int(
                    table_format.alignment()
                    & (Qt.AlignLeft | Qt.AlignRight | Qt.AlignHCenter | Qt.AlignJustify)
                ),
                "frame_rect": [
                    rectangle.x(),
                    rectangle.y(),
                    rectangle.width(),
                    rectangle.height(),
                ],
                "content_width": content_width,
                "root_rect": [
                    root_rect.x(),
                    root_rect.y(),
                    root_rect.width(),
                    root_rect.height(),
                ],
                "root_margins": [root_format.leftMargin(), root_format.rightMargin()],
                "left_gutter": left_gutter,
                "right_gutter": right_gutter,
                "raw_document_left_gutter": raw_left_gutter,
                "raw_document_right_gutter": raw_right_gutter,
                "qt_content_inset": inset,
                "border_width": table_format.border(),
                "border_style": int(
                    getattr(
                        table_format.borderStyle(), "value", table_format.borderStyle()
                    )
                ),
                "border_color": table_format.borderBrush().color().name(),
                "cell_padding": table_format.cellPadding(),
                "background": frame.cellAt(0, 0).format().background().color().name(),
                "text_chars": len(text),
                "nested_frames": len(frame.childFrames()),
                "quoted_questions": [],
            }
            if user:
                assert table_format.border() == 0
            else:
                assert table_format.border() == 1
                assert table_format.borderStyle() == QTextFrameFormat.BorderStyle_Solid
                assert table_format.borderBrush().color().name() == (
                    assistant_border.name()
                )
                assert geometry["background"] == assistant_surface.name()
                counts["assistant_outline_checks"] += 1
            for quote in frame.childFrames():
                if not isinstance(quote, QTextTable):
                    continue
                quote_format = quote.format()
                quote_geometry = {
                    "border_width": quote_format.border(),
                    "border_style": int(
                        getattr(
                            quote_format.borderStyle(),
                            "value",
                            quote_format.borderStyle(),
                        )
                    ),
                    "border_color": quote_format.borderBrush().color().name(),
                    "background": (
                        quote.cellAt(0, 0).format().background().color().name()
                    ),
                    "cell_padding": quote_format.cellPadding(),
                    "width_percent": quote_format.width().rawValue(),
                }
                geometry["quoted_questions"].append(quote_geometry)
                assert quote_format.border() == 1
                assert quote_format.borderStyle() == QTextFrameFormat.BorderStyle_Solid
                assert quote_geometry["border_color"] == quote_border.name()
                assert quote_geometry["background"] == quote_surface.name()
                assert quote_geometry["border_color"] != geometry["background"]
                assert quote_format.cellPadding() == 10
                assert quote_format.width().rawValue() == 100
                counts["quoted_question_outline_checks"] += 1
            result.append(geometry)
            assert geometry["percentage_width"]
            assert geometry["width_percent"] == expected_percent
            assert geometry["horizontal_alignment"] == int(
                Qt.AlignRight if user else Qt.AlignLeft
            )
            assert (
                abs(rectangle.width() - content_width * expected_percent / 100) <= 4
            ), f"Unexpected {geometry['role']} bubble width: {geometry}"
            assert left_gutter >= -2 and right_gutter >= -2, (
                f"Bubble extends outside the assumed document content: {geometry}"
            )
            assert (
                left_gutter > right_gutter + 5
                if user
                else right_gutter > left_gutter + 5
            ), f"Bubble alignment did not produce its expected gutter: {geometry}"
            if previous_rectangle is not None:
                assert rectangle.top() >= previous_rectangle.bottom(), (
                    f"Successive bubble frames overlap: {geometry}"
                )
            previous_rectangle = rectangle
            counts["chat_bubble_geometry_checks"] += 1
        return result

    def screenshot(widget, name, requested_size=None):
        flush()
        if (
            isinstance(widget, workflow_assistant.WorkflowAssistantDialog)
            and widget.stage_tabs.currentIndex() == 1
            and widget.transcript_edit.isVisibleTo(widget)
        ):
            for _attempt in range(20):
                QTest.qWait(10)
                flush()
                if not (
                    widget._appearance_timer.isActive()
                    or widget.transcript_edit._fit_timer.isActive()
                ):
                    break
            assert not widget._appearance_timer.isActive()
            assert not widget.transcript_edit._fit_timer.isActive()
        path = output / name
        pixels = widget.grab()
        assert pixels.save(str(path))
        screen_size = widget.screen().availableGeometry().size()
        frame_size = widget.frameGeometry().size()
        assistant_state = {}
        preview_state = {}
        discussion_state = {}
        button_spacing = []
        for button in widget.findChildren(QPushButton):
            if not button.isVisibleTo(widget):
                continue
            option_builder = getattr(button, "_toolbar_style_option", None)
            if option_builder is None:
                continue
            option = option_builder()
            spacer = button._ICON_TEXT_SPACER
            icon_present = not button.icon().isNull()
            button_spacing.append(
                {
                    "text": button.text(),
                    "painted_text": option.text,
                    "font_family": button.font().family(),
                    "font_point_size": button.font().pointSizeF(),
                    "font_height": option.fontMetrics.height(),
                    "direction": "rtl" if option.direction == Qt.RightToLeft else "ltr",
                    "icon_present": icon_present,
                    "extra_spacer_advance": option.fontMetrics.horizontalAdvance(spacer)
                    if icon_present
                    else 0,
                    "logical_size": [button.width(), button.height()],
                    "size_hint": [
                        button.sizeHint().width(),
                        button.sizeHint().height(),
                    ],
                }
            )
        if isinstance(widget, workflow_assistant.WorkflowAssistantDialog):
            send_buttons = {
                "goal": widget.prepare_button,
                "message": widget.send_button,
                "answers": widget.revise_button,
            }
            if hasattr(widget, "recovery_button"):
                send_buttons["recovery"] = widget.recovery_button
            assistant_state = {
                "stage_index": widget.stage_tabs.currentIndex(),
                "busy": widget._busy,
                "job_kind": widget._job_kind,
                "goal_visible": widget.goal_edit.isVisibleTo(widget),
                "prepare_text": widget.prepare_button.text(),
                "prepare_enabled": widget.prepare_button.isEnabled(),
                "prepare_icon_present": not widget.prepare_button.icon().isNull(),
                "header_icon_present": not widget.header_icon.pixmap().isNull(),
                "next_visible": widget.next_button.isVisibleTo(widget),
                "next_enabled": widget.next_button.isEnabled(),
                "next_text": widget.next_button.text(),
                "close_visible": widget.close_button.isVisibleTo(widget),
                "close_in_bottom_note": widget.bottom_frame.isAncestorOf(
                    widget.close_button
                ),
                "close_in_navigation": widget.footer_frame.isAncestorOf(
                    widget.close_button
                ),
                "sending_action": widget._sending_action,
                "initiative": {
                    "detent": widget.initiative_slider.value(),
                    "range": [
                        widget.initiative_slider.minimum(),
                        widget.initiative_slider.maximum(),
                    ],
                    "label": workflow_assistant._INITIATIVE_CHOICES[
                        widget.initiative_slider.value()
                    ][1],
                    "explanation": widget.initiative_explanation.text(),
                    "preferences": widget._planning_preferences().to_dict(),
                    "single_step": widget.initiative_slider.singleStep(),
                    "page_step": widget.initiative_slider.pageStep(),
                    "tick_interval": widget.initiative_slider.tickInterval(),
                    "accessible_name": widget.initiative_slider.accessibleName(),
                    "accessible_description": (
                        widget.initiative_slider.accessibleDescription()
                    ),
                    "card_labels": [
                        control.text()
                        for control in widget.initiative_card.findChildren(QLabel)
                    ],
                },
                "send_buttons": {
                    action: {
                        "text": button.text(),
                        "sending": button._sending,
                        "spinner_timer_active": button._spinner_timer.isActive(),
                        "spinner_angle": button._spinner_angle,
                        "icon_present": not button.icon().isNull(),
                        "size": [button.width(), button.height()],
                        "size_hint": [
                            button.sizeHint().width(),
                            button.sizeHint().height(),
                        ],
                        "font_height": button.fontMetrics().height(),
                        "height_fits_size_hint": button.height()
                        >= button.sizeHint().height(),
                    }
                    for action, button in send_buttons.items()
                },
                "request_progress_visible": widget.request_progress.isVisibleTo(widget),
                "request_progress_range": [
                    widget.request_progress.minimum(),
                    widget.request_progress.maximum(),
                ],
            }
            assert assistant_state["close_visible"]
            assert assistant_state["initiative"]["range"] == [0, 2]
            assert assistant_state["initiative"]["single_step"] == 1
            assert assistant_state["initiative"]["page_step"] == 1
            assert assistant_state["initiative"]["tick_interval"] == 1
            assert assistant_state["close_in_bottom_note"]
            assert not assistant_state["close_in_navigation"]
            if assistant_state["stage_index"] == 0:
                goal_viewport = widget.goal_scroll.viewport()
                slider = widget.initiative_slider
                card = widget.initiative_card
                slider_option = QStyleOptionSlider()
                slider.initStyleOption(slider_option)
                groove = slider.style().subControlRect(
                    QStyle.CC_Slider,
                    slider_option,
                    QStyle.SC_SliderGroove,
                    slider,
                )
                groove_position = slider.mapTo(card, groove.topLeft())
                endpoint_tops = []
                endpoint_bottoms = []
                for endpoint in (
                    widget.initiative_ask_label,
                    widget.initiative_best_label,
                ):
                    position = endpoint.mapTo(card, endpoint.rect().topLeft())
                    endpoint_tops.append(position.y())
                    endpoint_bottoms.append(position.y() + endpoint.height())
                explanation_position = widget.initiative_explanation.mapTo(
                    card, widget.initiative_explanation.rect().topLeft()
                )
                margins = card.layout().contentsMargins()
                controls = card.findChild(QWidget, "AssistantInitiativeControls")
                assistant_state["initiative"]["geometry"] = {
                    "card_size": [card.width(), card.height()],
                    "margins": [
                        margins.left(),
                        margins.top(),
                        margins.right(),
                        margins.bottom(),
                    ],
                    "layout_spacing": card.layout().spacing(),
                    "controls_spacing": controls.layout().spacing(),
                    "slider_height": slider.height(),
                    "slider_minimum_height": slider.minimumHeight(),
                    "track_rect_in_card": [
                        groove_position.x(),
                        groove_position.y(),
                        groove.width(),
                        groove.height(),
                    ],
                    "track_to_endpoints_gap": min(endpoint_tops)
                    - (groove_position.y() + groove.height()),
                    "endpoints_to_explanation_gap": explanation_position.y()
                    - max(endpoint_bottoms),
                    "explanation_height": widget.initiative_explanation.height(),
                }
                geometry = assistant_state["initiative"]["geometry"]
                assert geometry["margins"] == [14, 10, 14, 10]
                assert geometry["layout_spacing"] == 4
                assert geometry["controls_spacing"] == 0
                assert geometry["endpoints_to_explanation_gap"] >= 6

                def goal_control_visible(control):
                    if not goal_viewport.isAncestorOf(control):
                        return False
                    position = control.mapTo(goal_viewport, control.rect().topLeft())
                    return goal_viewport.rect().contains(
                        control.rect().translated(position)
                    )

                assistant_state["goal_scroll"] = {
                    "maximum": widget.goal_scroll.verticalScrollBar().maximum(),
                    "value": widget.goal_scroll.verticalScrollBar().value(),
                    "viewport_size": [goal_viewport.width(), goal_viewport.height()],
                    "goal_fully_in_viewport": goal_control_visible(widget.goal_edit),
                    "initiative_fully_in_viewport": goal_control_visible(
                        widget.initiative_card
                    ),
                    "send_goal_fully_in_viewport": goal_control_visible(
                        widget.prepare_button
                    ),
                }
            assert assistant_state["header_icon_present"]
            assert assistant_state["prepare_icon_present"]
            assert all(
                button.height() >= button.sizeHint().height()
                for button in send_buttons.values()
                if button.isVisibleTo(widget)
            )
            if assistant_state["stage_index"] == 0:
                assert assistant_state["goal_visible"]
                assert assistant_state["next_visible"] == bool(widget._drafts)
                if widget._drafts:
                    assert assistant_state["next_text"] == "Next: Discussion"
                    assert assistant_state["next_enabled"] == (not widget._busy)
                else:
                    assert not any(
                        button.isVisibleTo(widget) and "Next" in button.text()
                        for button in widget.findChildren(QPushButton)
                    )
            assert assistant_state["request_progress_visible"] == widget._busy
            assert assistant_state["request_progress_range"] == (
                [0, 0] if widget._busy else [0, 1]
            )
            if widget._busy and widget._job_kind == "proposal":
                action = widget._sending_action
                pending = send_buttons[action]
                if action == "recovery":
                    assert pending.text() == (
                        "Updating plan…"
                        if widget._recovery_kind == "update"
                        else "Retrying…"
                    )
                else:
                    assert pending.text() == f"Sending {action}…"
                assert pending._sending and not pending.isEnabled()
                assert pending._spinner_timer.isActive()
                assert all(
                    not button._sending
                    for other, button in send_buttons.items()
                    if other != action
                )
                if action != "recovery":
                    assert assistant_state["stage_index"] == (
                        0 if action == "goal" else 1
                    )
            else:
                assert all(not button._sending for button in send_buttons.values())
            if assistant_state["stage_index"] == 1:
                scroll = widget.discussion_scroll
                viewport = scroll.viewport()

                def in_viewport(control):
                    if not viewport.isAncestorOf(control):
                        return False
                    position = control.mapTo(viewport, control.rect().topLeft())
                    return viewport.rect().contains(control.rect().translated(position))

                disclosure_geometry = {}
                for key, frame, header in (
                    (
                        "original_goal",
                        widget.original_goal_frame,
                        widget.original_goal_toggle,
                    ),
                    (
                        "conversation",
                        widget.conversation_frame,
                        widget.conversation_toggle,
                    ),
                ):
                    margins = frame.layout().contentsMargins()
                    geometry = {
                        "visible": frame.isVisibleTo(widget),
                        "expanded": header.isChecked(),
                        "frame_height": frame.height(),
                        "header_height": header.height(),
                        "frame_overhead": frame.height() - header.height(),
                        "layout_margins": [
                            margins.left(),
                            margins.top(),
                            margins.right(),
                            margins.bottom(),
                        ],
                    }
                    disclosure_geometry[key] = geometry
                    if geometry["visible"] and not geometry["expanded"]:
                        assert geometry["frame_height"] <= (
                            geometry["header_height"] + 4
                        ), f"{name}: {key} closed frame is taller than its header + 4"

                discussion_state = {
                    "disclosure_geometry": disclosure_geometry,
                    "original_goal_expanded": widget.original_goal_toggle.isChecked(),
                    "original_goal_visible": widget.original_goal_label.isVisibleTo(
                        widget
                    ),
                    "original_goal_chars": len(widget.goal_edit.toPlainText()),
                    "original_goal_icon_present": not (
                        widget.original_goal_toggle.icon().isNull()
                    ),
                    "conversation_expanded": widget.conversation_toggle.isChecked(),
                    "conversation_visible": widget.transcript_edit.isVisibleTo(widget),
                    "conversation_icon_present": not (
                        widget.conversation_toggle.icon().isNull()
                    ),
                    "retained_chat_entries": len(widget._chat),
                    "retained_chat_sha256": hashlib.sha256(
                        json.dumps(widget._chat, ensure_ascii=False).encode("utf-8")
                    ).hexdigest(),
                    "outer_scroll_maximum": scroll.verticalScrollBar().maximum(),
                    "outer_scroll_value": scroll.verticalScrollBar().value(),
                    "outer_viewport_size": [viewport.width(), viewport.height()],
                    "transcript_height": widget.transcript_edit.height(),
                    "transcript_internal_scroll_maximum": (
                        widget.transcript_edit.verticalScrollBar().maximum()
                    ),
                    "question_count": len(widget._question_fields),
                    "question_rows": [
                        {
                            "question_chars": len(question),
                            "answer_chars": len(field.text()),
                            "answer_fully_in_viewport": in_viewport(field),
                            "answer_height": field.height(),
                            "answer_placeholder": field.placeholderText(),
                        }
                        for question, field in widget._question_fields
                    ],
                    "composer_height": widget.discussion_edit.height(),
                    "composer_visible_lines": (
                        widget.discussion_edit.viewport().height()
                        // widget.discussion_edit.fontMetrics().lineSpacing()
                    ),
                    "composer_fully_in_viewport": in_viewport(widget.discussion_edit),
                    "send_message_fully_in_viewport": in_viewport(widget.send_button),
                    "current_question_occurrences": [
                        {
                            "in_transcript": widget.transcript_edit.toPlainText().count(
                                question
                            ),
                            "in_question_cards": sum(
                                label.text() == question
                                for label in widget.questions_group.findChildren(QLabel)
                            ),
                            "retained_in_chat": sum(
                                text.count(question) for _speaker, text in widget._chat
                            ),
                        }
                        for question, _field in widget._question_fields
                    ],
                    "chat_bubbles": chat_bubble_geometry(widget)
                    if widget.transcript_edit.isVisibleTo(widget)
                    else [],
                }
                assert discussion_state["original_goal_icon_present"]
                assert discussion_state["conversation_icon_present"]
                assert (
                    discussion_state["conversation_visible"]
                    == (discussion_state["conversation_expanded"])
                )
                if widget.questions_group.isVisibleTo(widget):
                    assert all(
                        question["in_transcript"] == 0
                        and question["in_question_cards"] == 1
                        and question["retained_in_chat"] >= 1
                        for question in discussion_state["current_question_occurrences"]
                    )
        elif isinstance(widget, workflow_assistant.SharingPreviewDialog):
            panel = widget.scroll.widget()
            details = widget.findChild(QPlainTextEdit, "SharedRequestJSON")
            advanced = widget.findChild(QCheckBox, "SharedRequestAdvancedToggle")
            last_bottom = max(
                item.widget().geometry().bottom()
                for index in range(panel.layout().count())
                if (item := panel.layout().itemAt(index)).widget() is not None
                and item.widget().isVisibleTo(widget)
            )
            content_bottom = (
                panel.mapTo(widget, panel.rect().topLeft()).y() + last_bottom
            )
            close_top = widget.close_button.mapTo(
                widget, widget.close_button.rect().topLeft()
            ).y()
            preview_state = {
                "viewport_object_name": widget.scroll.viewport().objectName(),
                "scroll_object_name": widget.scroll.objectName(),
                "advanced_visible": advanced.isChecked(),
                "json_readonly": details.isReadOnly(),
                "vertical_scroll_maximum": widget.scroll.verticalScrollBar().maximum(),
                "gap_from_last_content_to_close": close_top - content_bottom,
                "close_visible": widget.close_button.isVisibleTo(widget),
                "planning_preferences": widget.findChild(
                    QLabel, "SharedPlanningPreferences"
                ).text(),
                "deferred_questions": [
                    label.text()
                    for label in widget.findChildren(QLabel, "SharedDeferredQuestion")
                ],
            }
            assert preview_state["json_readonly"] and preview_state["close_visible"]
            assert preview_state["viewport_object_name"] == "SharedRequestViewport"
            assert preview_state["scroll_object_name"] == "SharedRequestScroll"
            assert widget.height() <= 740
            if preview_state["vertical_scroll_maximum"] == 0:
                assert preview_state["gap_from_last_content_to_close"] <= 60
        # Scrolled content can intentionally extend beyond its viewport. Inspect
        # only outer controls and labels here; the screenshots show scroll areas.
        outer_issues = []
        for control in widget.findChildren(QWidget):
            if not isinstance(control, (QPushButton, QLabel)):
                continue
            if not control.isVisibleTo(widget):
                continue
            ancestor = control.parentWidget()
            scrolled = False
            while ancestor is not None and ancestor is not widget:
                container = ancestor.parentWidget()
                if isinstance(container, QAbstractScrollArea) and (
                    ancestor is container.viewport()
                ):
                    scrolled = True
                    break
                ancestor = ancestor.parentWidget()
            if scrolled:
                continue
            point = control.mapTo(widget, control.rect().topLeft())
            if (
                point.x() < -1
                or point.y() < -1
                or point.x() + control.width() > widget.width() + 1
                or point.y() + control.height() > widget.height() + 1
            ):
                outer_issues.append(control.text()[:80])
        receipt["states"].append(
            {
                "screenshot": name,
                "logical_size": [widget.width(), widget.height()],
                "pixel_size": [pixels.width(), pixels.height()],
                "device_pixel_ratio": widget.devicePixelRatioF(),
                "available_screen_size": [
                    screen_size.width(),
                    screen_size.height(),
                ],
                "native_frame_size": [frame_size.width(), frame_size.height()],
                "fits_available_screen": frame_size.width() <= screen_size.width()
                and frame_size.height() <= screen_size.height(),
                "minimum_size": [
                    widget.minimumWidth(),
                    widget.minimumHeight(),
                ],
                "minimum_size_hint": [
                    widget.minimumSizeHint().width(),
                    widget.minimumSizeHint().height(),
                ],
                "requested_size": requested_size,
                "requested_size_matched": requested_size is None
                or [widget.width(), widget.height()] == requested_size,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "outer_control_clipping": outer_issues,
                "assistant_state": assistant_state,
                "preview_state": preview_state,
                "discussion_state": discussion_state,
                "button_spacing": button_spacing,
            }
        )

    def record_json_details(theme, dialog):
        goal = (
            'Offline JSON fixture: α/µm, actual 🧪, escaped quote " and literal '
            r"Unicode sequence \u03b1. No request is sent."
        )
        dialog.goal_edit.setPlainText(goal)
        request = dialog._ensure_context(goal)
        proposal = reference_proposal(dialog._context)
        proposal["summary"] = (
            'Offline JSON display: actual 🧪; escaped 😀; "quotes", slash \\ and α/µm.'
        )
        proposal["assumptions"] = [
            r"Literal <tag> & Unicode \u03b1 is data, "
            "not markup or a scientific result."
        ]
        dialog._accept_proposal(proposal)
        mixed = dialog.json_edit.toPlainText().replace("😀", r"\ud83d\ude00")
        assert json.loads(mixed) == proposal
        dialog.json_edit.blockSignals(True)
        dialog.json_edit.setPlainText(mixed)
        dialog.json_edit.blockSignals(False)
        dialog.validate()
        assert dialog.json_edit.toPlainText() == mixed
        retained_history = tuple(dialog._drafts)
        retained_chat = tuple(dialog._chat)
        assert len(retained_history) == 2
        assert retained_history[-1].proposal_text == mixed
        assert not isinstance(
            dialog.preview_edit, workflow_assistant_details.AssistantDetailsEditor
        )
        assert not isinstance(
            dialog.goal_edit, workflow_assistant_details.AssistantDetailsEditor
        )
        assert dialog.trace_edit._json_highlighter is None

        def select_emoji(editor):
            value = editor.toPlainText()
            offset = len(value[: value.index("🧪")].encode("utf-16-le")) // 2
            cursor = editor.textCursor()
            cursor.setPosition(offset)
            cursor.setPosition(offset + 2, QTextCursor.KeepAnchor)
            editor.setTextCursor(cursor)
            assert editor.textCursor().selectedText() == "🧪"

        def details_state(editor, highlighted=True):
            assert isinstance(editor, workflow_assistant_details.AssistantDetailsEditor)
            assert QFontInfo(editor.font()).fixedPitch()
            assert editor.fontMetrics().horizontalAdvance("W") == (
                editor.fontMetrics().horizontalAdvance("i")
            )
            document = editor.document()
            cursor = editor.textCursor()
            before = (
                editor.toPlainText(),
                cursor.position(),
                cursor.anchor(),
                document.isModified(),
                document.availableUndoSteps(),
            )
            editor.refresh_highlighting()
            cursor = editor.textCursor()
            assert before == (
                editor.toPlainText(),
                cursor.position(),
                cursor.anchor(),
                document.isModified(),
                document.availableUndoSteps(),
            )
            formats = []
            block = document.begin()
            while block.isValid():
                text = block.text()
                offsets = {0}
                length = 0
                for character in text:
                    length += 2 if ord(character) > 0xFFFF else 1
                    offsets.add(length)
                for span in block.layout().formats():
                    assert span.start in offsets
                    assert span.start + span.length in offsets
                    formats.append(
                        {
                            "block": block.blockNumber(),
                            "start_utf16": span.start,
                            "length_utf16": span.length,
                            "color": span.format.foreground().color().name(),
                        }
                    )
                block = block.next()
            if highlighted:
                assert editor._json_highlighter is not None and formats
                background = editor.palette().color(QPalette.Base)
                contrasts = {
                    name: workflow_assistant_details._contrast(
                        format_.foreground().color(), background
                    )
                    for name, format_ in editor._json_highlighter.formats.items()
                }
                assert all(value >= 4.5 for value in contrasts.values())
                assert len({span["color"] for span in formats}) >= 3
            else:
                assert editor._json_highlighter is None and not formats
                contrasts = {}
            return {
                "font_family": QFontInfo(editor.font()).family(),
                "font_point_size": editor.font().pointSizeF(),
                "fixed_pitch": True,
                "native_equal_W_and_i_advance": (
                    editor.fontMetrics().horizontalAdvance("W")
                ),
                "read_only": editor.isReadOnly(),
                "text_sha256": hashlib.sha256(
                    editor.toPlainText().encode("utf-8")
                ).hexdigest(),
                "cursor_utf16": [cursor.position(), cursor.anchor()],
                "text_cursor_modified_undo_preserved": True,
                "utf16_ranges_valid": True,
                "format_ranges": len(formats),
                "format_sample": formats[:16],
                "token_contrasts": contrasts,
            }

        select_emoji(dialog.json_edit)
        if theme == "dark":
            technical_parent = dialog.json_edit.parentWidget()
            parent_stylesheet = technical_parent.styleSheet()
            editors = {
                "proposal": dialog.json_edit,
                "request": dialog.request_edit,
                "attempts": dialog.trace_edit,
            }

            def literal_state(editor):
                cursor = editor.textCursor()
                document = editor.document()
                return (
                    editor.toPlainText(),
                    cursor.position(),
                    cursor.anchor(),
                    document.isModified(),
                    document.availableUndoSteps(),
                )

            before = {name: literal_state(editor) for name, editor in editors.items()}
            sizes_before = {
                name: editor.font().pointSizeF() for name, editor in editors.items()
            }
            spies = {
                f"{name}.{signal}": QSignalSpy(source)
                for name, editor in editors.items()
                for signal, source in (
                    ("textChanged", editor.textChanged),
                    ("contentsChanged", editor.document().contentsChanged),
                    ("contentsChange", editor.document().contentsChange),
                )
            }
            authority = render_authority_snapshot(dialog)
            try:
                technical_parent.setStyleSheet("font-size: 16pt;")
                QTest.qWait(80)
                flush()
                assert technical_parent.font().pointSizeF() == 16
                sizes_grown = {}
                for name, editor in editors.items():
                    assert editor.parentWidget() is technical_parent
                    assert editor.font().pointSizeF() == 16
                    assert editor.font().pointSizeF() > sizes_before[name]
                    assert QFontInfo(editor.font()).fixedPitch()
                    assert editor.fontMetrics().horizontalAdvance("W") == (
                        editor.fontMetrics().horizontalAdvance("i")
                    )
                    assert literal_state(editor) == before[name]
                    sizes_grown[name] = editor.font().pointSizeF()
                assert render_authority_snapshot(dialog) == authority
                assert tuple(dialog._drafts) == retained_history
                assert tuple(dialog._chat) == retained_chat
            finally:
                technical_parent.setStyleSheet(parent_stylesheet)
                QTest.qWait(80)
                flush()
            for name, editor in editors.items():
                assert editor.font().pointSizeF() == sizes_before[name]
                assert literal_state(editor) == before[name]
            assert all(len(spy) == 0 for spy in spies.values())
            assert render_authority_snapshot(dialog) == authority
            receipt["json_font_growth_checks"].append(
                {
                    "theme": theme,
                    "scope": "Actual immediate technical-parent local 16pt stylesheet; "
                    "not a global dialog font-cascade qualification.",
                    "parent_class": type(technical_parent).__name__,
                    "sizes_before_pt": sizes_before,
                    "sizes_grown_pt": sizes_grown,
                    "normal_size_restored": True,
                    "fixed_family_and_equal_glyph_advances_preserved": True,
                    "text_cursor_modified_undo_history_chat_authority_preserved": True,
                    "edit_signals": {name: len(spy) for name, spy in spies.items()},
                }
            )
            receipt["fixture_notes"].append(
                "Original ui evidence predates the helper font-size inheritance fix. "
                "This final capture includes one passive immediate-parent growth "
                "check and restores normal sizes before the eight screenshots."
            )
        request_state = details_state(dialog.request_edit)
        attempts_state = details_state(dialog.trace_edit, highlighted=False)
        for width, height in ((760, 800), (1040, 820)):
            dialog.resize(width, height)
            dialog.advanced_checkbox.setChecked(True)
            dialog.tabs.setCurrentIndex(1)
            QTest.qWait(80)
            flush()
            point = dialog.json_edit.mapTo(
                dialog.review_scroll.widget(), dialog.json_edit.rect().topLeft()
            )
            dialog.review_scroll.verticalScrollBar().setValue(point.y() - 30)
            authority = render_authority_snapshot(dialog)
            state = details_state(dialog.json_edit)
            screenshot(dialog, f"{theme}-proposal-json-{width}.png", [width, height])
            assert receipt["states"][-1]["requested_size_matched"]
            assert render_authority_snapshot(dialog) == authority
            assert tuple(dialog._drafts) == retained_history
            assert tuple(dialog._chat) == retained_chat
            receipt["json_details_checks"].append(
                {
                    "theme": theme,
                    "scene": "proposal",
                    "width": width,
                    "editor": state,
                    "request_editor": request_state,
                    "attempts_editor": attempts_state,
                    "escaped_unicode_and_surrogate_pair_preserved": True,
                    "history_chat_authority_unchanged": True,
                }
            )
            counts["render_authority_checks"] += 1

            preview = dialog._sharing_preview_dialog(request)
            try:
                preview.resize(width, 740)
                preview.show()
                preview.findChild(QCheckBox, "SharedRequestAdvancedToggle").setChecked(
                    True
                )
                QTest.qWait(80)
                flush()
                editor = preview.findChild(QPlainTextEdit, "SharedRequestJSON")
                assert editor.isReadOnly()
                assert editor.toPlainText() == dialog.request_edit.toPlainText()
                select_emoji(editor)
                preview.scroll.ensureWidgetVisible(editor, 0, 0)
                QTest.qWait(80)
                flush()
                assert preview.width() == width
                authority = render_authority_snapshot(dialog)
                state = details_state(editor)
                screenshot(
                    preview,
                    f"{theme}-sharing-json-{width}.png",
                    [width, preview.height()],
                )
                assert render_authority_snapshot(dialog) == authority
                assert tuple(dialog._drafts) == retained_history
                assert tuple(dialog._chat) == retained_chat
                receipt["json_details_checks"].append(
                    {
                        "theme": theme,
                        "scene": "sharing",
                        "width": width,
                        "height": preview.height(),
                        "size_policy": "requested width, content-fitted height",
                        "editor": state,
                        "request_literal_and_read_only": True,
                        "history_chat_authority_unchanged": True,
                    }
                )
                counts["render_authority_checks"] += 1
            finally:
                preview.close()

    def record_plan_recovery(theme, dialog):
        if not receipt["fixture_notes"]:
            receipt["fixture_notes"].append(
                "Recovery busy scenes use the real _start status/lifecycle with a "
                "test-held QThreadPool; no captured callable or provider is run. "
                "The earlier ui capture retained a manual stale busy warning and "
                "is superseded by this fixture-corrected capture."
            )
        assert not dialog._drafts
        assert counts["context_captures"] == (0 if theme == "dark" else 2)
        initial_captures = counts["context_captures"]
        dialog.initiative_slider.setValue(0)
        assert dialog._recovery_kind is None
        assert not dialog.recovery_button.isVisibleTo(dialog)
        dialog.images_checkbox.setChecked(True)
        dialog.images_checkbox.setChecked(False)
        assert dialog._recovery_kind is None
        assert not dialog.recovery_button.isVisibleTo(dialog)
        assert not dialog._context and not dialog._owner
        assert counts["context_captures"] == initial_captures
        assert dialog.status_icon.property("tone") == "info"

        goal = (
            "Offline recovery fixture: retain the observed source and review "
            "separate masks before accepting a draft."
        )
        question = "Offline question: which observed boundary should remain separate?"
        answer = "Offline typed answer: keep the two observed boundaries separate."
        message = "Offline unsent message: do not include this in Update plan."
        dialog.goal_edit.setPlainText(goal)
        dialog._ensure_context(goal)
        dialog._accept_proposal(reference_proposal(dialog._context))
        proposal = reference_proposal(dialog._context)
        proposal["questions"] = [question]
        dialog._accept_proposal(proposal)
        assert len(dialog._drafts) == 2
        field = dialog._question_fields[0][1]
        field.setText(answer)
        dialog.discussion_edit.setPlainText(message)
        retained_chat = tuple(dialog._chat)
        retained_drafts = tuple(dialog._drafts)
        requests = []
        dialog.stage_tabs.setCurrentIndex(2)
        dialog.initiative_slider.setValue(1)
        assert dialog._recovery_kind == "update"
        assert "Balanced" in dialog.status_label.text()
        assert dialog.recovery_button.text() == "Update plan"
        assert dialog.recovery_button.isEnabled()
        assert not requests and not dialog._busy
        assert tuple(dialog._chat) == retained_chat

        def retained():
            assert dialog.goal_edit.toPlainText() == goal
            assert dialog.discussion_edit.toPlainText() == message
            assert dialog._question_fields == [(question, field)]
            assert field.text() == answer
            assert tuple(dialog._drafts) == retained_drafts
            assert tuple(dialog._chat[: len(retained_chat)]) == retained_chat
            assert not dialog._draft_authorized
            assert not dialog.apply_button.isEnabled()
            assert not dialog.run_button.isEnabled()
            assert not dialog._added_ids

        def captures(scene):
            retained()
            for size in ((760, 800), (1040, 820)):
                dialog.resize(*size)
                QTest.qWait(80)
                flush()
                authority = render_authority_snapshot(dialog)
                button = dialog.recovery_button
                assert button.isVisibleTo(dialog)
                assert dialog.status_frame.isAncestorOf(button)
                point = button.mapTo(dialog, button.rect().topLeft())
                assert dialog.rect().contains(button.rect().translated(point))
                if dialog._busy:
                    assert_spinner_animation(button)
                screenshot(
                    dialog, f"{theme}-plan-recovery-{scene}-{size[0]}.png", list(size)
                )
                assert receipt["states"][-1]["requested_size_matched"]
                assert render_authority_snapshot(dialog) == authority
                retained()
                receipt["plan_recovery_checks"].append(
                    {
                        "theme": theme,
                        "scene": scene,
                        "size": list(size),
                        "recovery_kind": dialog._recovery_kind,
                        "recovery_text": button.text(),
                        "recovery_enabled": button.isEnabled(),
                        "recovering": dialog._recovering,
                        "busy_state_real_start_held_pool": scene == "updating-busy",
                        "status": dialog.status_label.text(),
                        "initial_choices_no_stale_warning_or_capture": True,
                        "goal_answers_unsent_message_history_retained": True,
                        "old_plan_not_authorized": True,
                        "render_authority_unchanged": True,
                    }
                )
                counts["render_authority_checks"] += 1

        captures("changed-initiative")

        def send_stub():
            counts["recovery_stub_calls"] += 1
            requests.append(json.loads(dialog.request_edit.toPlainText()))
            dialog._start("proposal", forbidden("model_calls"))

        real_send = dialog._send
        real_pool = workflow_assistant.QThreadPool
        held_jobs = []
        held_pool = SimpleNamespace(start=held_jobs.append)
        workflow_assistant.QThreadPool = SimpleNamespace(
            globalInstance=lambda: held_pool
        )
        dialog._send = send_stub
        try:
            dialog.recovery_button.click()
        finally:
            dialog._send = real_send
            workflow_assistant.QThreadPool = real_pool
        assert len(requests) == 1
        assert len(held_jobs) == 1
        request = requests[0]
        assert request["objective"] == goal
        assert request["planning_preferences"]["mode"] == "balanced"
        assert request["clarifications"] == [{"question": question, "answer": answer}]
        assert message not in json.dumps(request, ensure_ascii=False)
        assert dialog._sending_action == "recovery" and dialog._recovering
        assert dialog.recovery_button.text() == "Updating plan…"
        assert dialog.status_label.text().startswith("Requesting a plan…")
        assert dialog.status_icon.property("tone") == "info"
        captures("updating-busy")

        diagnostic = workflow_proposal_model.SafeModelDiagnostic(
            provider="codex", stage="request", category="network"
        )
        error = workflow_proposal_model.ModelProposalError(
            "Offline synthetic network failure", diagnostic=diagnostic
        )
        dialog._failed(dialog._generation, error)
        assert dialog._recovery_kind == "retry"
        assert dialog._retry_action == "goal"
        assert dialog.recovery_button.text() == "Retry plan"
        assert not dialog._busy and not dialog._recovering
        assert not dialog.recovery_button._spinner_timer.isActive()
        captures("network-failure")
        receipt["plan_recovery_checks"][-1]["explicit_local_update_request"] = request

    def record_plan_review(theme, dialog):
        goal = "Offline Plans review fixture: inspect a conservative mask draft."
        assumptions = [
            "The source is interpreted as a 2D YX image, not a 3D ZYX volume. "
            "Review axes before treating the result as a volume.",
            "Objects are brighter than background; global Otsu is only a "
            "starting point. Uneven signal may need a different threshold.",
            "One connected component represents one candidate object; touching "
            "objects are not separated. Review small bridges and contacts "
            "before accepting labels.",
            "No cavity filling or biological cavity identity is assumed. "
            "Empty-looking regions remain uncertain unless supported by "
            "observed signal.",
            "Objects touching an acquisition edge are left open; border closure "
            "is not inferred. Review cut-off structures before interpreting "
            "completeness.",
            "Smoothing is specified in pixels/voxels; calibration is not "
            "invented. Use only the declared physical calibration when "
            "reviewing size.",
            "Offline unrecognized condition: do not infer the imagined teal "
            "category & do not assert that absence proves empty space.",
            "Literal markup is not instruction: <b>do not merge</b> & "
            "<script>alert('offline')</script>. Preserve this exact text for review.",
        ]
        dialog.goal_edit.setPlainText(goal)
        dialog._ensure_context(goal)
        proposal = reference_proposal(dialog._context)
        proposal["assumptions"] = assumptions
        proposal["summary"] = "Offline earlier checked draft, not a scientific result."
        dialog._accept_proposal(proposal)
        proposal["summary"] = "Offline current checked draft, not a scientific result."
        dialog._accept_proposal(proposal)
        assert len(dialog._drafts) == 2
        assert dialog._prepared.assumptions == tuple(assumptions)
        for index in (0, 1):
            dialog.draft_combo.setCurrentIndex(index)
            flush()
            draft = dialog._drafts[index]
            assert draft.assumptions == tuple(assumptions)
            assert json.loads(draft.proposal_text)["assumptions"] == assumptions
            assert json.loads(dialog.json_edit.toPlainText())["assumptions"] == (
                assumptions
            )
            assert all(item in draft.plan_text for item in assumptions)
            assert all(
                item in dialog.preview_edit.toPlainText() for item in assumptions
            )
            if index == 0:
                assert not dialog.apply_button.isEnabled()
                assert not dialog.run_button.isEnabled()

        def rich_list(label, count):
            document = QTextDocument()
            document.setDefaultFont(label.font())
            document.setHtml(label.text())
            document.setTextWidth(label.width())
            plain = document.toPlainText()
            assert "Assumptions to check" not in plain
            assert "What to check after running" not in plain
            bullets = []
            block = document.begin()
            while block.isValid():
                text_list = block.textList()
                if text_list is not None:
                    assert text_list.format().indent() > 0
                    bullets.append(
                        {
                            "text": block.text(),
                            "list_indent": text_list.format().indent(),
                            "block_height": document.documentLayout()
                            .blockBoundingRect(block)
                            .height(),
                        }
                    )
                block = block.next()
            assert len(bullets) == count
            return plain, bullets

        for size in ((760, 800), (1040, 820)):
            dialog.resize(*size)
            dialog.advanced_checkbox.setChecked(False)
            dialog.assumptions_toggle.setChecked(False)
            dialog.review_checks_toggle.setChecked(False)
            QTest.qWait(80)
            flush()
            scroll = dialog.review_scroll
            label = dialog.assumptions_label
            checks_label = dialog.review_checks_label
            assumptions_plain, assumption_bullets = rich_list(label, len(assumptions))
            checks_plain, review_bullets = rich_list(checks_label, 3)
            assert [item["text"] for item in assumption_bullets] == assumptions
            assert all(item in assumptions_plain for item in assumptions)
            assert "not a complete assessment" in dialog.review_checks_note.text()
            assert "not confirmed facts" in dialog.review_checks_note.text()
            assert "not automatically assessed" in dialog.review_checks_note.text()
            for scene, assumptions_open, checks_open, target in (
                ("collapsed", False, False, dialog.assumptions_frame),
                ("assumptions-start", True, False, dialog.assumptions_frame),
                ("review-checks", False, True, dialog.review_checks_frame),
            ):
                authority = render_authority_snapshot(dialog)
                dialog.assumptions_toggle.setChecked(assumptions_open)
                dialog.review_checks_toggle.setChecked(checks_open)
                flush()
                assert (
                    dialog.assumptions_content.isVisibleTo(dialog) is assumptions_open
                )
                assert dialog.review_checks_content.isVisibleTo(dialog) is checks_open
                assert dialog.review_checks_note.isVisibleTo(dialog) is checks_open
                if scene == "collapsed":
                    for frame, toggle in (
                        (dialog.assumptions_frame, dialog.assumptions_toggle),
                        (dialog.review_checks_frame, dialog.review_checks_toggle),
                    ):
                        assert frame.height() <= toggle.height() + 4
                scroll.verticalScrollBar().setValue(
                    target.mapTo(scroll.widget(), target.rect().topLeft()).y()
                )
                flush()
                for button in (dialog.apply_button, dialog.run_button):
                    assert dialog.plans_actions_frame.isAncestorOf(button)
                    point = button.mapTo(dialog, button.rect().topLeft())
                    assert dialog.rect().contains(button.rect().translated(point))
                screenshot(
                    dialog,
                    f"{theme}-plan-review-{scene}-{size[0]}.png",
                    list(size),
                )
                assert receipt["states"][-1]["requested_size_matched"]
                assert render_authority_snapshot(dialog) == authority
                receipt["plan_review_checks"].append(
                    {
                        "theme": theme,
                        "scene": scene,
                        "assumptions_expanded": assumptions_open,
                        "checks_expanded": checks_open,
                        "size": list(size),
                        "assumptions": assumptions,
                        "display_rich": label.text(),
                        "display_plain": assumptions_plain,
                        "assumption_bullets": assumption_bullets,
                        "review_checks_rich": checks_label.text(),
                        "review_checks_plain": checks_plain,
                        "review_bullets": review_bullets,
                        "review_checks_note": dialog.review_checks_note.text(),
                        "label_height": label.height(),
                        "outer_scroll_maximum": scroll.verticalScrollBar().maximum(),
                        "outer_scroll_value": scroll.verticalScrollBar().value(),
                        "prepared_and_history_exact": True,
                        "advanced_json_and_plain_plan_exact": True,
                        "render_authority_unchanged": True,
                    }
                )
                counts["render_authority_checks"] += 1
        authority = render_authority_snapshot(dialog)
        dialog.advanced_checkbox.setChecked(True)
        flush()
        assert dialog.advanced_group.isVisibleTo(dialog)
        assert json.loads(dialog.json_edit.toPlainText())["assumptions"] == assumptions
        assert render_authority_snapshot(dialog) == authority

    def record_return_navigation(theme, dialog):
        goal = (
            "Offline navigation fixture: review separate observed masks and their "
            "boundaries before creating meshes."
        )
        dialog.goal_edit.setPlainText(goal)

        def capture_navigation(scene, size):
            dialog.resize(*size)
            QTest.qWait(80)
            flush()
            before = render_authority_snapshot(dialog)
            screenshot(
                dialog,
                f"{theme}-navigation-{scene}-{size[0]}.png",
                list(size),
            )
            assert receipt["states"][-1]["requested_size_matched"]
            assert render_authority_snapshot(dialog) == before
            receipt["return_navigation_checks"].append(
                {
                    "theme": theme,
                    "scene": scene,
                    "size": list(size),
                    "retained_drafts": len(dialog._drafts),
                    "next_visible": dialog.next_button.isVisibleTo(dialog),
                    "next_enabled": dialog.next_button.isEnabled(),
                    "next_text": dialog.next_button.text(),
                    "literal_goal_unchanged": dialog.goal_edit.toPlainText() == goal,
                    "render_authority_unchanged": True,
                }
            )
            counts["render_authority_checks"] += 1

        sizes = ((760, 800), (1040, 820))
        assert not dialog._drafts
        for size in sizes:
            capture_navigation("initial-goal", size)

        dialog._ensure_context(goal)
        dialog._accept_proposal(reference_proposal(dialog._context))
        proposal = reference_proposal(dialog._context)
        proposal["questions"] = [
            "Offline question: which observed boundary should remain separate?"
        ]
        dialog._accept_proposal(proposal)
        assert len(dialog._drafts) == 2
        assert len(dialog._question_fields) == 1
        assert not dialog.apply_button.isEnabled()
        assert not dialog.run_button.isEnabled()
        retained_chat = tuple(dialog._chat)
        fields = tuple(dialog._question_fields)
        dialog.stage_tabs.setCurrentIndex(0)
        for size in sizes:
            capture_navigation("returned-goal", size)
            authority = render_authority_snapshot(dialog)
            dialog.next_button.click()
            flush()
            assert dialog.stage_tabs.currentIndex() == 1
            assert dialog.next_button.text() == "View plan"
            assert dialog.next_button.isVisibleTo(dialog)
            dialog.next_button.click()
            flush()
            assert dialog.stage_tabs.currentIndex() == 2
            assert not dialog.next_button.isVisibleTo(dialog)
            dialog.back_button.click()
            flush()
            assert dialog.stage_tabs.currentIndex() == 1
            dialog.back_button.click()
            flush()
            assert dialog.stage_tabs.currentIndex() == 0
            assert render_authority_snapshot(dialog) == authority
            assert tuple(dialog._chat) == retained_chat
            assert tuple(dialog._question_fields) == fields
            assert all(not field.text() for _question, field in fields)

        dialog.draft_combo.setCurrentIndex(0)
        flush()
        assert not dialog.apply_button.isEnabled()
        assert not dialog.run_button.isEnabled()
        authority = render_authority_snapshot(dialog)
        dialog._busy = True
        dialog._job_kind = "proposal"
        dialog._sending_action = "goal"
        dialog._sync_buttons()
        for size in sizes:
            assert dialog.next_button.isVisibleTo(dialog)
            assert not dialog.next_button.isEnabled()
            capture_navigation("busy-old-history", size)
        dialog._busy = False
        dialog._job_kind = ""
        dialog._sync_buttons()
        assert render_authority_snapshot(dialog) == authority
        assert tuple(dialog._chat) == retained_chat

    def record_initiative_label(theme, dialog):
        goal = (
            "Offline layout fixture: distinguish the observed structures as "
            "separate masks, then review their boundaries before creating meshes."
        )
        dialog.goal_edit.setPlainText(goal)
        assert not hasattr(dialog, "initiative_label")
        assert (
            dialog.initiative_card.findChild(QLabel, "AssistantInitiativeChoice")
            is None
        )
        for value, mode, label in (
            (0, "ask_first", "Ask first"),
            (1, "balanced", "Balanced"),
            (2, "use_best_judgment", "Use best judgment"),
        ):
            dialog.initiative_slider.setValue(value)
            dialog._ensure_context(goal)
            assert dialog._planning_preferences().mode == mode
            rich = dialog.initiative_explanation.text()
            document = QTextDocument()
            document.setHtml(rich)
            plain = document.toPlainText()
            assert plain.startswith(label)
            assert plain.count(label) == 1
            first_fragment = document.begin().begin().fragment()
            assert first_fragment.text().startswith(label)
            assert first_fragment.charFormat().font().bold()
            assert dialog.initiative_ask_label.text() == "Ask me more"
            assert dialog.initiative_best_label.text() == "Use best judgment"
            for width, height in ((760, 800), (1040, 820)):
                authority = render_authority_snapshot(dialog)
                dialog.resize(width, height)
                QTest.qWait(80)
                flush()
                scroll = dialog.goal_scroll
                scroll.verticalScrollBar().setValue(
                    scroll.verticalScrollBar().maximum()
                )
                flush()
                viewport = scroll.viewport()
                card = dialog.initiative_card
                point = card.mapTo(viewport, card.rect().topLeft())
                assert viewport.rect().contains(card.rect().translated(point))
                for button in (dialog.preview_button, dialog.prepare_button):
                    assert dialog.goal_actions_frame.isAncestorOf(button)
                    point = button.mapTo(dialog, button.rect().topLeft())
                    assert dialog.rect().contains(button.rect().translated(point))
                screenshot(
                    dialog,
                    f"{theme}-initiative-{mode}-{width}.png",
                    [width, height],
                )
                assert receipt["states"][-1]["requested_size_matched"]
                assert dialog.goal_edit.toPlainText() == goal
                assert render_authority_snapshot(dialog) == authority
                receipt["initiative_label_checks"].append(
                    {
                        "theme": theme,
                        "mode": mode,
                        "size": [width, height],
                        "selected_mode": label,
                        "description_plain": plain,
                        "description_rich": rich,
                        "selected_mode_bold": True,
                        "separate_choice_widget_absent": True,
                        "endpoints_unchanged": True,
                        "card_fully_visible": True,
                        "render_authority_unchanged": True,
                    }
                )
                counts["initiative_choice_checks"] += 1
                counts["render_authority_checks"] += 1

    def record_responsive(theme, dialog):
        screen = dialog.screen().availableGeometry().size()
        large = (min(1440, screen.width() - 80), min(1000, screen.height() - 80))
        assert large[0] >= 1040 and large[1] >= 820
        sizes = ((760, 600), (760, 800), (1040, 820), large)
        goal = (
            "Offline responsive fixture: distinguish the observed structures as "
            "separate masks. Retain the original source and physical calibration. "
            "Use conservative boundary assumptions, do not fill uncertain signal, "
            "and review each reversible draft choice before adding a workflow. "
            "This is authored layout evidence, not a scientific assessment.\n\n"
        ) * 4
        dialog.goal_edit.setPlainText(goal)
        dialog._ensure_context(goal)

        def settle():
            QTest.qWait(80)
            flush()

        def pinned(stage, controls, scroll):
            frame = getattr(dialog, f"{stage}_actions_frame")
            page = getattr(dialog, f"{stage}_page")
            assert frame.parentWidget() is page
            assert frame.objectName() == "AssistantPageActions"
            positions = {}
            for key, control in controls:
                assert control.isVisibleTo(dialog), f"{stage}: hidden {key}"
                assert frame.isAncestorOf(control), f"{stage}: unpinned {key}"
                assert not scroll.widget().isAncestorOf(control)
                point = control.mapTo(dialog, control.rect().topLeft())
                rectangle = control.rect().translated(point)
                assert dialog.rect().contains(rectangle), f"Clipped action: {key}"
                positions[key] = [
                    rectangle.x(),
                    rectangle.y(),
                    rectangle.width(),
                    rectangle.height(),
                ]
                counts["persistent_action_checks"] += 1
            return positions

        def reach(control, scroll):
            viewport = scroll.viewport()
            point = control.mapTo(scroll.widget(), control.rect().topLeft())
            scroll.verticalScrollBar().setValue(
                point.y() - (viewport.height() - control.height()) // 2
            )
            flush()
            point = control.mapTo(viewport, control.rect().topLeft())
            assert viewport.rect().contains(control.rect().translated(point)), (
                f"Input not reachable: {control.objectName()}"
            )
            counts["responsive_input_reachability_checks"] += 1

        def editor_state(editor):
            minimum, maximum = editor.height_bounds()
            assert minimum <= editor.height() <= maximum
            return {
                "height": editor.height(),
                "height_bounds": [minimum, maximum],
                "text_chars": len(editor.toPlainText()),
                "internal_scroll_maximum": editor.verticalScrollBar().maximum(),
            }

        def capture(stage, scroll, controls, size, suffix, expected_positions=None):
            positions = pinned(stage, controls, scroll)
            if expected_positions is not None:
                assert positions == expected_positions, (
                    f"{stage}: persistent actions moved with content scrolling"
                )
            screenshot(
                dialog,
                f"{theme}-responsive-{stage}-{size[0]}x{size[1]}-{suffix}.png",
                list(size),
            )
            assert [dialog.width(), dialog.height()] == list(size), (
                f"Requested responsive size grew: {size}"
            )
            receipt["responsive_checks"].append(
                {
                    "theme": theme,
                    "stage": stage,
                    "size": list(size),
                    "scene": suffix,
                    "actions": positions,
                    "scroll_value": scroll.verticalScrollBar().value(),
                    "scroll_maximum": scroll.verticalScrollBar().maximum(),
                    "goal_editor": editor_state(dialog.goal_edit),
                    "message_editor": editor_state(dialog.discussion_edit),
                }
            )
            return positions

        def idle_pair(stage, scroll, controls, size, advanced=False):
            dialog.resize(*size)
            settle()
            authority = render_authority_snapshot(dialog)
            if stage == "goal":
                reach(dialog.goal_edit, scroll)
            elif stage == "discussion":
                reach(dialog.discussion_edit, scroll)
                for _question, field in dialog._question_fields:
                    reach(field, scroll)
            scroll.verticalScrollBar().setValue(0)
            first = capture(stage, scroll, controls, size, "top")
            if advanced:
                dialog.advanced_checkbox.setChecked(True)
                settle()
            scroll.verticalScrollBar().setValue(scroll.verticalScrollBar().maximum())
            capture(
                stage,
                scroll,
                controls,
                size,
                "advanced-bottom" if advanced else "bottom",
                first,
            )
            assert render_authority_snapshot(dialog) == authority
            counts["responsive_authority_checks"] += 2

        def busy(stage, scroll, controls, action, job="proposal"):
            size = (760, 600)
            dialog.resize(*size)
            settle()
            authority = render_authority_snapshot(dialog)
            dialog._sending_action = action
            dialog._busy, dialog._job_kind = True, job
            dialog._sync_buttons()
            settle()
            scroll.verticalScrollBar().setValue(scroll.verticalScrollBar().maximum())
            capture(stage, scroll, controls, size, f"busy-{action}")
            dialog._busy, dialog._job_kind = False, ""
            dialog._sync_buttons()
            settle()
            assert render_authority_snapshot(dialog) == authority
            counts["responsive_authority_checks"] += 1

        goal_controls = (
            ("preview", dialog.preview_button),
            ("send_goal", dialog.prepare_button),
        )
        for size in sizes:
            idle_pair("goal", dialog.goal_scroll, goal_controls, size)
        busy("goal", dialog.goal_scroll, goal_controls, "goal")

        dialog.initiative_slider.setValue(0)
        dialog._ensure_context(goal)
        dialog._chat.clear()
        dialog._chat_replies.clear()
        dialog._append_chat("You · Goal", goal)
        for index in range(3):
            dialog._append_chat("You", f"Offline message {index + 1}: " + goal[:350])
            dialog._append_chat("Assistant", "Offline retained reply: " + goal[:350])
        proposal = reference_proposal(dialog._context)
        proposal["questions"] = [
            f"Offline question {index + 1}: {topic} " + goal[:220]
            for index, topic in enumerate(
                (
                    "Which observed boundary should remain separate?",
                    "Which source and calibration must be retained?",
                    "What manual review precedes mesh creation?",
                )
            )
        ]
        dialog._accept_proposal(proposal)
        for _question, field in dialog._question_fields:
            field.setText(
                "Offline answer: keep the observed source and calibration. " * 3
            )
        dialog.discussion_edit.setPlainText("Offline unsent message: " + goal)
        dialog.original_goal_toggle.setChecked(True)
        dialog.conversation_toggle.setChecked(True)
        dialogue = dialog._answered_objective()
        dialog._ensure_context(dialogue)
        discussion_controls = (
            ("preview", dialog.discussion_preview_button),
            ("send_answers", dialog.revise_button),
            ("send_message", dialog.send_button),
        )
        for size in sizes:
            idle_pair("discussion", dialog.discussion_scroll, discussion_controls, size)
        busy("discussion", dialog.discussion_scroll, discussion_controls, "message")
        busy("discussion", dialog.discussion_scroll, discussion_controls, "answers")

        proposal = reference_proposal(dialog._context)
        proposal["summary"] = "Offline long plan summary: " + goal
        proposal["assumptions"] = [
            "Offline assumption: preserve observed signal and declared metadata. " * 5
        ] * 3
        dialog._accept_proposal(proposal)
        dialog.stage_tabs.setCurrentIndex(2)
        plan_controls = (("add", dialog.apply_button), ("run", dialog.run_button))
        for size in sizes:
            dialog.advanced_checkbox.setChecked(False)
            idle_pair("plans", dialog.review_scroll, plan_controls, size, advanced=True)
        busy("plans", dialog.review_scroll, plan_controls, "run", job="run")

    for theme in ("dark", "light"):
        QDir.addSearchPath(f"theme_{theme}", str(_theme_path(theme)))
        app.setStyleSheet(get_stylesheet(theme))
        parent = QWidget()
        dialog = workflow_assistant.WorkflowAssistantDialog(
            parent,
            sources=lambda: [("input", "Synthetic cached YX image · 24 × 32")],
            capture=capture,
            assert_current=assert_current,
            apply=apply,
            run=run,
        )
        dialog.setWindowTitle("VIPP · Workflow assistant — OFFLINE UI evidence")
        dialog.resize(1040, 820)
        dialog.show()

        def record(theme=theme, dialog=dialog, parent=parent):
            try:
                assert dialog._codex_authenticated and not dialog._connection_busy
                if args.json_details:
                    record_json_details(theme, dialog)
                    return
                if args.plan_recovery:
                    record_plan_recovery(theme, dialog)
                    return
                if args.plan_review:
                    record_plan_review(theme, dialog)
                    return
                if args.return_navigation:
                    record_return_navigation(theme, dialog)
                    return
                if args.initiative_label:
                    record_initiative_label(theme, dialog)
                    return
                if args.responsive:
                    record_responsive(theme, dialog)
                    return
                dialog.goal_edit.setPlainText(
                    "Identify the two observed structures as separate masks, "
                    "then review their boundaries before creating meshes."
                )
                screenshot(dialog, f"{theme}-goal-1040.png", [1040, 820])
                assert dialog.initiative_slider.value() == 2
                for value, mode, label in (
                    (0, "ask_first", "Ask first"),
                    (1, "balanced", "Balanced"),
                    (2, "use_best_judgment", "Use best judgment"),
                ):
                    previous_context = dialog._context
                    previous_preferences = (
                        previous_context.planning_preferences.to_dict()
                        if previous_context is not None
                        else None
                    )
                    dialog.initiative_slider.setValue(value)
                    for width, height in ((1040, 820), (760, 800)):
                        dialog.resize(width, height)
                        flush()
                        goal_scroll = dialog.goal_scroll
                        viewport = goal_scroll.viewport()
                        for control in (
                            dialog.goal_edit,
                            dialog.initiative_card,
                            dialog.prepare_button,
                        ):
                            point = control.mapTo(
                                goal_scroll.widget(), control.rect().topLeft()
                            )
                            goal_scroll.verticalScrollBar().setValue(
                                point.y() - (viewport.height() - control.height()) // 2
                            )
                            flush()
                            position = control.mapTo(viewport, control.rect().topLeft())
                            assert viewport.rect().contains(
                                control.rect().translated(position)
                            ), f"Goal control not reachable: {control.objectName()}"
                            counts["goal_reachability_checks"] += 1
                        goal_scroll.verticalScrollBar().setValue(
                            goal_scroll.verticalScrollBar().maximum()
                        )
                        screenshot(
                            dialog,
                            f"{theme}-initiative-{mode}-{width}.png",
                            [width, height],
                        )
                        goal_state = receipt["states"][-1]["assistant_state"][
                            "goal_scroll"
                        ]
                        assert goal_state["initiative_fully_in_viewport"]
                        assert goal_state["send_goal_fully_in_viewport"]
                    request = dialog._ensure_context(
                        dialog.goal_edit.toPlainText().strip()
                    )
                    assert request["planning_preferences"]["mode"] == mode
                    assert request["planning_preferences"]["max_questions"] == 3 - value
                    if previous_context is not None:
                        assert previous_context.planning_preferences.to_dict() == (
                            previous_preferences
                        )
                        assert dialog._context.source_summary == (
                            previous_context.source_summary
                        )
                        assert dialog._context.baseline_fingerprint == (
                            previous_context.baseline_fingerprint
                        )
                    preview = dialog._sharing_preview_dialog(request)
                    preview.show()
                    preview_text = "\n".join(
                        control.text() for control in preview.findChildren(QLabel)
                    )
                    assert label in preview_text
                    screenshot(preview, f"{theme}-initiative-{mode}-sharing.png")
                    receipt["initiative_preview_checks"].append(
                        {
                            "theme": theme,
                            "preferences": request["planning_preferences"],
                            "choice_visible_in_preview": label in preview_text,
                            "prior_context_policy_unchanged": True,
                        }
                    )
                    preview.reject()
                    counts["initiative_choice_checks"] += 1
                dialog.initiative_slider.setValue(2)
                dialog.resize(1040, 820)
                # Simulate the request's visible state only. No request worker,
                # provider, authentication, Add, Run or calculation is invoked.
                dialog._busy = True
                dialog._job_kind = "proposal"
                dialog._sending_action = "goal"
                dialog._sync_buttons()
                assert_spinner_animation(dialog.prepare_button)
                screenshot(dialog, f"{theme}-goal-loading-1040.png", [1040, 820])
                # Use the previously accepted native narrow capture height.
                # The receipt still records any platform-imposed growth.
                dialog.resize(760, 800)
                screenshot(dialog, f"{theme}-goal-loading-760.png", [760, 800])
                dialog._busy = False
                dialog._job_kind = ""
                dialog._sync_buttons()
                dialog.resize(1040, 820)
                # The existing question-bearing layout fixtures must respect
                # the provider contract's three-question Ask first budget.
                dialog.initiative_slider.setValue(0)
                context, token = capture("input")
                dialog._context, dialog._owner = context, token
                preview_request = context.public_request(
                    "Offline preview fixture: separate the observed objects."
                )
                dialog.request_edit.setPlainText(
                    json.dumps(preview_request, indent=2, ensure_ascii=False)
                )
                preview = dialog._sharing_preview_dialog(preview_request)
                preview.show()
                screenshot(preview, f"{theme}-sharing-short.png")
                preview.findChild(QCheckBox, "SharedRequestAdvancedToggle").setChecked(
                    True
                )
                screenshot(preview, f"{theme}-sharing-advanced.png")
                preview.reject()
                long_request = context.public_request(
                    "Offline long preview fixture. " * 120
                )
                long_request["clarifications"] = [
                    {
                        "question": (
                            f"Offline fixture question {index + 1}: review boundaries?"
                        ),
                        "answer": (
                            "Offline fixture answer: retain observed signal only. " * 6
                        ),
                    }
                    for index in range(8)
                ]
                dialog.request_edit.setPlainText(
                    json.dumps(long_request, indent=2, ensure_ascii=False)
                )
                preview = dialog._sharing_preview_dialog(long_request)
                preview.show()
                screenshot(preview, f"{theme}-sharing-long.png")
                assert preview.scroll.verticalScrollBar().maximum() > 0
                preview.reject()
                initial = reference_proposal(context)
                initial["questions"] = [
                    "Should masks follow observed signal, or infer missing geometry?",
                    "What evidence should decide where the two structures overlap?",
                ]
                dialog._accept_proposal(initial)
                for (_question, field), answer in zip(
                    dialog._question_fields,
                    (
                        "Observed signal only for this initial draft.",
                        "Review shared boundaries manually.",
                    ),
                    strict=True,
                ):
                    field.setText(answer)
                screenshot(dialog, f"{theme}-discussion-1040.png", [1040, 820])
                dialog.discussion_edit.setPlainText(
                    "Offline message fixture: retain the observed boundaries."
                )
                dialog._sending_action = "message"
                dialog._busy = True
                dialog._job_kind = "proposal"
                dialog._sync_buttons()
                assert_spinner_animation(dialog.send_button)
                screenshot(dialog, f"{theme}-discussion-loading-1040.png", [1040, 820])
                dialog._busy = False
                dialog._job_kind = ""
                dialog._sync_buttons()
                revised = reference_proposal(context)
                revised["summary"] = (
                    "Offline sample revision: inspect masks before mesh creation."
                )
                dialog._accept_proposal(revised)
                assert dialog.draft_combo.count() == 2
                screenshot(dialog, f"{theme}-plans-1040.png", [1040, 820])
                dialog.resize(760, 800)
                screenshot(dialog, f"{theme}-plans-760.png", [760, 800])
                dialog.stage_tabs.setCurrentIndex(0)
                screenshot(dialog, f"{theme}-goal-760.png", [760, 800])
                settings = workflow_assistant._AssistantSettingsDialog(dialog)
                settings.show()
                screenshot(settings, f"{theme}-settings.png")
                settings.reject()
                # Long-content evidence stays entirely local. These are authored
                # fixture goals, messages and answers, never user conversation.
                long_goal = (
                    "Offline long-goal fixture: distinguish the two observed "
                    "structures as separate masks and retain their shared boundary "
                    "for review. Do not infer unseen geometry or fill missing signal "
                    "without an explicit later decision. Keep the source image and "
                    "its declared calibration unchanged while drafting existing "
                    "processing steps. Show the assumptions and ask for the evidence "
                    "needed to resolve uncertain regions before adding a workflow.\n\n"
                ) * 3
                dialog.goal_edit.setPlainText(long_goal)
                context, token = capture("input")
                dialog._context, dialog._owner = context, token
                long_proposal = reference_proposal(context)
                long_proposal["summary"] = (
                    "Offline discussion fixture: review the proposed observed-signal "
                    "masks, boundary policy and calibration before accepting any plan."
                )
                long_proposal["questions"] = [
                    f"Offline question {index + 1}: {topic} "
                    "Please explain the review evidence and the permitted handling "
                    "of uncertain signal. Retain the original image metadata and "
                    "distinguish a visible structure from an inferred reconstruction "
                    "so this drafting conversation does not imply scientific proof."
                    for index, topic in enumerate(
                        (
                            "How should faint edges be separated from background?",
                            "What should happen at the boundary between the two masks?",
                            "Which source and physical calibration should be retained?",
                        )
                    )
                ]
                dialog._chat.clear()
                dialog._append_chat("You · Goal", long_goal)
                for index in range(3):
                    dialog._append_chat(
                        "You · Offline fixture",
                        f"Offline message {index + 1}: keep the tentative masks "
                        "separate while reviewing faint edges and uncertain shared "
                        "boundaries. The image remains the original analysis source; "
                        "this is a local layout fixture and is not sent to a provider.",
                    )
                    dialog._append_chat(
                        "Assistant · Offline fixture",
                        "Offline reply fixture: the original goal is retained with "
                        "this discussion. I can describe a starting workflow while "
                        "keeping uncertain regions explicit. The three questions "
                        "below need answers before the proposal can be accepted.",
                    )
                # The current question-bearing reply follows earlier exchanges,
                # matching the real order of a returned proposal in discussion.
                dialog._accept_proposal(long_proposal)
                for question, field in dialog._question_fields:
                    field.setText(
                        "Offline answer fixture: use observed signal only and retain "
                        "the declared source metadata. Keep this policy explicit, "
                        "then ask for manual boundary review before adding nodes."
                    )
                    assert question.startswith("Offline question")
                dialog.discussion_edit.setPlainText(
                    "Offline unsent composer fixture: keep the image calibration "
                    "and boundary-review choices visible in the next draft.\n"
                    "This message is retained locally; the harness never sends it."
                )
                dialog.resize(1040, 820)
                dialog.stage_tabs.setCurrentIndex(1)
                # Let deferred document wrapping finish before choosing a
                # viewport position; this waits only on these owned Qt windows.
                QTest.qWait(80)
                flush()
                assert dialog.original_goal_label.text() == long_goal
                assert not dialog.original_goal_toggle.isChecked()
                assert not dialog.conversation_toggle.isChecked()
                scroll = dialog.discussion_scroll
                scroll.verticalScrollBar().setValue(0)
                screenshot(
                    dialog, f"{theme}-long-discussion-collapsed-1040.png", [1040, 820]
                )

                def capture_disclosure(toggle, name, requested):
                    assert not toggle.isChecked()
                    retained_chat = tuple(dialog._chat)
                    toggle.setChecked(True)
                    QTest.qWait(80)
                    scroll.verticalScrollBar().setValue(0)
                    screenshot(dialog, name, requested)
                    assert tuple(dialog._chat) == retained_chat
                    if toggle is dialog.original_goal_toggle:
                        assert dialog.original_goal_label.isVisibleTo(dialog)
                    else:
                        assert dialog.transcript_edit.isVisibleTo(dialog)
                    counts["disclosure_retention_checks"] += 1
                    toggle.setChecked(False)
                    QTest.qWait(80)

                capture_disclosure(
                    dialog.original_goal_toggle,
                    f"{theme}-long-discussion-expanded-1040.png",
                    [1040, 820],
                )
                capture_disclosure(
                    dialog.conversation_toggle,
                    f"{theme}-long-discussion-history-1040.png",
                    [1040, 820],
                )

                def capture_questions(name, requested):
                    flush()
                    point = dialog.questions_group.mapTo(
                        scroll.widget(), dialog.questions_group.rect().topLeft()
                    )
                    scroll.verticalScrollBar().setValue(point.y())
                    screenshot(dialog, name, requested)

                def capture_composer(name, requested):
                    viewport = scroll.viewport()
                    for _question, field in dialog._question_fields:
                        point = field.mapTo(scroll.widget(), field.rect().topLeft())
                        scroll.verticalScrollBar().setValue(
                            point.y() - (viewport.height() - field.height()) // 2
                        )
                        flush()
                        position = field.mapTo(viewport, field.rect().topLeft())
                        assert viewport.rect().contains(
                            field.rect().translated(position)
                        ), (
                            "Question field not reachable at "
                            f"{position.x()}, {position.y()}"
                        )
                        counts["question_reachability_checks"] += 1
                    flush()
                    scroll.verticalScrollBar().setValue(
                        scroll.verticalScrollBar().maximum()
                    )
                    screenshot(dialog, name, requested)
                    state = receipt["states"][-1]["discussion_state"]
                    assert state["composer_fully_in_viewport"]
                    assert state["send_message_fully_in_viewport"]
                    assert state["question_count"] == 3
                    assert all(
                        row["answer_chars"] > 100 for row in state["question_rows"]
                    )

                capture_questions(
                    f"{theme}-long-discussion-questions-1040.png", [1040, 820]
                )
                capture_composer(
                    f"{theme}-long-discussion-composer-1040.png", [1040, 820]
                )
                dialog.resize(760, 800)
                QTest.qWait(80)
                scroll.verticalScrollBar().setValue(0)
                screenshot(
                    dialog, f"{theme}-long-discussion-collapsed-760.png", [760, 800]
                )
                capture_disclosure(
                    dialog.original_goal_toggle,
                    f"{theme}-long-discussion-expanded-760.png",
                    [760, 800],
                )
                capture_disclosure(
                    dialog.conversation_toggle,
                    f"{theme}-long-discussion-history-760.png",
                    [760, 800],
                )
                capture_questions(
                    f"{theme}-long-discussion-questions-760.png", [760, 800]
                )
                capture_composer(
                    f"{theme}-long-discussion-composer-760.png", [760, 800]
                )
                retained_goal = dialog.goal_edit.toPlainText()
                retained_chat = tuple(dialog._chat)
                retained_questions = tuple(
                    (question, field, field.text())
                    for question, field in dialog._question_fields
                )
                dialog.initiative_slider.setValue(2)
                assert dialog.goal_edit.toPlainText() == retained_goal
                assert tuple(dialog._chat) == retained_chat
                assert (
                    tuple(
                        (question, field, field.text())
                        for question, field in dialog._question_fields
                    )
                    == retained_questions
                )
                assert not dialog._draft_authorized
                assert not dialog.apply_button.isEnabled()
                assert not dialog.run_button.isEnabled()
                receipt["initiative_retention_checks"].append(
                    {
                        "theme": theme,
                        "goal_unchanged": True,
                        "chat_unchanged": True,
                        "question_fields_and_answers_unchanged": True,
                        "prior_draft_authority_revoked": True,
                    }
                )
                counts["initiative_retention_checks"] += 1
                dialog._ensure_context(dialog.goal_edit.toPlainText().strip())
                blank_proposal = reference_proposal(dialog._context)
                blank_proposal["summary"] = (
                    "Offline blank-answer fixture: request a conservative draft "
                    "with explicit assumptions; no graph is added or calculated."
                )
                blank_proposal["questions"] = [
                    "Offline essential question: which observed structure needs "
                    "priority during this boundary review?"
                ]
                dialog._accept_proposal(blank_proposal)
                dialog.discussion_edit.clear()
                assert len(dialog._question_fields) == 1
                question, field = dialog._question_fields[0]
                field.clear()
                assert "leave blank" in field.placeholderText().casefold(), (
                    f"Blank-answer guidance missing: {field.placeholderText()}"
                )
                dialog._sync_buttons()
                assert dialog.revise_button.isEnabled()
                assert not dialog.apply_button.isEnabled()
                assert not dialog.run_button.isEnabled()
                # Inspect the unanswered form before the local send stub adds
                # the user's explicit deferral event to retained chat history.
                for width, height in ((1040, 820), (760, 800)):
                    dialog.resize(width, height)
                    QTest.qWait(80)
                    point = dialog.questions_group.mapTo(
                        scroll.widget(), dialog.questions_group.rect().topLeft()
                    )
                    scroll.verticalScrollBar().setValue(point.y())
                    screenshot(
                        dialog,
                        f"{theme}-discussion-blank-answers-{width}.png",
                        [width, height],
                    )
                captured_requests = []

                def blank_revision_stub():
                    counts["blank_revision_stub_calls"] += 1
                    captured_requests.append(
                        json.loads(dialog.request_edit.toPlainText())
                    )

                real_send = dialog._send
                dialog._send = blank_revision_stub
                try:
                    dialog._send_answers_clicked()
                finally:
                    dialog._send = real_send
                assert len(captured_requests) == 1
                blank_request = captured_requests[0]
                assert blank_request["planning_preferences"]["mode"] == (
                    "use_best_judgment"
                )
                assert blank_request["objective"] == long_goal, (
                    "Blank revision did not retain the literal original goal"
                )
                assert blank_request["deferred_questions"] == [question]
                assert not dialog._busy and not field.text()
                receipt["blank_revision_checks"].append(
                    {
                        "theme": theme,
                        "question": question,
                        "answer_placeholder": field.placeholderText(),
                        "answer_height": field.height(),
                        "preferences": blank_request["planning_preferences"],
                        "clarifications": blank_request.get("clarifications", []),
                        "deferred_questions": blank_request["deferred_questions"],
                        "only_local_stub_called": True,
                        "add_disabled": not dialog.apply_button.isEnabled(),
                        "run_disabled": not dialog.run_button.isEnabled(),
                    }
                )
                preview = dialog._sharing_preview_dialog(blank_request)
                preview.show()
                deferred_labels = preview.findChildren(QLabel, "SharedDeferredQuestion")
                assert len(deferred_labels) == 1
                assert question in deferred_labels[0].text()
                screenshot(preview, f"{theme}-sharing-deferred-question.png")
                preview.reject()
                counts["deferred_preview_checks"] += 1
                # Exercise the real request-chat attribution path with entirely
                # authored, multi-line question/answer content and a local reply.
                qa_goal = (
                    "Offline goal fixture: keep observed boundaries explicit "
                    "while choosing a conservative draft."
                )
                qa_question = (
                    "Should uncertain faint edges stay separate for review?\n"
                    "Which observed boundary needs priority in this draft?"
                )
                qa_answer = (
                    "Keep the two observed masks separate.\n"
                    "Use conservative edge handling without filling uncertain signal."
                    "\n\nRetain the original calibration and ask for visual review "
                    "before adding nodes."
                )
                dialog.goal_edit.setPlainText(qa_goal)
                dialog._chat.clear()
                dialog._chat_replies.clear()
                dialogue = ProposalDialogue(qa_goal, ((qa_question, qa_answer),))
                qa_request = dialog._ensure_context(dialogue)
                assert qa_request["clarifications"] == [
                    {"question": qa_question, "answer": qa_answer}
                ]
                dialog._append_request_chat(dialogue)
                qa_response = reference_proposal(dialog._context)
                qa_response["summary"] = (
                    "Offline response fixture: keep the masks separate and make "
                    "edge assumptions explicit.\n\nReview the candidate before Add; "
                    "Run remains a separate action."
                )
                dialog._accept_proposal(qa_response)
                dialog.stage_tabs.setCurrentIndex(1)
                dialog.conversation_toggle.setChecked(True)
                raw_chat = tuple(dialog._chat)
                qa_authority = render_authority_snapshot(dialog)
                assert ("You", f"{qa_question}\n{qa_answer}") in raw_chat
                for width, height in ((1040, 820), (760, 800)):
                    dialog.resize(width, height)
                    QTest.qWait(80)
                    flush()
                    rendered = dialog.transcript_edit.toPlainText()
                    assert rendered.count("Assistant’s question") == 1
                    assert rendered.count("Your answer") == 1
                    assert qa_question in rendered and qa_answer in rendered
                    assert tuple(dialog._chat) == raw_chat
                    assert render_authority_snapshot(dialog) == qa_authority
                    point = dialog.conversation_frame.mapTo(
                        scroll.widget(), dialog.conversation_frame.rect().topLeft()
                    )
                    scroll.verticalScrollBar().setValue(point.y())
                    screenshot(
                        dialog,
                        f"{theme}-conversation-question-answer-{width}.png",
                        [width, height],
                    )
                    assert render_authority_snapshot(dialog) == qa_authority
                    counts["render_authority_checks"] += 1
                    receipt["question_answer_checks"].append(
                        {
                            "theme": theme,
                            "width": width,
                            "question_newlines": qa_question.count("\n"),
                            "answer_newlines": qa_answer.count("\n"),
                            "question_label_count": rendered.count(
                                "Assistant’s question"
                            ),
                            "answer_label_count": rendered.count("Your answer"),
                            "literal_question_preserved": qa_question in rendered,
                            "literal_answer_preserved": qa_answer in rendered,
                            "plain_chat_unchanged": tuple(dialog._chat) == raw_chat,
                            "render_authority_unchanged": True,
                            "transcript_internal_scroll_maximum": (
                                dialog.transcript_edit.verticalScrollBar().maximum()
                            ),
                        }
                    )
                    counts["question_answer_response_checks"] += 1
                # Four alternating authored messages show both short and long
                # bubbles without any provider request or scientific calculation.
                dialog._chat.clear()
                dialog._chat_replies.clear()
                dialog._append_chat("You · Goal", qa_goal)
                turns = (
                    ("You", "Keep both masks separate."),
                    (
                        "Assistant",
                        "Use a conservative draft and review the boundary.",
                    ),
                    (
                        "You",
                        "Keep the original source and calibration unchanged.\n"
                        "Do not merge uncertain edges or fill gaps without review. "
                        "Make each reversible draft choice explicit, and keep Add "
                        "and Run as separate actions.",
                    ),
                    (
                        "Assistant",
                        "I can describe a starting candidate using only the "
                        "observed structures, with the boundary convention recorded "
                        "as an assumption.\n\nThe draft is not a scientific "
                        "assessment. Review it before Add; Run remains your "
                        "separate choice.",
                    ),
                )
                for speaker, text in turns:
                    dialog._append_chat(speaker, text)
                retained_turns = tuple(dialog._chat)
                turns_authority = render_authority_snapshot(dialog)
                for width, height in ((1040, 820), (760, 800)):
                    dialog.resize(width, height)
                    QTest.qWait(80)
                    flush()
                    rendered = dialog.transcript_edit.toPlainText()
                    assert all(text in rendered for _speaker, text in turns)
                    assert "Assistant’s question" not in rendered
                    assert "Your answer" not in rendered
                    assert tuple(dialog._chat) == retained_turns
                    assert render_authority_snapshot(dialog) == turns_authority
                    point = dialog.conversation_frame.mapTo(
                        scroll.widget(), dialog.conversation_frame.rect().topLeft()
                    )
                    scroll.verticalScrollBar().setValue(point.y())
                    screenshot(
                        dialog,
                        f"{theme}-conversation-alternating-bubbles-{width}.png",
                        [width, height],
                    )
                    assert render_authority_snapshot(dialog) == turns_authority
                    counts["render_authority_checks"] += 1
                    bubbles = receipt["states"][-1]["discussion_state"]["chat_bubbles"]
                    assert [bubble["role"] for bubble in bubbles] == [
                        "user",
                        "assistant",
                        "user",
                        "assistant",
                    ]
                    receipt["multi_turn_chat_checks"].append(
                        {
                            "theme": theme,
                            "width": width,
                            "literal_messages_preserved": True,
                            "raw_chat_unchanged": tuple(dialog._chat) == retained_turns,
                            "render_authority_unchanged": True,
                            "short_long_bubbles": bubbles,
                            "transcript_internal_scroll_maximum": (
                                dialog.transcript_edit.verticalScrollBar().maximum()
                            ),
                        }
                    )
                    counts["multi_turn_chat_checks"] += 1
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
                receipt["error_traceback"] = traceback.format_exc()
            finally:
                dialog.close()
                parent.close()
                app.quit()

        QTimer.singleShot(150, record)
        app.exec()
        if errors:
            break

    receipt["errors"] = errors
    receipt["source_sha256_after"] = hashlib.sha256(source.tobytes()).hexdigest()
    receipt["ui_source_sha256_after"] = hashlib.sha256(ui_path.read_bytes()).hexdigest()
    receipt["ui_sources_sha256_after"] = ui_fingerprints()
    receipt["planning_policy_sha256_after"] = hashlib.sha256(
        policy_path.read_bytes()
    ).hexdigest()
    receipt["source_unchanged"] = (
        receipt["source_sha256_before"] == receipt["source_sha256_after"]
    )
    receipt["ui_source_unchanged"] = (
        receipt["ui_sources_sha256_before"] == receipt["ui_sources_sha256_after"]
    )
    receipt["planning_policy_unchanged"] = (
        receipt["planning_policy_sha256_before"]
        == receipt["planning_policy_sha256_after"]
    )
    receipt["source_readonly"] = not source.flags.writeable
    receipt["requested_sizes_matched"] = all(
        item["requested_size_matched"] for item in receipt["states"]
    )
    receipt["frames_fit_available_screen"] = all(
        item["fits_available_screen"] for item in receipt["states"]
    )
    receipt["passed"] = (
        not errors
        and len(receipt["states"])
        == (
            8
            if args.json_details
            else 12
            if args.plan_recovery
            else 12
            if args.plan_review
            else 12
            if args.return_navigation or args.initiative_label
            else 56
            if args.responsive
            else 76
        )
        and receipt["source_unchanged"]
        and receipt["ui_source_unchanged"]
        and receipt["planning_policy_unchanged"]
        and receipt["frames_fit_available_screen"]
        and all(not item["outer_control_clipping"] for item in receipt["states"])
        and all(
            counts[name] == 0
            for name in (
                "auth_calls",
                "model_calls",
                "model_discovery_calls",
                "add_spy_calls",
                "run_spy_calls",
            )
        )
    )
    (output / "native-window-receipt.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "passed": receipt["passed"],
                "states": len(receipt["states"]),
                "errors": errors,
                "counts": counts,
            }
        )
    )
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
