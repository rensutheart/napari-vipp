"""Compact input summaries retain readable, complete scientific context."""

from dataclasses import replace

import pytest
from pytestqt.exceptions import TimeoutError as QtWaitTimeoutError
from qtpy.compat import isalive
from qtpy.QtCore import QCoreApplication, QEvent, Qt, QTimer
from qtpy.QtGui import QFont
from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget

from napari_vipp._tests.test_ui_connected_inputs import _palette
from napari_vipp.ui import inspector as inspector_module
from napari_vipp.ui.connected_inputs import ConnectedInputBinding, ConnectedInputsCard
from napari_vipp.ui.design import SPACE_SM, SPACE_XS, body_point_size
from napari_vipp.ui.inspector import constrain_layout_minimum_height


def _bindings():
    return [
        ConnectedInputBinding(
            "Moving image / time series",
            "image",
            "Synthetic moving fluorescence source <literal>",
            "Image",
            "TCZYX: 3 × 2 × 32 × 128 × 128 · float32 (32-bit float) · "
            "sampling Z 1 µm, Y 0.25 µm, X 0.25 µm · axes inferred",
        ),
        ConnectedInputBinding(
            "Reference image",
            "image",
            "Synthetic reference fluorescence source",
            "Image",
            "ZYX: 32 × 128 × 128 · float32 (32-bit float) · "
            "sampling Z 1 µm, Y 0.25 µm, X 0.25 µm",
        ),
    ]


def _host(qtbot, points):
    host = QWidget()
    qtbot.addWidget(host)
    font = QFont(host.font())
    font.setPointSizeF(points)
    host.setFont(font)
    return host


def _constraint_path(monkeypatch, portable):
    if portable:
        monkeypatch.setattr(
            inspector_module,
            "_independent_layout_constraints_available",
            lambda _: False,
        )
    elif not callable(getattr(QVBoxLayout, "setSizeConstraints", None)):
        pytest.skip("This binding has no native independent-axis layout constraints.")


def _shown_card(qtbot, points):
    host = _host(qtbot, points)
    layout = QVBoxLayout(host)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(SPACE_SM)
    card = ConnectedInputsCard(host)
    card.set_bindings(_bindings())
    following = QLabel("Registration settings follow the complete context", host)
    following.setWordWrap(True)
    layout.addWidget(card)
    layout.addWidget(following)
    constrain_layout_minimum_height(layout)
    host.resize(260, 200)
    host.show()
    return host, card, following


def _wait_for_card_layout(qtbot, host, card, width):
    previous = None
    stable_observations = 0

    def settled():
        nonlocal previous, stable_observations
        previous_snapshot = previous
        previous = None
        assert host.width() == card.width() == width
        assert card.height() >= card.heightForWidth(width)
        assert not card._font_refresh_timer.isActive()
        geometry = [card.geometry().getRect(), card.minimumHeight()]
        for owner in (host, card, *card.rows):
            guard = owner._vipp_minimum_height_constraint
            assert not guard._geometry_sync_timer.isActive()
            geometry.extend((owner.minimumHeight(), owner.geometry().getRect()))
        for row in card.rows:
            assert card.rect().contains(row.geometry())
            for label in (row.role_label, row.source_label, row.scientific_label):
                assert row.rect().contains(label.geometry())
                assert label.height() >= label.heightForWidth(label.width())
                font_observation = {
                    "label": label.objectName(),
                    "label_points": label.font().pointSizeF(),
                    "host_points": host.font().pointSizeF(),
                    "card_points": card.font().pointSizeF(),
                    "title_points": card.title_label.font().pointSizeF(),
                    "body_for_card": body_point_size(card),
                    "body_for_title": body_point_size(card.title_label),
                    "card_has_own_font": card.testAttribute(Qt.WA_SetFont),
                    "recorded_theme_refreshes": getattr(
                        card, "_test_theme_refreshes", None
                    ),
                }
                if label.font().pointSizeF() != host.font().pointSizeF():
                    raise AssertionError(f"Connected-input font: {font_observation!r}")
                geometry.append(label.geometry().getRect())
        current = tuple(geometry)
        stable_observations = (
            stable_observations + 1 if current == previous_snapshot else 1
        )
        previous = current
        return stable_observations >= 3

    qtbot.waitUntil(settled, timeout=2000)


@pytest.mark.parametrize("points", [9, 16])
@pytest.mark.parametrize("width", [260, 400])
def test_card_saves_measured_height_in_spacing_not_scientific_text(
    qtbot, points, width
):
    host = _host(qtbot, points)
    host_layout = QVBoxLayout(host)
    host_layout.setContentsMargins(0, 0, 0, 0)
    host_layout.setSpacing(SPACE_SM)
    compact = ConnectedInputsCard(host)
    previous_spacing = ConnectedInputsCard(host)
    for card in (compact, previous_spacing):
        card.set_bindings(_bindings())
        card.ensurePolished()
        card.resize(width, 1000)
    previous_spacing.title_label.setContentsMargins(
        SPACE_SM, SPACE_SM, SPACE_SM, SPACE_XS
    )
    for row in previous_spacing.rows:
        row.layout().setContentsMargins(SPACE_SM, SPACE_SM, SPACE_SM, SPACE_SM)
        row.layout().setSpacing(SPACE_SM)
        row.layout().itemAt(1).layout().setSpacing(SPACE_XS)

    host_layout.addWidget(compact)
    host_layout.addWidget(previous_spacing)
    constrain_layout_minimum_height(host_layout)
    host.resize(width, 200)
    host.show()
    _wait_for_card_layout(qtbot, host, compact, width)
    _wait_for_card_layout(qtbot, host, previous_spacing, width)

    compact.layout().invalidate()
    previous_spacing.layout().invalidate()
    assert compact.layout().hasHeightForWidth()
    compact_height = compact.layout().totalHeightForWidth(width)
    previous_height = previous_spacing.layout().totalHeightForWidth(width)
    assert 0 < compact_height < previous_height
    # Two rows each save 8px vertical padding; the header saves another 4px.
    assert previous_height - compact_height >= 20
    assert [row.binding for row in compact.rows] == _bindings()
    for row in compact.rows:
        assert row.source_label.text() == row.binding.source_summary
        assert row.scientific_label.text() == row.binding.scientific_summary
        assert row.source_label.font().pointSizeF() >= points
        assert row.scientific_label.font().pointSizeF() >= points


@pytest.mark.parametrize("dark", [True, False])
@pytest.mark.parametrize("points", [9, 16])
def test_input_card_hierarchy_is_bold_without_shrinking_or_replacing_context(
    qtbot, dark, points
):
    host = _host(qtbot, points)
    card = ConnectedInputsCard(host)
    card.set_bindings(_bindings())
    card.refresh_theme(_palette(dark=dark))
    card.ensurePolished()
    first_style = card.styleSheet()
    for _ in range(3):
        card.refresh_theme(_palette(dark=dark))
    assert card.styleSheet() == first_style
    assert card.title_label.font().weight() >= QFont.Bold
    assert card.title_label.font().pointSizeF() >= points
    assert (
        card.title_label.font().pointSizeF()
        == card.rows[0].source_label.font().pointSizeF()
    )
    for row in card.rows:
        assert row.role_label.font().weight() >= QFont.DemiBold
        for label in (row.role_label, row.source_label, row.scientific_label):
            assert label.wordWrap()
            assert label.textFormat() == Qt.PlainText
        for label in (row.source_label, row.scientific_label):
            assert label.textInteractionFlags() & Qt.TextSelectableByMouse
        assert row.scientific_label.text() in row.accessibleName()
        assert row.binding.source_title in row.source_label.text()
        assert "&lt;literal&gt;" in card.rows[0].toolTip()
        assert row.property("hasDivider") == (row is card.rows[1])
    rows = tuple(card.rows)
    card.set_bindings([replace(binding) for binding in _bindings()])
    assert tuple(card.rows) == rows


@pytest.mark.parametrize("points", [9, 16])
@pytest.mark.parametrize("portable", [False, True], ids=["native", "portable"])
def test_compact_card_wraps_at_owner_width_without_overlap_or_forced_width(
    qtbot, monkeypatch, points, portable
):
    _constraint_path(monkeypatch, portable)
    host = _host(qtbot, points)
    layout = QVBoxLayout(host)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(SPACE_SM)
    # Match the real parameter-content layout: default outer alignment, not
    # AlignTop's width-independent alignment rectangle.
    assert layout.alignment() == Qt.AlignmentFlag(0)
    card = ConnectedInputsCard(host)
    card.set_bindings(_bindings())
    following = QLabel(
        "Registration settings remain below the complete input context", host
    )
    following.setWordWrap(True)
    layout.addWidget(card)
    layout.addWidget(following)
    constrain_layout_minimum_height(layout)
    host.resize(400, 200)
    host.show()
    qtbot.waitUntil(lambda: card.isVisible() and card.width() > 0)

    def wait_for_complete_geometry(expected_width):
        host.resize(expected_width, 200)
        previous_geometry = None
        stable_observations = 0
        last_observation = None

        def complete_geometry():
            nonlocal last_observation
            # A Resize can expose the new width before the owned zero-turn
            # ancestor commit has reserved its newly wrapped content height.
            # Merely sitting below the old card height is not a settled state.
            guard = host._vipp_minimum_height_constraint
            violations = []
            checks = {
                "observer_timer_pending": any(
                    owner._vipp_minimum_height_constraint._geometry_sync_timer.isActive()
                    for owner in (host, card, *card.rows)
                ),
                "font_refresh_pending": card._font_refresh_timer.isActive(),
                "host_width": host.width() != expected_width,
                "card_width": card.width() != expected_width,
                "card_hfw": card.height() < card.heightForWidth(card.width()),
                "following_position": (
                    following.y() < card.geometry().bottom() + SPACE_SM
                ),
                "owner_hfw_reservation": (
                    host.minimumHeight() < layout.totalHeightForWidth(expected_width)
                ),
            }
            violations.extend(name for name, failed in checks.items() if failed)
            geometry = [
                host.geometry().getRect(),
                card.geometry().getRect(),
                following.geometry().getRect(),
                host.minimumHeight(),
                card.heightForWidth(card.width()),
            ]
            row_observations = []
            for index, row in enumerate(card.rows):
                if (
                    row.geometry().right() >= card.width()
                    or row.geometry().bottom() >= card.height()
                ):
                    violations.append(f"row_{index}_containment")
                geometry.append(row.geometry().getRect())
                labels = []
                for label in (
                    row.role_label,
                    row.source_label,
                    row.scientific_label,
                ):
                    if (
                        label.geometry().right() >= row.width()
                        or label.height() < label.heightForWidth(label.width())
                    ):
                        violations.append(f"row_{index}_{label.objectName()}_hfw")
                    geometry.append(label.geometry().getRect())
                    labels.append(
                        {
                            "name": label.objectName(),
                            "rect": label.geometry().getRect(),
                            "hfw": label.heightForWidth(label.width()),
                            "font": (
                                label.font().family(),
                                label.font().pointSizeF(),
                                label.font().weight(),
                            ),
                        }
                    )
                row_observations.append(
                    {
                        "rect": row.geometry().getRect(),
                        "hfw": row.heightForWidth(row.width()),
                        "labels": labels,
                    }
                )
            last_observation = {
                "expected_width": expected_width,
                "host_rect": host.geometry().getRect(),
                "host_font": (host.font().family(), host.font().pointSizeF()),
                "owner_minimum_height": host.minimumHeight(),
                "owner_hfw": layout.totalHeightForWidth(expected_width),
                "observer_reserved_height": guard._reserved_height,
                "observer_timer_pending": guard._geometry_sync_timer.isActive(),
                "card_rect": card.geometry().getRect(),
                "card_hfw": card.heightForWidth(card.width()),
                "card_layout_hfw": card.layout().totalHeightForWidth(card.width()),
                "card_minimum_height": card.minimumHeight(),
                "card_size_hint": (card.sizeHint().width(), card.sizeHint().height()),
                "owner_card_item": {
                    "minimum": (
                        layout.itemAt(0).minimumSize().width(),
                        layout.itemAt(0).minimumSize().height(),
                    ),
                    "maximum": (
                        layout.itemAt(0).maximumSize().width(),
                        layout.itemAt(0).maximumSize().height(),
                    ),
                    "size_hint": (
                        layout.itemAt(0).sizeHint().width(),
                        layout.itemAt(0).sizeHint().height(),
                    ),
                    "hfw": layout.itemAt(0).heightForWidth(expected_width),
                },
                "following_rect": following.geometry().getRect(),
                "violations": violations,
                "rows": row_observations,
            }
            if violations:
                return None
            return tuple(geometry)

        def geometry_is_settled():
            nonlocal previous_geometry, stable_observations
            current = complete_geometry()
            stable_observations = (
                stable_observations + 1
                if current is not None and current == previous_geometry
                else 1
                if current is not None
                else 0
            )
            previous_geometry = current
            return stable_observations >= 3

        try:
            qtbot.waitUntil(geometry_is_settled, timeout=2000)
        except QtWaitTimeoutError as error:
            raise AssertionError(
                f"Connected-input geometry did not settle: {last_observation!r}"
            ) from error

    for width in (400, 260, 400):
        wait_for_complete_geometry(width)
        assert host.width() == width
        assert card.minimumWidth() == 0
        assert card.height() >= card.heightForWidth(card.width())
        for row in card.rows:
            assert row.geometry().right() < card.width()
            for label in (row.role_label, row.source_label, row.scientific_label):
                assert label.geometry().right() < row.width()
                assert label.height() >= label.heightForWidth(label.width())
        assert following.y() >= card.geometry().bottom() + SPACE_SM
        assert [row.binding for row in card.rows] == _bindings()


@pytest.mark.parametrize("portable", [False, True], ids=["native", "portable"])
def test_input_card_height_returns_after_font_binding_and_visibility_changes(
    qtbot, monkeypatch, portable
):
    _constraint_path(monkeypatch, portable)
    host, card, following = _shown_card(qtbot, 9)
    _wait_for_card_layout(qtbot, host, card, 260)
    original_height = card.minimumHeight()
    card._test_theme_refreshes = []
    refresh_theme = card.refresh_theme

    def record_refresh(*args, **kwargs):
        card._test_theme_refreshes.append(
            (host.font().pointSizeF(), body_point_size(card))
        )
        return refresh_theme(*args, **kwargs)

    monkeypatch.setattr(card, "refresh_theme", record_refresh)
    font = QFont(host.font())
    font.setPointSizeF(16)
    host.setFont(font)
    _wait_for_card_layout(qtbot, host, card, 260)
    assert card.minimumHeight() > original_height
    font.setPointSizeF(9)
    host.setFont(font)
    _wait_for_card_layout(qtbot, host, card, 260)
    assert card.minimumHeight() == original_height
    short = [
        replace(binding, source_title=title, scientific_summary="YX: 4 × 5 · uint8")
        for binding, title in zip(_bindings(), ("Moving", "Reference"), strict=True)
    ]
    card.set_bindings(short)
    _wait_for_card_layout(qtbot, host, card, 260)
    short_height = card.minimumHeight()
    short_owner_height = host.minimumHeight()
    assert short_height < original_height
    card.set_bindings(short[::-1])
    _wait_for_card_layout(qtbot, host, card, 260)
    assert [row.binding for row in card.rows] == short[::-1]
    card.set_bindings([])

    def empty_settled():
        assert card.isHidden() and not card.rows and card.form_layout.rowCount() == 0
        assert not host._vipp_minimum_height_constraint._geometry_sync_timer.isActive()
        assert not hasattr(card, "_vipp_minimum_height_constraint")
        assert not hasattr(card.layout(), "_vipp_minimum_height_state")
        assert not card._font_refresh_timer.isActive()
        assert host.minimumHeight() < short_owner_height
        assert following.y() == 0

    qtbot.waitUntil(empty_settled, timeout=2000)
    card.set_bindings(short)
    _wait_for_card_layout(qtbot, host, card, 260)
    assert card.minimumHeight() == short_height
    assert [row.binding for row in card.rows] == short
    assert following.y() >= card.geometry().bottom() + SPACE_SM


@pytest.mark.parametrize("portable", [False, True], ids=["native", "portable"])
def test_empty_card_retires_pending_geometry_and_rearms_only_with_bindings(
    qtbot, monkeypatch, portable
):
    _constraint_path(monkeypatch, portable)
    host = _host(qtbot, 16)
    layout = QVBoxLayout(host)
    layout.setContentsMargins(0, 0, 0, 0)
    card = ConnectedInputsCard(host)
    layout.addWidget(card)
    following = QLabel("The ordinary node form follows", host)
    following.setWordWrap(True)
    layout.addWidget(following)
    original_constraint = card.layout().sizeConstraint()
    original_minimum = card.minimumSize()
    assert not hasattr(card, "_vipp_minimum_height_constraint")
    assert not hasattr(card.layout(), "_vipp_minimum_height_state")
    card.set_bindings(_bindings())
    constrain_layout_minimum_height(layout)
    host.resize(260, 200)
    host.show()
    _wait_for_card_layout(qtbot, host, card, 260)
    guard = card._vipp_minimum_height_constraint
    timer = guard._geometry_sync_timer
    fired = []
    timer.timeout.connect(lambda: fired.append(True))
    timer.start(0)
    assert timer.isActive()

    ancestor_commits = []
    sync = inspector_module.sync_layout_ancestor_geometry

    def record_commit(origin):
        if origin is card and not card.rows:
            ancestor_commits.append(True)
        return sync(origin)

    monkeypatch.setattr(
        inspector_module, "sync_layout_ancestor_geometry", record_commit
    )
    card.set_bindings([])
    assert not timer.isActive()
    assert card.layout().sizeConstraint() == original_constraint
    assert card.minimumSize() == original_minimum
    assert not hasattr(card, "_vipp_minimum_height_constraint")
    assert not hasattr(card.layout(), "_vipp_minimum_height_state")
    # Repeated empty presentation must neither rearm nor invent a reservation.
    card.set_bindings([])
    assert not hasattr(card, "_vipp_minimum_height_constraint")
    turns = []
    turn_timer = QTimer(host)
    turn_timer.setSingleShot(True)

    def tick():
        turns.append(True)
        if len(turns) < 3:
            turn_timer.start(0)

    turn_timer.timeout.connect(tick)
    turn_timer.start(0)
    qtbot.waitUntil(lambda: len(turns) == 3)
    assert fired == []
    assert ancestor_commits == []
    assert not hasattr(card, "_vipp_minimum_height_constraint")
    assert not hasattr(card.layout(), "_vipp_minimum_height_state")
    card.set_bindings(_bindings())
    assert card._vipp_minimum_height_constraint is not guard
    _wait_for_card_layout(qtbot, host, card, 260)
    assert [row.binding for row in card.rows] == _bindings()
    assert following.y() >= card.geometry().bottom() + layout.spacing()


@pytest.mark.parametrize("portable", [False, True], ids=["native", "portable"])
def test_font_observer_rebinds_coalesces_and_stops_with_its_owned_card(
    qtbot, monkeypatch, portable
):
    _constraint_path(monkeypatch, portable)
    previous_host, card, _following = _shown_card(qtbot, 9)
    _wait_for_card_layout(qtbot, previous_host, card, 260)
    host = _host(qtbot, 16)
    layout = QVBoxLayout(host)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(card)
    constrain_layout_minimum_height(layout)
    host.resize(260, 200)
    host.show()
    previous_host.deleteLater()
    QCoreApplication.sendPostedEvents(previous_host, QEvent.DeferredDelete)
    assert not isalive(previous_host)
    _wait_for_card_layout(qtbot, host, card, 260)
    assert card._observed_font_host is host

    calls = []
    refresh = card.refresh_theme

    def record_refresh(*args, **kwargs):
        calls.append(host.font().pointSizeF())
        return refresh(*args, **kwargs)

    monkeypatch.setattr(card, "refresh_theme", record_refresh)
    font = QFont(host.font())
    for points in (9, 16, 9, 16):
        font.setPointSizeF(points)
        host.setFont(font)
    _wait_for_card_layout(qtbot, host, card, 260)
    assert calls == [16]
    card.set_bindings([replace(binding) for binding in _bindings()])
    turns = []
    timer = QTimer(host)
    timer.setSingleShot(True)

    def tick():
        turns.append(True)
        if len(turns) < 3:
            timer.start(0)

    timer.timeout.connect(tick)
    timer.start(0)
    qtbot.waitUntil(lambda: len(turns) == 3)
    assert calls == [16]
    assert not card._font_refresh_timer.isActive()
    assert all(
        not owner._vipp_minimum_height_constraint._geometry_sync_timer.isActive()
        for owner in (host, card, *card.rows)
    )
    fired = []
    card._font_refresh_timer.timeout.connect(lambda: fired.append(True))
    card._font_refresh_timer.start(0)
    assert card._font_refresh_timer.isActive()
    card.deleteLater()
    QCoreApplication.sendPostedEvents(card, QEvent.DeferredDelete)
    assert not isalive(card)
    turns.clear()
    timer.start(0)
    qtbot.waitUntil(lambda: len(turns) == 3)
    assert fired == []


@pytest.mark.parametrize("portable", [False, True], ids=["native", "portable"])
def test_replacing_rows_cancels_only_their_owned_pending_geometry_timers(
    qtbot, monkeypatch, portable
):
    _constraint_path(monkeypatch, portable)
    host, card, following = _shown_card(qtbot, 16)
    _wait_for_card_layout(qtbot, host, card, 260)
    previous_rows = tuple(card.rows)
    fired = []
    for index, row in enumerate(previous_rows):
        timer = row._vipp_minimum_height_constraint._geometry_sync_timer
        timer.timeout.connect(lambda index=index: fired.append(index))
        timer.start(0)
        assert timer.isActive()  # Prove deletion happens with pending callbacks.
    replacement = replace(_bindings()[0], scientific_summary="YX: 4 × 5 · uint8")
    card.set_bindings([replacement])
    qtbot.waitUntil(lambda: all(not isalive(row) for row in previous_rows))
    _wait_for_card_layout(qtbot, host, card, 260)
    assert fired == []
    assert [row.binding for row in card.rows] == [replacement]
    assert following.y() >= card.geometry().bottom() + SPACE_SM
