"""Keep bundled diagrams legible as their calculated cards expand."""

import importlib.util
from pathlib import Path

import pytest
from qtpy.QtCore import QEvent

from napari_vipp.ui.examples import EXAMPLE_WORKFLOWS

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "example_layout_audit", ROOT / "scripts" / "check_example_layouts.py"
)
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)


@pytest.mark.parametrize("phase", ["initial", "ready"])
@pytest.mark.parametrize("example", EXAMPLE_WORKFLOWS, ids=lambda spec: spec.id)
def test_example_cards_notes_and_tunnels_have_clear_layout(qtbot, qapp, example, phase):
    old_font, old_palette = qapp.font(), qapp.palette()
    view = None
    try:
        audit.configure_application()
        view = audit.build_example(ROOT / "examples" / example.filename, phase)
        qtbot.addWidget(view)
        report = audit.layout_diagnostics(view)
        assert not report["collisions"], report["collisions"]
        assert not report["wire_card_intersections"], report["wire_card_intersections"]
        assert not report["wire_tunnel_intersections"], report[
            "wire_tunnel_intersections"
        ]
        assert not report["wire_note_intersections"], report["wire_note_intersections"]
    finally:
        if view is not None:
            view.close()
            view.deleteLater()
            # Offscreen tests have no persistent event loop; release each
            # scene before changing the application font for the next example.
            qapp.sendPostedEvents(None, QEvent.DeferredDelete)
        qapp.setFont(old_font)
        qapp.setPalette(old_palette)
