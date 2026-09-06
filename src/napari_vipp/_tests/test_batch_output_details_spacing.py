"""Planned-output bullets group title/format without trailing blank lines."""

from dataclasses import replace

import pytest
from qtpy.QtCore import QUrl

from napari_vipp._tests.test_batch_table_theme import _palette
from napari_vipp._tests.test_ui_batch import _actions, _preview_result
from napari_vipp.core.batch import BatchOutputPlan, ExistingFilePolicy
from napari_vipp.ui.batch import CollectionBatchDialog


def _details_plan(tmp_path):
    result = _preview_result(tmp_path, count=1)
    titles = (
        "Composite → RGB",
        "Label Connected Components",
        "Label Connected Components",
        "Combine Channels",
    )
    outputs = tuple(
        BatchOutputPlan(
            node_id=f"node_{index}",
            node_title=title,
            tag="result",
            kind="image",
            format="ome-tiff",
            path=tmp_path / f"sample-output-{index}.ome.tif",
            existing_file_policy=ExistingFilePolicy.ERROR,
        )
        for index, title in enumerate(titles, 1)
    )
    item = replace(result.items[0], outputs=outputs)
    row = replace(
        result.rows[0],
        outputs=[output.path for output in outputs],
        output_statuses=tuple("new" for _ in outputs),
    )
    return replace(result, items=(item,), rows=(row,))


@pytest.mark.parametrize("dark", [True, False])
@pytest.mark.parametrize("width", [300, 500])
def test_planned_output_bullets_are_compact_with_or_without_paths(
    qtbot, tmp_path, dark, width
):
    plan = _details_plan(tmp_path)
    dialog = CollectionBatchDialog(actions=_actions(plan, []))
    qtbot.addWidget(dialog)
    dialog.setPalette(_palette(dark))
    dialog.apply_preview_result(plan, preview_representative=False)
    document = dialog.item_details.document()

    for show_paths in (False, True, False):
        if getattr(dialog, "_show_output_filenames", False) != show_paths:
            dialog._item_detail_link(QUrl("output-names:toggle"))
        document.setTextWidth(width)
        document.documentLayout().documentSize()
        bullets = []
        block = document.begin()
        while block.isValid():
            if block.textList() is not None:
                bullets.append(block)
            block = block.next()
        assert len(bullets) == 4
        for block, output in zip(bullets, plan.items[0].outputs, strict=True):
            assert output.node_title in block.text()
            assert "Image · OME-TIFF" in block.text()
            assert (str(output.path) in block.text()) is show_paths
            assert not block.text().endswith("\u2028")
            assert "\u2028\u2028" not in block.text()
            assert block.blockFormat().bottomMargin() == 6
        layout = document.documentLayout()
        for previous, following in zip(bullets, bullets[1:], strict=False):
            gap = (
                layout.blockBoundingRect(following).top()
                - layout.blockBoundingRect(previous).bottom()
            )
            assert 0 <= gap <= 8
        link = bullets[-1].next()
        assert "file paths" in link.text()
        assert (
            layout.blockBoundingRect(link).top()
            - layout.blockBoundingRect(bullets[-1]).bottom()
        ) <= 10
