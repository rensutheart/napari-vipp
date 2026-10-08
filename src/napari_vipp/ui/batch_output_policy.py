"""Presentation and policy-only updates using already-checked output facts."""

from collections import Counter
from dataclasses import replace
from string import Formatter

from qtpy.QtCore import QSignalBlocker, Qt, Signal
from qtpy.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from napari_vipp.core.batch import (
    BatchItemFilePolicy,
    ExistingFilePolicy,
    apply_batch_item_file_policies,
    batch_item_file_policy_key,
)
from napari_vipp.ui.toolbar_controls import ToolbarCommandButton, toolbar_icon


def output_action(output, config=None) -> str:
    """Do not infer a previous successful run merely from an existing path."""
    if output.duplicate or output.input_collision:
        return "blocked"
    if not output.exists:
        return "create"
    if output.existing_file_policy is ExistingFilePolicy.SKIP:
        return "keep"
    if output.existing_file_policy is ExistingFilePolicy.OVERWRITE:
        return "overwrite"
    if config is not None and any(
        entry.node_id == output.node_id and entry.overwrite == "no"
        for entry in config.outputs
    ):
        return "blocked"
    return "ask"


def output_problem_kind(output, config=None) -> str | None:
    """Name the primary blocking reason using the same facts as output_action."""
    if output.duplicate:
        return "duplicate"
    if output.input_collision:
        return "input_overlap"
    if output_action(output, config) == "blocked":
        return "protected"
    return None


def output_problem(output, config=None) -> str:
    """Explain a blocked destination and its remedy without changing its policy."""
    kind = output_problem_kind(output, config)
    if kind == "duplicate":
        message = (
            "Duplicate output path: more than one planned output would save to "
            f"{output.path.name}. Use {{batch_id}}__{{node_id}} in Filename template "
            "so each batch item and output node has its own file."
        )
        if config is not None and len(config.outputs) > 1:
            spec = next(
                (entry for entry in config.outputs if entry.node_id == output.node_id),
                None,
            )
            if _template_uses_tag(spec):
                message += (
                    " If this path is shared with another output node, changing "
                    "this node's Tag or Subfolder can separate those nodes."
                )
            else:
                message += (
                    " If different output nodes share this path, give them "
                    "different Subfolder values."
                )
        message += " Overwrite cannot resolve a shared destination."
        if output.input_collision:
            message += (
                " This path also overlaps an input; choose a destination outside "
                "that input."
            )
        return message
    if kind == "input_overlap":
        return (
            "Output path overlaps an input source. Choose a different output "
            "folder or filename so the source is preserved. An input stored as "
            "a folder also protects files inside that folder. "
            "Overwrite cannot replace an input."
        )
    if kind == "protected":
        return (
            "This file already exists and this Batch Output node has Overwrite = "
            "no. Choose Keep existing outputs for this item, choose a new "
            "destination, or change the node's Overwrite to batch default before "
            "choosing an existing-file policy."
        )
    return ""


def _template_uses_tag(spec) -> bool:
    if spec is None:
        return False
    try:
        return any(
            field == "tag" for _text, field, _format, _conversion
            in Formatter().parse(spec.filename_template)
        )
    except ValueError:
        return False


def _duplicate_remedy(outputs, config) -> str:
    names = _problem_node_names(outputs)
    node_ids = {output.node_id for output in outputs}
    repeated = Counter((output.node_id, output.path) for output in outputs)
    if len(node_ids) == 1 or any(count > 1 for count in repeated.values()):
        return (
            f"Use {{batch_id}} in Filename template on {names} to separate batch "
            "items; {batch_id}__{node_id} also separates output nodes."
        )
    specs = {spec.node_id: spec for spec in config.outputs}
    fields = (
        "Tag or Subfolder" if all(_template_uses_tag(specs.get(node_id))
        for node_id in node_ids) else "Subfolder"
    )
    return (
        f"Give {names} different {fields} values, or use "
        "{batch_id}__{node_id} in Filename template."
    )


def _problem_node_names(outputs) -> str:
    names = dict.fromkeys(
        (
            f"{output.node_title} ({output.node_id})"
            if output.node_title and output.node_title != output.node_id
            else output.node_id
        )
        for output in outputs
    )
    shown = ", ".join(tuple(names)[:2])
    if len(names) > 2:
        shown += f", and {len(names) - 2} more nodes"
    return shown


def output_counts(outputs, config=None):
    return Counter(output_action(output, config) for output in outputs)


def output_counts_text(outputs, config=None, *, compact=False) -> str:
    counts = output_counts(outputs, config)
    return (
        " · ".join(
            f"{counts[action]:,} {label}"
            for action, label in (
                ("create", "to create"),
                ("keep", "existing" if compact else "existing to keep"),
                ("overwrite", "to overwrite"),
                ("ask", "existing" if compact else "need decision"),
                ("blocked", "blocked"),
            )
            if counts[action]
        )
        or "No outputs"
    )


def planned_item_status(outputs, config=None) -> str:
    counts = output_counts(outputs, config)
    if counts["blocked"]:
        return "Blocked"
    if counts["ask"]:
        return "Needs decision"
    if counts["keep"] == len(outputs) and outputs:
        return "Keep existing"
    if counts["keep"]:
        return "Create missing" if not counts["overwrite"] else "Keep + overwrite"
    if counts["overwrite"]:
        return "Will overwrite"
    return "Not run"


def batch_work_counts(preview):
    keep = sum(
        bool(item.outputs)
        and all(
            output_action(output, preview.config) == "keep" for output in item.outputs
        )
        for item in preview.items
    )
    return len(preview.items) - keep, keep


def checked_output_message(preview) -> str:
    outputs = tuple(output for item in preview.items for output in item.outputs)
    counts = output_counts(outputs, preview.config)
    if counts["blocked"]:
        problems = {
            "duplicate": tuple(output for output in outputs if output.duplicate),
            "input_overlap": tuple(
                output for output in outputs if output.input_collision
            ),
            "protected": tuple(
                output
                for output in outputs
                if output_problem_kind(output, preview.config) == "protected"
            ),
        }
        destination_text = (
            "output destination is" if counts["blocked"] == 1
            else "output destinations are"
        )
        messages = [f"{counts['blocked']:,} {destination_text} blocked."]
        if duplicates := problems["duplicate"]:
            shared_text = (
                "output shares a planned file path" if len(duplicates) == 1
                else "outputs share planned file paths"
            )
            messages.append(
                f"{len(duplicates):,} {shared_text}. "
                + _duplicate_remedy(duplicates, preview.config)
            )
        if overlaps := problems["input_overlap"]:
            overlap_text = (
                "output path overlaps an input" if len(overlaps) == 1
                else "output paths overlap inputs"
            )
            messages.append(
                f"{len(overlaps):,} {overlap_text}. Choose a different "
                f"output folder or filename for {_problem_node_names(overlaps)}."
            )
        if protected := problems["protected"]:
            protected_text = (
                "existing output is protected" if len(protected) == 1
                else "existing outputs are protected"
            )
            messages.append(
                f"{len(protected):,} {protected_text} by Overwrite = "
                f"no on {_problem_node_names(protected)}. Keep these outputs, "
                "choose a new destination, or change the node's Overwrite to "
                "batch default."
            )
        messages.append("Recheck after changing output settings.")
        return " ".join(messages)
    if counts["ask"]:
        return (
            f"{counts['ask']:,} existing outputs need a decision. Ask before overwrite "
            "prompts when you press Run, or choose an existing-file policy below."
        )
    return (
        f"Ready: {len(preview.items):,} batch items checked. "
        f"{output_counts_text(outputs, preview.config)}. "
        "Nothing was saved or calculated."
    )


def with_existing_file_policy(preview, policy):
    """Change dispositions only: preserve exact source identities and path checks.

    Explicit Batch Output yes/no settings and genuine collisions are retained.
    Execution still revalidates current disk contents before any publication.
    """
    policy = ExistingFilePolicy(policy)
    config = replace(preview.config, existing_file_policy=policy)
    return with_output_policy_config(preview, config)


def item_file_choice(config, item):
    if not config.item_file_policies:
        return None
    key = batch_item_file_policy_key(config, item)
    return next(
        (entry for entry in config.item_file_policies if entry.item_key == key), None
    )


def item_file_choice_label(config, item):
    choice = item_file_choice(config, item)
    if choice is None:
        return "Use batch default"
    return (
        "Keep existing outputs · item choice"
        if choice.policy is ExistingFilePolicy.SKIP
        else "Rerun and overwrite outputs · item choice"
    )


def with_item_file_policy(preview, position, policy):
    item = preview.items[position]
    key = batch_item_file_policy_key(preview.config, item)
    if key is None:
        raise ValueError(
            "Check this item first to identify its exact sources and outputs."
        )
    choices = {entry.item_key: entry for entry in preview.config.item_file_policies}
    if policy is None:
        choices.pop(key, None)
    else:
        choices[key] = BatchItemFilePolicy(
            key, ExistingFilePolicy(policy), item.batch_id
        )
    return with_output_policy_config(
        preview,
        replace(
            preview.config,
            item_file_policies=tuple(choices.values()),
        ),
    )


def with_output_policy_config(preview, config):
    items = apply_batch_item_file_policies(config, preview.items)
    by_index = {item.index: item for item in items}
    rows = tuple(
        replace(
            row,
            output_statuses=tuple(
                output.status_text for output in by_index[row.batch_index].outputs
            ),
        )
        for row in preview.rows
    )
    collisions = sum(
        output.duplicate
        or output.input_collision
        or (output.exists and output.existing_file_policy is ExistingFilePolicy.ERROR)
        for item in items
        for output in item.outputs
    )
    return replace(
        preview, config=config, items=items, rows=rows, collision_count=collisions
    )


class BatchExistingFilesControls(QWidget):
    """A whole-batch choice in the review page, mirrored to Setup."""

    policyChanged = Signal(str)
    resetItemsRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 6)
        layout.setSpacing(4)
        row = QHBoxLayout()
        title = QLabel("Existing files · batch default")
        title.setStyleSheet("font-weight: bold;")
        title.setWordWrap(False)
        self.policy_combo = QComboBox()
        self.policy_combo.addItem("Ask before overwrite", "error")
        self.policy_combo.addItem("Skip existing", "skip")
        self.policy_combo.addItem("Overwrite without asking", "overwrite")
        self.policy_combo.setAccessibleName("Existing files batch default")
        self.policy_combo.setToolTip(
            "Applies to items using the batch default, not just checked rows. "
            "Right-click a row to choose for that item. Updates output "
            "decisions immediately without rereading source images. Item choices "
            "override this default; protected destinations remain protected. "
            "No files change until Run."
        )
        title.setBuddy(self.policy_combo)
        row.addWidget(title)
        row.addWidget(self.policy_combo)
        self.reset_items_button = ToolbarCommandButton("Reset item choices")
        self.reset_items_button.setToolTip(
            "Remove all per-item file decisions and use the batch default again. "
            "Parameter overrides and node behavior are unchanged. No files are changed."
        )
        self.reset_items_button.clicked.connect(self.resetItemsRequested.emit)
        row.addWidget(self.reset_items_button)
        row.addStretch(1)
        layout.addLayout(row)
        self.detail = QLabel()
        self.detail.setWordWrap(True)
        self.detail.setTextFormat(Qt.PlainText)
        layout.addWidget(self.detail)
        self.policy_combo.currentIndexChanged.connect(
            lambda: self.policyChanged.emit(str(self.policy_combo.currentData()))
        )
        self.hide()

    def set_plan(self, preview, *, enabled=True, item_choices=None, reset_enabled=None):
        if item_choices is None:
            item_choices = preview.config.item_file_policies if preview else ()
        outputs = (
            tuple(output for item in preview.items for output in item.outputs)
            if preview
            else ()
        )
        existing = sum(output.exists for output in outputs)
        self.setVisible(bool(existing or item_choices))
        self.reset_items_button.setVisible(bool(item_choices))
        self.reset_items_button.setEnabled(
            enabled if reset_enabled is None else reset_enabled
        )
        self.reset_items_button.setIcon(toolbar_icon("reset", self.palette()))
        self.policy_combo.setVisible(preview is not None)
        self.policy_combo.setEnabled(enabled)
        if preview is None:
            self.detail.setText(
                f"{len(item_choices)} saved item choices. Check batch to verify their "
                "sources and destinations, or reset these choices."
            )
            return
        with QSignalBlocker(self.policy_combo):
            self.policy_combo.setCurrentIndex(
                self.policy_combo.findData(preview.config.existing_file_policy.value)
            )
        self.detail.setText(
            f"{existing:,} files already exist. "
            + output_counts_text(outputs, preview.config)
            + (
                f". {len(item_choices)} item choices override the batch default"
                if item_choices
                else ""
            )
            + ". Files are not changed until you run."
        )
