"""Session-only acknowledgements of plot advice, never scientific recipe state."""

from __future__ import annotations

from dataclasses import dataclass
from weakref import ref

# These controls cannot change which measurements contribute to a plot.
# Log axes are deliberately NOT included: they can exclude nonpositive values.
_PRESENTATION_FIELDS = frozenset(
    {"title", "point_size", "show_grid", "x_tick_interval", "y_tick_interval"}
)


def visible_plot_warnings(result):
    """Match the advice shown elsewhere; the point-unit control has its own help."""
    return (
        tuple(
            warning
            for warning in result.warnings
            if "independent biological sample" not in warning
        )
        if result is not None
        else ()
    )


def plot_notes_are_caution(warnings):
    """Quietly explain summary rows; keep exclusions and other issues prominent."""
    return any(
        not warning.startswith("Each point represents one summary row,")
        for warning in warnings
    )


@dataclass
class _Acknowledgement:
    table_ref: object
    signature: tuple
    dismissed: bool = False


class PlotNoticeState:
    """Remember dismissal only for each plot's current input/analysis revision.

    TableData is immutable. Its identity is a cheap conservative revision token:
    a newly published input is reviewed again, without hashing a large table on
    the UI thread. Weak references neither retain those tables nor confuse a
    recycled object ID with the input that the user acknowledged. Display-only
    recipe changes and a newly drawn PlotData around the same input are ignored.
    """

    def __init__(self):
        self._by_plot = {}
        self._current = None

    @property
    def dismissed(self):
        return bool(self._current is not None and self._current.dismissed)

    def update(self, plot_id, table, recipe_params, warnings):
        self._current = None
        if table is None or not warnings:
            self._by_plot.pop(plot_id, None)
            return False
        # Plot recipe parameters are scalar strings/numbers/bools. A tuple
        # preserves their exact values without serializing any scientific data.
        signature = (
            tuple(
                sorted(
                    (key, value)
                    for key, value in recipe_params.items()
                    if key not in _PRESENTATION_FIELDS
                )
            ),
            tuple(warnings),
        )
        previous = self._by_plot.get(plot_id)
        if (
            previous is None
            or previous.table_ref() is not table
            or previous.signature != signature
        ):
            previous = _Acknowledgement(ref(table), signature)
            self._by_plot[plot_id] = previous
        self._current = previous
        return self.dismissed

    def dismiss(self):
        if self._current is not None:
            self._current.dismissed = True

    def reopen(self):
        if self._current is not None:
            self._current.dismissed = False
