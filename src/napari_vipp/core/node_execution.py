"""Qt-free invocation seam for prepared pipeline node calls.

Pipeline preparation and host metadata finalization deliberately remain in
``core.pipeline`` for now.  This module owns only the narrow boundary between a
fully prepared operation call and the implementation that executes it.  The
default executor invokes the existing authoritative CPU callable.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class PreparedNodeCall:
    """One validated operation invocation, independent of its implementation.

    ``inputs`` are ordered by target port.  Existing multi-input CPU functions
    receive one list containing those values, while single-input functions
    receive the sole value directly.  Runtime-only values such as arrays and a
    progress context are intentionally held by reference; the tuple and mapping
    containers themselves are immutable.
    """

    node_id: str
    operation_id: str
    cpu_function: Callable[..., Any] = field(repr=False, compare=False)
    inputs: tuple[Any, ...] = field(repr=False, compare=False)
    input_states: tuple[Any, ...] = field(default=(), repr=False, compare=False)
    kwargs: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    multiple_inputs: bool = False
    output_port_count: int = 1

    __hash__ = None

    def __post_init__(self) -> None:
        node_id = str(self.node_id).strip()
        operation_id = str(self.operation_id).strip()
        if not node_id or not operation_id:
            raise ValueError("node_id and operation_id must not be empty.")
        if not callable(self.cpu_function):
            raise TypeError("cpu_function must be callable.")

        inputs = tuple(self.inputs)
        if not inputs:
            raise ValueError("A prepared operation call requires at least one input.")
        if not self.multiple_inputs and len(inputs) != 1:
            raise ValueError("A single-input call must contain exactly one input.")

        input_states = tuple(self.input_states)
        if input_states and len(input_states) != len(inputs):
            raise ValueError("input_states must describe every prepared input.")
        if (
            isinstance(self.output_port_count, bool)
            or not isinstance(self.output_port_count, int)
            or self.output_port_count < 1
        ):
            raise ValueError("output_port_count must be a positive integer.")
        if not isinstance(self.kwargs, Mapping):
            raise TypeError("kwargs must be a mapping.")
        keyword_arguments: dict[str, Any] = {}
        for name, value in self.kwargs.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Prepared keyword names must be non-empty strings.")
            keyword_arguments[name] = value

        object.__setattr__(self, "node_id", node_id)
        object.__setattr__(self, "operation_id", operation_id)
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "input_states", input_states)
        object.__setattr__(
            self,
            "kwargs",
            MappingProxyType(keyword_arguments),
        )

    def positional_input(self) -> Any:
        """Materialize the established CPU function's positional argument."""
        if self.multiple_inputs:
            return list(self.inputs)
        return self.inputs[0]

    def keyword_arguments(self) -> dict[str, Any]:
        """Return a detached mapping suitable for ``**kwargs`` invocation."""
        return dict(self.kwargs)


class NodeCallExecutor(Protocol):
    """Structural interface implemented by prepared-call executors."""

    def execute(self, call: PreparedNodeCall, /) -> Any:
        """Execute ``call`` and return its raw operation output."""
        ...


@dataclass(frozen=True, slots=True)
class CPUNodeExecutor:
    """Invoke the authoritative CPU callable stored on a prepared call."""

    def execute(self, call: PreparedNodeCall, /) -> Any:
        return call.cpu_function(
            call.positional_input(),
            **call.keyword_arguments(),
        )


DEFAULT_CPU_NODE_EXECUTOR = CPUNodeExecutor()


def execute_prepared_node_call(
    call: PreparedNodeCall,
    executor: NodeCallExecutor | None = None,
) -> Any:
    """Execute a prepared call with ``executor`` or the default CPU path."""
    selected = DEFAULT_CPU_NODE_EXECUTOR if executor is None else executor
    return selected.execute(call)


__all__ = [
    "CPUNodeExecutor",
    "DEFAULT_CPU_NODE_EXECUTOR",
    "NodeCallExecutor",
    "PreparedNodeCall",
    "execute_prepared_node_call",
]
