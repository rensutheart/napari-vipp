"""Lazy, provider-neutral finalization of device-produced host payloads.

Some accelerator implementations return an internal array payload whose public
VIPP value is a scalar or :class:`~napari_vipp.core.tables.TableData`.  The
payload remains runtime-owned until the ordinary device-to-host transfer and
runtime cleanup have completed.  This module owns the small shared seam which
then resolves and invokes the declared host finalizer.

No accelerator package is imported here.  Callers decide when the finalizer is
safe to run and must pass a :class:`PreparedNodeCall` whose inputs do not retain
device-owned values.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Protocol

from napari_vipp.core.node_execution import PreparedNodeCall


class HostFinalizerLoadError(RuntimeError):
    """Raised when a declared host finalizer cannot be resolved safely."""


class HostFinalizer(Protocol):
    """Convert transferred payload outputs into public operation outputs."""

    def __call__(
        self,
        host_outputs: tuple[object, ...],
        /,
        *,
        call: PreparedNodeCall,
    ) -> object: ...


def resolve_host_finalizer(reference: str) -> HostFinalizer:
    """Resolve one ``module:attribute`` reference only when it is executed."""

    normalized = str(reference).strip()
    module_name, separator, attribute_path = normalized.partition(":")
    if (
        not separator
        or not module_name.strip()
        or not attribute_path.strip()
        or any(not part for part in attribute_path.split("."))
    ):
        raise HostFinalizerLoadError(
            "host_finalizer_ref must use 'module:attribute' syntax."
        )
    try:
        candidate: object = importlib.import_module(module_name)
        for attribute in attribute_path.split("."):
            candidate = getattr(candidate, attribute)
    except Exception as exc:
        raise HostFinalizerLoadError(
            f"Could not load host finalizer {normalized!r}: {type(exc).__name__}: {exc}"
        ) from exc
    if not callable(candidate):
        raise HostFinalizerLoadError(
            f"Declared host finalizer {normalized!r} is not callable."
        )
    return candidate


def normalize_operation_outputs(
    raw: object,
    output_count: int,
) -> tuple[object, ...]:
    """Normalize one operation result against its declared public port count."""

    if isinstance(output_count, bool) or not isinstance(output_count, int):
        raise TypeError("output_count must be an integer.")
    if output_count < 1:
        raise ValueError("output_count must be positive.")
    if output_count == 1:
        return (raw,)
    if not isinstance(raw, (tuple, list)):
        raise TypeError(
            f"An operation with {output_count} outputs must return a tuple or list."
        )
    outputs = tuple(raw)
    if len(outputs) != output_count:
        raise ValueError(
            f"An operation declared {output_count} outputs but returned {len(outputs)}."
        )
    return outputs


def apply_host_finalizer(
    reference: str,
    host_outputs: Sequence[object],
    call: PreparedNodeCall,
) -> tuple[object, ...]:
    """Apply one declared finalizer and normalize its public output tuple.

    ``host_outputs`` are always supplied as a tuple, including for a one-output
    operation.  This keeps the finalizer ABI stable for generic multi-output
    implementations.  The finalizer's return value follows the ordinary VIPP
    convention: a direct value for one output and a tuple/list for many.
    """

    if not isinstance(call, PreparedNodeCall):
        raise TypeError("call must be a PreparedNodeCall.")
    payloads = tuple(host_outputs)
    if len(payloads) != call.output_port_count:
        raise ValueError(
            "Transferred payload count does not match the prepared operation "
            "output count."
        )
    finalizer = resolve_host_finalizer(reference)
    raw = finalizer(payloads, call=call)
    return normalize_operation_outputs(raw, call.output_port_count)


__all__ = [
    "HostFinalizer",
    "HostFinalizerLoadError",
    "apply_host_finalizer",
    "normalize_operation_outputs",
    "resolve_host_finalizer",
]
