"""Human-launched isolated source binding for VIPP's optional MCP/stdio host.

Only the trusted local command line names a configuration/source. MCP tools
cannot change that binding. This first external entry is a new detached CPU
session, not attachment to an open VIPP window, and does not publish any output.
"""

from __future__ import annotations

import argparse
import math
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from napari_vipp.core.compute import ComputeMode, ComputeRequest
from napari_vipp.core.file_sources import load_frozen_file_source_snapshot
from napari_vipp.core.metadata import AxisMetadata, image_state_from_array
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.source_identity import (
    capture_local_source_identity,
    verify_local_source_identity,
)
from napari_vipp.core.workflow_agent_mcp import (
    WorkflowAgentMCPHost,
    _decode,
    serve_workflow_agent_mcp,
)
from napari_vipp.core.workflow_assessment import MAX_SNAPSHOT_BYTES
from napari_vipp.core.workflow_investigation import (
    InvestigationLimits,
    InvestigationSession,
)

_CONFIG_KIND = "napari-vipp-agent-stdio-config"
_CONFIG_MAX_BYTES = 16 * 1024
_DEFAULT_LIMITS = {
    "max_attempts": 3,
    "max_actions": 32,
    "max_model_calls": 40,
    "max_seconds": 1800,
    "max_image_bytes": 10 * 1024 * 1024,
}


@dataclass(frozen=True, slots=True)
class OwnedMCPSourceSession:
    """Private owner lease; model-visible output never contains these paths."""

    session: InvestigationSession
    _cancelled: threading.Event

    def close(self):
        self._cancelled.set()


def _local_path(raw, *, suffix):
    if (
        type(raw) is not str
        or not raw
        or "\x00" in raw
        or "://" in raw
        or raw.startswith(("\\\\", "//"))
    ):
        raise ValueError("An explicit local absolute file path is required.")
    path = Path(raw)
    if not path.is_absolute() or path.suffix.lower() != suffix:
        raise ValueError("The local absolute file has an unsupported suffix.")
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError("The configured local path must be a regular file.")
    return path


def _calibration(value):
    if (
        type(value) is not dict
        or set(value) != {"axes", "scale", "unit", "origin"}
        or value["axes"] != "ZYX"
        or type(value["unit"]) is not str
        or value["unit"] not in {"nm", "um", "mm", "m", "pixel"}
    ):
        raise ValueError("Explicit scalar ZYX calibration is required.")
    for key in ("scale", "origin"):
        if type(value[key]) is not list or len(value[key]) != 3:
            raise ValueError("Calibration must declare each Z, Y and X value.")
        if any(
            type(number) not in {int, float} or not math.isfinite(number)
            for number in value[key]
        ):
            raise ValueError("Calibration values must be finite real numbers.")
    if any(number <= 0 for number in value["scale"]):
        raise ValueError("Each spatial scale must be positive.")
    return tuple(
        AxisMetadata(
            axis.lower(),
            "space",
            value["unit"],
            scale=scale,
            translation=origin,
        )
        for axis, scale, origin in zip(
            "ZYX", value["scale"], value["origin"], strict=True
        )
    )


def _preflight_npy(path):
    # A mapped numeric header supplies shape/dtype without allocating the
    # declared full volume. Pickled/object sources are never accepted.
    mapped = np.load(path, allow_pickle=False, mmap_mode="r", max_header_size=16384)
    try:
        if (
            type(mapped) is not np.memmap
            or mapped.ndim != 3
            or any(size <= 0 for size in mapped.shape)
            or mapped.dtype.kind not in "iuf"
            or mapped.dtype.hasobject
            or mapped.nbytes > MAX_SNAPSHOT_BYTES
        ):
            raise ValueError("MCP requires a bounded scalar numeric ZYX NPY source.")
        return tuple(mapped.shape), mapped.dtype.str, mapped.nbytes
    finally:
        # Release this header-only map before the verified ordinary loader.
        mapping = getattr(mapped, "_mmap", None)
        if mapping is not None:
            mapping.close()


def create_source_bound_mcp_session(config_path: str | Path) -> OwnedMCPSourceSession:
    """Freeze an explicitly human-configured NPY via ordinary source services.

    This callable is application/CLI authority, never an exposed MCP tool. The
    exact source identity is checked again by the session before each action
    and evidence delivery. Axes/calibration are authored, never guessed.
    """
    config_file = _local_path(str(config_path), suffix=".json")
    with config_file.open("rb") as stream:
        raw = stream.read(_CONFIG_MAX_BYTES + 1)
    if len(raw) > _CONFIG_MAX_BYTES:
        raise ValueError("MCP launch configuration exceeds its bounded size.")
    value = _decode(raw)
    if (
        type(value) is not dict
        or set(value)
        != {
            "kind",
            "version",
            "source_path",
            "goal",
            "calibration",
            "include_input_previews",
            "include_output_previews",
            "limits",
        }
        or value["kind"] != _CONFIG_KIND
        or type(value["version"]) is not int
        or value["version"] != 1
        or type(value["goal"]) is not str
        or not value["goal"].strip()
        or len(value["goal"]) > 4000
        or type(value["include_input_previews"]) is not bool
        or type(value["include_output_previews"]) is not bool
        or type(value["limits"]) is not dict
        or value["limits"].keys() - _DEFAULT_LIMITS.keys()
    ):
        raise ValueError("Invalid explicit source-bound MCP launch configuration.")
    axes = _calibration(value["calibration"])
    limits = InvestigationLimits(**(_DEFAULT_LIMITS | value["limits"]))
    source_path = _local_path(value["source_path"], suffix=".npy")
    if source_path.stat().st_size > MAX_SNAPSHOT_BYTES + 16384:
        raise ValueError("MCP source exceeds the unchanged snapshot boundary.")
    identity = capture_local_source_identity(source_path)
    if identity.size_bytes > MAX_SNAPSHOT_BYTES + 16384:
        raise ValueError("MCP source changed beyond its snapshot boundary.")
    shape, dtype, byte_count = _preflight_npy(source_path)
    cancelled = threading.Event()
    frozen = load_frozen_file_source_snapshot(
        source_path,
        axis_declaration="ZYX -> ZYX",
        expected_identity=identity,
        cancel_callback=cancelled.is_set,
    )
    data = frozen.payload.data
    if (
        tuple(data.shape) != shape
        or data.dtype.str != dtype
        or data.nbytes != byte_count
    ):
        raise ValueError("The source header changed before its frozen capture.")
    state = image_state_from_array(
        data,
        axes=axes,
        source_name="",
        metadata_source="Explicit MCP owner ZYX calibration",
        history=("Spatial calibration explicitly declared by the MCP session owner.",),
    )
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    pipeline.outputs["input"] = data
    pipeline.output_states["input"] = state

    def assert_current():
        if cancelled.is_set():
            raise ValueError("The isolated MCP owner lease is closed.")
        verify_local_source_identity(source_path, frozen.identity)

    session = InvestigationSession(
        pipeline,
        "input",
        value["goal"],
        compute_request=ComputeRequest(mode=ComputeMode.CPU),
        limits=limits,
        include_input_images=value["include_input_previews"],
        include_output_images=value["include_output_previews"],
        require_experiment_plan=True,
        # External Finish is only a draft nomination, never Add/Run/acceptance.
        review_before_finish=False,
        assert_current=assert_current,
        cancelled=cancelled.is_set,
    )
    return OwnedMCPSourceSession(session, cancelled)


def main(argv=None) -> int:
    """Start only on explicit human opt-in; stdout is reserved for MCP frames."""
    parser = argparse.ArgumentParser(
        description=(
            "Optional local source-bound VIPP MCP/stdio host. Use --config for "
            "an isolated CPU investigation, or --live for one explicitly "
            "enabled live VIPP assistant connection."
        )
    )
    parser.add_argument("--enable-mcp", action="store_true")
    binding = parser.add_mutually_exclusive_group(required=True)
    binding.add_argument("--config", type=Path)
    binding.add_argument("--live", type=Path)
    args = parser.parse_args(argv)
    if not args.enable_mcp:
        parser.error("--enable-mcp and exactly one explicit binding are required.")
    try:
        if args.live is not None:
            from napari_vipp.core.workflow_agent_live import (
                relay_live_workflow_agent_mcp,
            )

            relay_live_workflow_agent_mcp(
                args.live, sys.stdin.buffer, sys.stdout.buffer
            )
            return 0
        from napari_vipp.core.workflow_agent_tools import WorkflowAgentTools

        owned = create_source_bound_mcp_session(args.config)
        host = WorkflowAgentMCPHost(
            WorkflowAgentTools(owned.session), enabled=True, on_close=owned.close
        )
        serve_workflow_agent_mcp(host, sys.stdin.buffer, sys.stdout.buffer)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception:
        # The caller chose the private path; still do not leak it into an AI
        # client's captured protocol/log stream through arbitrary exceptions.
        print(
            "VIPP MCP could not start or continue. Check the local launch "
            "configuration, source binding, permissions and limits.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
