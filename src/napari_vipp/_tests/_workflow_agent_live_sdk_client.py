"""Official SDK client fixture; existing live session, synthetic data only."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path


def board(result):
    assert result.isError is False, result.structuredContent
    payload = result.structuredContent
    context = payload["session_context"]
    images = [row for row in result.content if row.type == "image"]
    assert len(images) == len(context["image_previews"]) > 0
    for image, manifest in zip(images, context["image_previews"], strict=True):
        raw = base64.b64decode(image.data, validate=True)
        assert image.mimeType == "image/png" and len(raw) == manifest["byte_count"]
        assert hashlib.sha256(raw).hexdigest() == manifest["sha256"]
    assert (
        json.loads(next(row.text for row in result.content if row.type == "text"))
        == payload
    )
    return context, len(images)


async def verify(args):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    env = dict(os.environ)
    env["PYTHONPATH"] = str(args.package_source)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    params = StdioServerParameters(
        command=str(args.server_python),
        args=[
            "-m",
            "napari_vipp.core.workflow_agent_mcp_cli",
            "--enable-mcp",
            "--live",
            str(args.live),
        ],
        env=env,
    )
    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as client:
            initialized = await client.initialize()
            listed = await client.list_tools()
            assert not {"shell", "read_file", "eval", "grant_allowance", "resume"} & {
                row.name for row in listed.tools
            }
            context, source_images = board(
                await client.call_tool("vipp_session_context", {})
            )
            session_id = context["session_id"]
            described = await client.call_tool(
                "vipp_describe_node", {"operation_id": "binary_threshold"}
            )
            context, _ = board(described)
            contract = described.structuredContent["tool_result"]["result"]
            parameters = {row["name"]: row["default"] for row in contract["parameters"]}
            parameters["threshold"] = 100
            proposal = {
                "kind": "napari-vipp-workflow-proposal",
                "version": 1,
                "context_id": context["proposal_context_id"],
                "summary": "Synthetic native ROI diagnostic.",
                "assumptions": ["Not a segmentation accuracy verdict."],
                "questions": [],
                "nodes": [
                    {
                        "key": "mask",
                        "operation_id": "binary_threshold",
                        "params": parameters,
                        "reason": "Synthetic test.",
                    }
                ],
                "connections": [
                    {
                        "source": "input",
                        "source_port": 0,
                        "target": "mask",
                        "target_port": 0,
                    }
                ],
            }
            probed = await client.call_tool(
                "vipp_diagnostic_probe",
                {
                    "input": {"kind": "source"},
                    "roi": {"axis_order": "ZYX", "bounds": [[1, 3], [2, 7], [3, 9]]},
                    "proposal": proposal,
                    "output_roles": [{"role": "mask", "node": "mask", "port": 0}],
                },
            )
            context, _ = board(probed)
            probe = probed.structuredContent["tool_result"]["result"]
            assert probe["status"] == "completed" and probe["input_grid"]["shape"] == [
                2,
                5,
                6,
            ]
            assert probe["execution"]["cleanup_succeeded"] is True
            inspected = await client.call_tool(
                "vipp_inspect_probe",
                {
                    "probe_id": probe["probe_id"],
                    "role": "mask",
                    "views": [{"axis": "Z", "indices": [0, 1], "montage": True}],
                },
            )
            context, output_images = board(inspected)
            assert context["session_id"] == session_id
            assert (
                inspected.structuredContent["tool_result"]["result"]["diagnostics"][
                    "element_count"
                ]
                == 60
            )
            report = {
                "kind": "vipp-live-mcp-official-sdk-interop",
                "version": 1,
                "status": "passed",
                "sdk_version": importlib.metadata.version("mcp"),
                "protocol": initialized.protocolVersion,
                "session_id": session_id,
                "same_existing_source_bound_session": True,
                "live_qt_pre_post_owner_gate": True,
                "authenticated_loopback_via_stdio_relay": True,
                "approved_source_png_count": source_images,
                "approved_final_board_png_count": output_images,
                "actual_png_hashes_and_lengths_verified": True,
                "actual_diagnostic_execution": {
                    "status": "completed",
                    "native_input_shape": [2, 5, 6],
                    "inspected_output_elements": 60,
                    "cleanup_succeeded": True,
                },
                "provider_calls": 0,
                "real_images_sent": False,
                "graph_mutations": 0,
                "automatic_client_registration": False,
            }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-python", type=Path, required=True)
    parser.add_argument("--package-source", type=Path, required=True)
    parser.add_argument("--live", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite a prior SDK receipt.")
    asyncio.run(verify(args))


if __name__ == "__main__":
    main()
