"""Measure an installed or source VIPP launch and retain its startup evidence.

Run this with any Python; ``--python`` selects the VIPP environment to measure.
For comparable results, run one launch at a time and distinguish the first
launch after boot from subsequent launches. This tool does not clear caches.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--source-root", type=Path, help="Optional source directory.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--profile", choices=("auto", "cpu", "prefer_gpu"), default="auto"
    )
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--importtime", action="store_true")
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # An exclusive log prevents accidentally overwriting earlier cold evidence.
    log_path = output / "startup.log"
    with log_path.open("x", encoding="utf-8") as log:
        environment = os.environ.copy()
        if args.source_root:
            environment["PYTHONPATH"] = os.pathsep.join(
                filter(
                    None,
                    (str(args.source_root.resolve()), environment.get("PYTHONPATH")),
                )
            )
        environment["NAPARI_VIPP_PIPELINE_TIMING_HISTORY_PATH"] = str(
            output / "isolated-timing-history.json"
        )
        with tempfile.TemporaryDirectory(prefix="vipp-startup-profile-") as channel_dir:
            channel = Path(channel_dir) / "status.jsonl"
            token = secrets.token_urlsafe(32)
            command = [str(args.python.resolve())]
            if args.importtime:
                command += ["-X", "importtime"]
            command += [
                "-m",
                "napari_vipp.app",
                "--profile",
                args.profile,
                "--startup-channel",
                str(channel),
                f"--startup-token={token}",
                "--desktop",
                "--smoke-exit-after-ready",
            ]
            print(
                "Measuring VIPP; the test window closes after readiness.\n"
                f"Log: {log_path}"
            )
            started = time.perf_counter()
            timed_out = False
            with subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                shell=False,
            ) as child:
                try:
                    returncode = child.wait(timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    if sys.platform == "win32":
                        # A Windows venv executable can forward to a base-Python
                        # child. Stop only this diagnostic launch's process tree.
                        subprocess.run(
                            ["taskkill", "/PID", str(child.pid), "/T", "/F"],
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=False,
                            creationflags=subprocess.CREATE_NO_WINDOW,
                        )
                    else:
                        child.kill()
                    returncode = child.wait()
            total = time.perf_counter() - started
            events = []
            if channel.exists():
                for line in channel.read_text(encoding="utf-8").splitlines():
                    event = json.loads(line)
                    if event.pop("token", None) != token:
                        raise ValueError("Unauthenticated startup event.")
                    events.append(event)
    stages = []
    for current, following in zip(events, events[1:], strict=False):
        if current["kind"] == "progress":
            stages.append(
                {
                    "stage": current["stage"],
                    "seconds": following["timestamp"] - current["timestamp"],
                }
            )
    clock = "protocol wall-clock timestamps"
    timings = [
        json.loads(line.removeprefix("VIPP startup timing: "))
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.startswith("VIPP startup timing: ")
    ]
    if timings:
        stages = [
            {"stage": item["previous_stage"], "seconds": item["previous_stage_seconds"]}
            for item in timings
            if item["previous_stage"] is not None
        ]
        clock = "child monotonic clock"
    result = {
        "python": str(args.python.resolve()),
        "source_root": str(args.source_root.resolve()) if args.source_root else None,
        "profile": args.profile,
        "importtime_enabled": args.importtime,
        "cache_state": "uncontrolled; record cold/warm conditions separately",
        "total_process_seconds": total,
        "returncode": returncode,
        "timed_out": timed_out,
        "ready": any(event["kind"] == "ready" for event in events),
        "stages": stages,
        "stage_clock": clock,
        "timing_records": timings,
        "events": events,
    }
    (output / "startup.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for stage in stages:
        print(f"{stage['seconds']:8.3f} s  {stage['stage']}")
    print(f"{total:8.3f} s  total process (includes normal Qt shutdown)")
    return 0 if result["ready"] and returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
