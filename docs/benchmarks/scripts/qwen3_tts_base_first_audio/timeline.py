# SPDX-License-Identifier: Apache-2.0
"""Median first-audio timeline per arm, over the timed requests only.

The benchmark client sends `--warmup` requests (default: max concurrency) in
one burst before the timed arrivals, and request events record them too. They
are the first admissions of each events run, so the first `--warmup` requests
by admission time are dropped before taking medians.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

MILESTONES = (
    ("coordinator", "request_admission"),
    ("preprocessing", "stage_input_received"),
    ("preprocessing", "stage_complete"),
    ("tts_engine", "scheduler_request_build_start"),
    ("tts_engine", "scheduler_request_build_end"),
    ("tts_engine", "scheduler_queue_enter"),
    ("tts_engine", "scheduler_prefill_start"),
    ("tts_engine", "scheduler_prefill_end"),
    ("tts_engine", "scheduler_first_emit"),
    ("vocoder", "stage_stream_chunk_received"),
    ("vocoder", "stage_first_stream_chunk_sent"),
    ("coordinator", "coordinator_stream_received"),
)
RUNS = ("r1-plain", "r1-events-cold", "r1-events-hot", "r20-plain")


def load(events_dir: Path, warmup: int) -> list[dict[tuple[str, str], int]]:
    first: dict[str, dict[tuple[str, str], int]] = defaultdict(dict)
    for path in sorted(events_dir.rglob("*.jsonl")):
        for line in path.read_text().splitlines():
            event = json.loads(line)
            key = (event["stage"], event["event_name"])
            seen = first[event["request_id"]]
            if key not in seen or event["timestamp_ns"] < seen[key]:
                seen[key] = event["timestamp_ns"]
    admitted = sorted(
        (seen for seen in first.values() if MILESTONES[0] in seen),
        key=lambda seen: seen[MILESTONES[0]],
    )
    return admitted[warmup:]


def report(events_dir: Path, warmup: int) -> dict:
    requests = load(events_dir, warmup)
    rows = []
    previous = None
    for milestone in MILESTONES:
        offsets = [
            (s[milestone] - s[MILESTONES[0]]) / 1e6 for s in requests if milestone in s
        ]
        gaps = [
            (s[milestone] - s[previous]) / 1e6
            for s in requests
            if previous is not None and milestone in s and previous in s
        ]
        if offsets:
            rows.append(
                {
                    "stage": milestone[0],
                    "event": milestone[1],
                    "requests": len(offsets),
                    "offset_p50_ms": round(float(np.median(offsets)), 2),
                    "gap_p50_ms": round(float(np.median(gaps)), 2) if gaps else None,
                }
            )
            previous = milestone
        else:
            pass
    return {"requests": len(requests), "milestones": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--warmup", type=int, required=True)
    args = parser.parse_args()
    result = {}
    for arm_dir in sorted(p for p in args.out.iterdir() if p.is_dir()):
        result[arm_dir.name] = {
            events_dir.name: report(events_dir, args.warmup)
            for events_dir in sorted(arm_dir.glob("events-*"))
        }
        for run in RUNS:
            speed = arm_dir / run / "speed_results.json"
            if speed.exists():
                summary = json.loads(speed.read_text())["summary"]
                result[arm_dir.name][run] = {
                    key: summary.get(key)
                    for key in (
                        "completed_requests",
                        "audio_ttfp_from_arrival_median_s",
                        "audio_ttfp_from_arrival_p95_s",
                        "client_slot_waits",
                    )
                }
            else:
                pass
    (args.out / "timeline-timed.json").write_text(json.dumps(result, indent=2))
    for arm, data in result.items():
        print(f"## {arm}")
        for run in RUNS:
            if run in data:
                print(f"  {run}: {data[run]}")
            else:
                pass
        for name, timeline in data.items():
            if not name.startswith("events-"):
                continue
            else:
                pass
            print(f"  ### {name} ({timeline['requests']} timed requests)")
            print("  | stage | event | offset p50 ms | gap p50 ms |")
            for row in timeline["milestones"]:
                print(
                    f"  | {row['stage']} | {row['event']} | {row['offset_p50_ms']} "
                    f"| {row['gap_p50_ms']} |"
                )
        print()


if __name__ == "__main__":
    main()
