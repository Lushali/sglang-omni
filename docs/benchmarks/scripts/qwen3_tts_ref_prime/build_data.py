# SPDX-License-Identifier: Apache-2.0
"""Collect the committed evidence for round 40 from the pulled run dirs."""
import collections
import csv
import json
import shutil
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
for speed in src.rglob("speed_results.json"):
    rel = speed.relative_to(src).parent
    out = dst / rel / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(json.loads(speed.read_text())["summary"], indent=1))
for events_dir in {p.parent for p in src.rglob("events_*.jsonl")}:
    rel = events_dir.relative_to(src)
    first = collections.defaultdict(dict)
    meta = collections.defaultdict(dict)
    for path in sorted(events_dir.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            e = json.loads(line)
            key = f"{e['stage']}.{e['event_name']}"
            if key.endswith("stream_chunk_received") or key.endswith(
                "stream_chunk_sent"
            ):
                key = f"{key}#{sum(1 for k in first[e['request_id']] if k.startswith(key))}"
            seen = first[e["request_id"]]
            if key not in seen or e["timestamp_ns"] < seen[key]:
                seen[key] = e["timestamp_ns"]
            if e["event_name"] in (
                "diag_plan_built",
                "diag_initial_enqueued",
            ) and e.get("metadata"):
                meta[e["request_id"]].update(e["metadata"])
    rows = []
    for rid, seen in first.items():
        t0 = seen.get("coordinator.request_admission")
        base = t0 if t0 is not None else min(seen.values())
        rows.append(
            {
                "request_id": rid,
                "admission_ns": base,
                **{
                    k: round((v - base) / 1e6, 3)
                    for k, v in seen.items()
                    if not k.split("#")[-1].isdigit() or int(k.split("#")[-1]) < 4
                },
                **{f"meta.{k}": v for k, v in meta[rid].items()},
            }
        )
    rows.sort(key=lambda r: r["admission_ns"])
    cols = sorted(
        {c for r in rows for c in r},
        key=lambda c: (c != "request_id", c != "admission_ns", c),
    )
    out = dst / rel / "milestones.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
for name in (
    "ab.md",
    "ab.json",
    "timeline-ab.md",
    "timeline2-timed.md",
    "timeline-timed.json",
    "probe.json",
    "probe.log",
    "unit.log",
    "unit-new.log",
):
    for p in src.rglob(name):
        target = dst / p.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(p, target)
print(sum(f.stat().st_size for f in dst.rglob("*") if f.is_file()), "bytes")
