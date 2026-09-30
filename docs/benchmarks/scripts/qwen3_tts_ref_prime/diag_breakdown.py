# SPDX-License-Identifier: Apache-2.0
"""Per-request timing of the vocoder's first chunk from the diag_* events.

Usage: diag_breakdown.py <arm dir holding events/>. Medians over requests of
each step from the first codec chunk reaching the vocoder to the first PCM
leaving it, as instrumented by diag_patch.py.
"""
import collections
import glob
import json
import statistics
import sys

arm = sys.argv[1]
ev = [
    json.loads(l)
    for f in glob.glob(f"{arm}/events/**/*.jsonl", recursive=True)
    for l in open(f)
]
by = collections.defaultdict(list)
for e in ev:
    by[e["request_id"]].append(e)
cohort = sorted(e["timestamp_ns"] for e in by.get("cohort", []))
steps = collections.defaultdict(list)


def first(es, stage, name):
    t = [
        e["timestamp_ns"] for e in es if e["stage"] == stage and e["event_name"] == name
    ]
    return min(t) if t else None


plans = collections.Counter()
for rid, es in by.items():
    if rid == "cohort":
        continue
    adm = first(es, "coordinator", "request_admission")
    if adm is None:
        continue
    recv = sorted(
        e["timestamp_ns"]
        for e in es
        if e["stage"] == "vocoder" and e["event_name"] == "stage_stream_chunk_received"
    )
    enq = first(es, "vocoder", "diag_initial_enqueued")
    bs = first(es, "vocoder", "diag_batch_start")
    pb = first(es, "vocoder", "diag_plan_built")
    dd = first(es, "vocoder", "diag_decode_done")
    rs = first(es, "vocoder", "diag_resolved")
    cp = first(es, "vocoder", "diag_commit_put")
    fs = first(es, "vocoder", "stage_first_stream_chunk_sent")
    if None in (enq, bs, pb, dd, rs, cp, fs) or not recv:
        continue
    ctx = [t for t in cohort if pb <= t <= dd]
    pm = [e["metadata"] for e in es if e["event_name"] == "diag_plan_built"][0]
    plans[(pm["fresh"], pm["context"], pm["trim"])] += 1
    em = [e["metadata"] for e in es if e["event_name"] == "diag_initial_enqueued"][0]
    steps["recv1->enqueued"].append((enq - recv[0]) / 1e6)
    steps["enqueued->batch_start"].append((bs - enq) / 1e6)
    steps["batch_start->plan_built"].append((pb - bs) / 1e6)
    if ctx:
        steps["plan_built->context_done"].append((ctx[0] - pb) / 1e6)
        steps["context_done->decode_done"].append((dd - ctx[0]) / 1e6)
    steps["plan_built->decode_done"].append((dd - pb) / 1e6)
    steps["decode_done->resolved"].append((rs - dd) / 1e6)
    steps["resolved->commit_put"].append((cp - rs) / 1e6)
    steps["commit_put->first_sent"].append((fs - cp) / 1e6)
    steps["recv1->first_sent"].append((fs - recv[0]) / 1e6)
    steps["enqueued_total_frames"].append(em["total_frames"])
    steps["enqueued_ref_frames"].append(em["ref_frames"])
print(arm, "n=", len(steps["recv1->first_sent"]))
for k, v in steps.items():
    print(
        f"  {k:28s} p50 {statistics.median(v):8.2f}  p90 {sorted(v)[int(len(v)*0.9)]:8.2f}"
    )
print("  plans (fresh, context, trim):", plans.most_common(4))
