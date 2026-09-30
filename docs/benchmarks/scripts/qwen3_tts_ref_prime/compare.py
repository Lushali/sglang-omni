# SPDX-License-Identifier: Apache-2.0
"""Output parity and first-audio latency of the arms written by run.sh.

For each pair of arms: the share of prompts whose whole PCM is identical, how
many have a first-second correlation below 0.9, and the worst and median SNR
of one arm's audio against the other's over the prompts that differ. Same-tree
pairs (main-a vs main-b, prime-a vs prime-b) are the restart controls a
cross-tree pair is read against. Latency is the client's first playable from
planned arrival, per run.
"""

from __future__ import annotations

import argparse
import itertools
import json
import wave
from pathlib import Path

import numpy as np

FIRST_SECOND_CORR_FLOOR = 0.9
RUNS = ("seeded", "r1-cold", "r1-hot", "r20")


def load_pcm(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float64), rate


def compare(left: Path, right: Path) -> dict:
    names = sorted(p.name for p in (left / "audio").glob("*.wav"))
    identical = 0
    low_corr = []
    snrs = []
    for name in names:
        a, rate = load_pcm(left / "audio" / name)
        b, _ = load_pcm(right / "audio" / name)
        if np.array_equal(a, b):
            identical += 1
            continue
        else:
            pass
        n = min(len(a), len(b), rate)
        corr = float(np.corrcoef(a[:n], b[:n])[0, 1])
        if corr < FIRST_SECOND_CORR_FLOOR:
            low_corr.append((name, round(corr, 3)))
        else:
            pass
        if len(a) == len(b):
            noise = float(np.sum((a - b) ** 2))
            snrs.append(10 * np.log10(float(np.sum(a**2)) / max(noise, 1e-12)))
        else:
            pass
    return {
        "prompts": len(names),
        "identical": identical,
        "first_second_corr_below_floor": low_corr,
        "same_length_differing": len(snrs),
        "snr_db_min": round(min(snrs), 1) if snrs else None,
        "snr_db_median": round(float(np.median(snrs)), 1) if snrs else None,
    }


def latency(run_dir: Path) -> dict:
    summary = json.loads((run_dir / "speed_results.json").read_text())["summary"]
    return {
        key: summary.get(key)
        for key in (
            "completed_requests",
            "audio_ttfp_from_arrival_median_s",
            "audio_ttfp_from_arrival_p95_s",
            "max_playback_underrun_p95_s",
            "rtf_median",
            "client_slot_waits",
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    arms = sorted(p.name for p in args.out.iterdir() if (p / "seeded").is_dir())
    report: dict = {
        "latency": {
            arm: {run: latency(args.out / arm / run) for run in RUNS} for arm in arms
        },
        "pairs": {
            f"{left} vs {right}": compare(
                args.out / left / "seeded", args.out / right / "seeded"
            )
            for left, right in itertools.combinations(arms, 2)
        },
    }
    (args.out / "ab.json").write_text(json.dumps(report, indent=2))
    print(
        "| pair | identical | corr < 0.9 | differing, same length | SNR dB min / median |"
    )
    print("|---|---|---|---|---|")
    for pair, row in report["pairs"].items():
        print(
            f"| {pair} | {row['identical']}/{row['prompts']} "
            f"| {len(row['first_second_corr_below_floor'])} "
            f"| {row['same_length_differing']} "
            f"| {row['snr_db_min']} / {row['snr_db_median']} |"
        )
    print()
    print("| arm | run | first playable p50 ms | p95 ms | underrun p95 ms | done |")
    print("|---|---|---|---|---|---|")
    for arm, runs in report["latency"].items():
        for run, row in runs.items():
            print(
                f"| {arm} | {run} | {row['audio_ttfp_from_arrival_median_s'] * 1000:.1f} "
                f"| {row['audio_ttfp_from_arrival_p95_s'] * 1000:.1f} "
                f"| {(row['max_playback_underrun_p95_s'] or 0) * 1000:.1f} "
                f"| {row['completed_requests']} |"
            )


if __name__ == "__main__":
    main()
