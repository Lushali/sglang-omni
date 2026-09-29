# SPDX-License-Identifier: Apache-2.0
"""Compare the saved audio of the parity arms written by run.sh.

For each pair of arms and each sampling mode it reports the share of prompts
whose whole PCM is identical, and how many prompts have a first-second
correlation below 0.9. Same-backend pairs are the restart controls: their
identity sets the floor a cross-backend pair is read against, since
restarting the same server already changes some utterances.
"""

from __future__ import annotations

import argparse
import itertools
import json
import wave
from pathlib import Path

import numpy as np

FIRST_SECOND_CORR_FLOOR = 0.9


def load_pcm(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    return np.frombuffer(frames, dtype=np.int16), rate


def first_second_corr(a: np.ndarray, b: np.ndarray, rate: int) -> float:
    n = min(len(a), len(b), rate)
    x = a[:n].astype(np.float64)
    y = b[:n].astype(np.float64)
    if np.array_equal(x, y):
        return 1.0
    return float(np.corrcoef(x, y)[0, 1])


def compare(left: Path, right: Path) -> dict:
    names = sorted(p.name for p in (left / "audio").glob("*.wav"))
    missing = sorted({p.name for p in (right / "audio").glob("*.wav")} ^ set(names))
    identical = 0
    low_corr = []
    for name in names:
        a, rate = load_pcm(left / "audio" / name)
        b, _ = load_pcm(right / "audio" / name)
        identical += int(np.array_equal(a, b))
        corr = first_second_corr(a, b, rate)
        if corr < FIRST_SECOND_CORR_FLOOR:
            low_corr.append((name, round(corr, 3)))
    return {
        "prompts": len(names),
        "missing": missing,
        "identical": identical,
        "identical_pct": round(100 * identical / len(names), 1),
        "first_second_corr_below_floor": low_corr,
    }


def first_playable_p50(arm_mode: Path) -> float:
    summary = json.loads((arm_mode / "speed_results.json").read_text())["summary"]
    return summary["audio_ttfp_from_arrival_median_s"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    arms = sorted(p.name for p in args.out.iterdir() if (p / "greedy").is_dir())
    report: dict = {"arms": {}, "pairs": {}}
    for arm in arms:
        report["arms"][arm] = {
            mode: {"first_playable_p50_s": first_playable_p50(args.out / arm / mode)}
            for mode in ("greedy", "seeded")
        }
    for left, right in itertools.combinations(arms, 2):
        report["pairs"][f"{left} vs {right}"] = {
            mode: compare(args.out / left / mode, args.out / right / mode)
            for mode in ("greedy", "seeded")
        }
    (args.out / "parity.json").write_text(json.dumps(report, indent=2))
    print("| pair | greedy identical | seeded identical | first-second corr < 0.9 |")
    print("|---|---|---|---|")
    for pair, modes in report["pairs"].items():
        low = sum(len(m["first_second_corr_below_floor"]) for m in modes.values())
        print(
            f"| {pair} | {modes['greedy']['identical_pct']}% "
            f"| {modes['seeded']['identical_pct']}% | {low} |"
        )
    print()
    for arm, modes in report["arms"].items():
        print(
            f"{arm}: first playable p50 greedy "
            f"{modes['greedy']['first_playable_p50_s'] * 1000:.1f} ms, seeded "
            f"{modes['seeded']['first_playable_p50_s'] * 1000:.1f} ms"
        )


if __name__ == "__main__":
    main()
