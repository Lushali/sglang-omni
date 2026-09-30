# SPDX-License-Identifier: Apache-2.0
"""Can the ICL bootstrap skip the conv stack over most of the reference?

The streaming vocoder's first decode of a voice-clone request feeds every
reference frame and the first generated frame through the whole codec decoder,
then trims the reference audio. The decoder's transformer attends 71 frames
back per layer over 8 layers, so it needs the whole reference; the conv stack
after it looks back only a few frames. This probe primes the state by running
quantizer, pre_conv and transformer over the reference head
(``advance_context``) and the full decoder over its last W frames, then
decodes the same generated frames as the one-shot bootstrap, and reports how
far the generated audio and the final state drift for each W. It also times
both bootstraps. Run it from a checkout that has ``advance_context``.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from functools import partial

import torch

from benchmarks.dataset.seedtts import load_seedtts_samples
from sglang_omni.models.qwen3_tts.incremental_codec import (
    Qwen3TTSIncrementalCodecState,
    Qwen3TTSIncrementalDecoder,
)
from sglang_omni.models.qwen3_tts.incremental_codec_cuda_graph import (
    split_frames_by_width,
)
from sglang_omni.models.qwen3_tts.stages import load_qwen3_tts_tokenizer
from sglang_omni.models.qwen3_tts.streaming_vocoder import (
    DEFAULT_QWEN3_TTS_INCREMENTAL_WINDOW_FRAMES,
)
from sglang_omni.utils.snake_beta import fuse_vocoder_decoder

TAILS = (0, 2, 4, 6, 8, 10, 12, 16, 24, 32)
FOLLOWUP_STRIDE = 4


def conv_receptive_frames(decoder) -> float:
    """Frames the post-transformer conv stack looks back, summed over layers."""
    rate = 1
    frames = 0.0
    for blocks in decoder.upsample:
        transconv = blocks[0]
        stride = int(transconv.conv.stride[0])
        frames += math.ceil(int(transconv.right_pad) / stride) / rate
        rate *= stride
        frames += int(blocks[1].dwconv.padding) / rate
    frames += int(decoder.decoder[0].padding) / rate
    for decoder_block in decoder.decoder[1:-2]:
        transconv = decoder_block.block[1]
        stride = int(transconv.conv.stride[0])
        frames += math.ceil(int(transconv.right_pad) / stride) / rate
        rate *= stride
        for residual in decoder_block.block[2:]:
            frames += (int(residual.conv1.padding) + int(residual.conv2.padding)) / rate
    frames += int(decoder.decoder[-1].padding) / rate
    return frames


def decode_split(incremental, codes, state) -> torch.Tensor:
    """Decode through the captured window widths, as the production bootstrap does."""
    split = split_frames_by_width(
        int(codes.shape[-1]), DEFAULT_QWEN3_TTS_INCREMENTAL_WINDOW_FRAMES
    )
    pieces = []
    offset = 0
    for width in split:
        pieces.append(incremental.decode(codes[..., offset : offset + width], state))
        offset += width
    return torch.cat(pieces, dim=-1)


def bootstrap(incremental, ref, gen, tail, *, split: bool):
    """Return the generated audio and the final state of one bootstrap variant.

    tail=None is the current bootstrap: every reference frame through the full
    decoder. An int primes the reference head through advance_context and runs
    the full decoder over the last `tail` reference frames only.
    """
    spf = int(incremental.total_upsample)
    state = Qwen3TTSIncrementalCodecState()
    decode = partial(decode_split, incremental) if split else incremental.decode
    ref_frames = int(ref.shape[-1])
    kept = ref_frames if tail is None else min(tail, ref_frames)
    if kept < ref_frames:
        incremental.advance_context(ref[..., : ref_frames - kept], state)
    else:
        pass
    first = torch.cat((ref[..., ref_frames - kept :], gen[..., :1]), dim=-1)
    audio = [decode(first, state)[..., kept * spf :]]
    for offset in range(1, int(gen.shape[-1]), FOLLOWUP_STRIDE):
        audio.append(decode(gen[..., offset : offset + FOLLOWUP_STRIDE], state))
    return torch.cat(audio, dim=-1).float(), state


def state_drift(left, right) -> float:
    """Largest absolute difference over every non-empty state buffer."""
    worst = 0.0
    for name in (
        "transformer_keys",
        "transformer_values",
        "conv_histories",
        "transconv_overlaps",
    ):
        a, b = getattr(left, name), getattr(right, name)
        for key in a:
            if a[key].numel():
                worst = max(worst, float((a[key].float() - b[key].float()).abs().max()))
            else:
                pass
    return worst


def audio_drift(left: torch.Tensor, right: torch.Tensor) -> dict:
    diff = left - right
    return {
        "max_abs": float(diff.abs().max()),
        "rel_l2": float(diff.norm() / left.norm().clamp_min(1e-12)),
    }


def time_ms(fn, reps: int) -> float:
    samples = []
    for _ in range(reps + 3):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples[3:])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--clips", type=int, default=8)
    parser.add_argument("--gen-frames", type=int, default=33)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    samples = load_seedtts_samples(
        "zhaochenyang20/seed-tts-eval-arrow", max_samples=args.clips + 1
    )
    report: dict = {"dtypes": {}}
    for dtype in ("float32", "bfloat16"):
        tokenizer = load_qwen3_tts_tokenizer(
            args.model, device="cuda", dtype=dtype, attn_implementation=None
        )
        fuse_vocoder_decoder(tokenizer.model.decoder)
        incremental = Qwen3TTSIncrementalDecoder(tokenizer.model.decoder)
        report["conv_receptive_frames"] = conv_receptive_frames(tokenizer.model.decoder)
        report["conv_lookback_frames"] = incremental.state_spec().conv_lookback_frames
        codes = [
            tokenizer.encode(sample.ref_audio).audio_codes[0].T.unsqueeze(0).cuda()
            for sample in samples
        ]
        clips = []
        with torch.inference_mode():
            for index in range(args.clips):
                ref = codes[index]
                gen = codes[index + 1][..., : args.gen_frames]
                one_shot, one_shot_state = bootstrap(
                    incremental, ref, gen, None, split=False
                )
                windowed, windowed_state = bootstrap(
                    incremental, ref, gen, None, split=True
                )
                row = {
                    "ref_frames": int(ref.shape[-1]),
                    "windowed_vs_one_shot": {
                        **audio_drift(one_shot, windowed),
                        "state_max_abs": state_drift(one_shot_state, windowed_state),
                    },
                    "tails": {},
                }
                for tail in TAILS:
                    primed, primed_state = bootstrap(
                        incremental, ref, gen, tail, split=True
                    )
                    row["tails"][tail] = {
                        **audio_drift(one_shot, primed),
                        "state_max_abs": state_drift(one_shot_state, primed_state),
                    }
                if dtype == "bfloat16":
                    first = torch.cat((ref, gen[..., :1]), dim=-1)
                    row["ms_current_one_shot"] = time_ms(
                        lambda: incremental.decode(
                            first, Qwen3TTSIncrementalCodecState()
                        ),
                        args.reps,
                    )
                    row["ms_current_windowed"] = time_ms(
                        lambda: decode_split(
                            incremental, first, Qwen3TTSIncrementalCodecState()
                        ),
                        args.reps,
                    )
                    for tail in (10, 16):
                        head = ref[..., : int(ref.shape[-1]) - tail]
                        rest = torch.cat(
                            (ref[..., int(ref.shape[-1]) - tail :], gen[..., :1]),
                            dim=-1,
                        )

                        def primed_bootstrap(head=head, rest=rest) -> None:
                            state = Qwen3TTSIncrementalCodecState()
                            incremental.advance_context(head, state)
                            decode_split(incremental, rest, state)

                        def head_only(head=head) -> None:
                            incremental.advance_context(
                                head, Qwen3TTSIncrementalCodecState()
                            )

                        row[f"ms_primed_tail{tail}"] = time_ms(
                            primed_bootstrap, args.reps
                        )
                        row[f"ms_advance_head_tail{tail}"] = time_ms(
                            head_only, args.reps
                        )
                else:
                    pass
                clips.append(row)
        report["dtypes"][dtype] = clips
        del incremental, tokenizer
        torch.cuda.empty_cache()
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2)
    print(
        f"conv receptive frames: {report['conv_receptive_frames']:.2f}, "
        f"conv_lookback_frames {report['conv_lookback_frames']}"
    )
    for dtype, clips in report["dtypes"].items():
        print(f"## {dtype}")
        floor = max(clip["windowed_vs_one_shot"]["rel_l2"] for clip in clips)
        print(f"  windowed vs one-shot rel_l2 worst {floor:.2e}")
        for tail in TAILS:
            worst = max(clip["tails"][tail]["rel_l2"] for clip in clips)
            state = max(clip["tails"][tail]["state_max_abs"] for clip in clips)
            print(
                f"  tail {tail:>2}: rel_l2 worst {worst:.2e}, state max abs {state:.2e}"
            )
        if dtype == "bfloat16":
            for key in sorted(k for k in clips[0] if k.startswith("ms_")):
                print(
                    f"  {key}: median {statistics.median(c[key] for c in clips):.2f} ms"
                )
            print(f"  ref frames: {[c['ref_frames'] for c in clips]}")
        else:
            pass


if __name__ == "__main__":
    main()
