# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o duplex token sampling and its mutable unit state."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from sglang_omni.models.minicpm_o.special_tokens import MiniCPMOSpecialTokenIds


@dataclass(kw_only=True)
class DuplexSamplerState:
    """Mutable sampling history and policy for one generated unit."""

    special_tokens: MiniCPMOSpecialTokenIds
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float
    listen_prob_scale: float
    greedy: bool
    max_new_tokens: int
    repetition_window_size: int
    generation_step: int = 0
    force_listen_count: int = 0
    force_listen_counter: int = 0
    generated_history: list[int] = field(default_factory=list)
    current_turn_ended: bool = True
    forbidden_index: torch.Tensor


def forbidden_token_index(
    special: MiniCPMOSpecialTokenIds, vocab_size: int, device: torch.device
) -> torch.Tensor:
    """Rows the second-stage draw never picks, resolved once instead of per step."""
    tokens = sorted(
        token for token in {special.chunk_eos, *special.forbidden} if token < vocab_size
    )
    return torch.tensor(tokens, dtype=torch.long, device=device)


def top_k_top_p(logits: torch.Tensor, *, top_k: int, top_p: float) -> torch.Tensor:
    filtered = logits.clone()
    if 0 < top_k < filtered.numel():
        threshold = torch.topk(filtered, top_k).values[-1]
        filtered[filtered < threshold] = -torch.inf
    else:
        pass
    if 0.0 < top_p < 1.0:
        values, indices = torch.sort(filtered, descending=True)
        cumulative = torch.cumsum(F.softmax(values, dim=-1), dim=-1)
        remove = cumulative > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        filtered[indices[remove]] = -torch.inf
    else:
        pass
    return filtered


def draw(logits: torch.Tensor, *, greedy: bool) -> int:
    if greedy:
        return int(torch.argmax(logits).item())
    else:
        probabilities = F.softmax(logits, dim=-1)
        if not torch.isfinite(probabilities).all() or float(probabilities.sum()) <= 0:
            raise RuntimeError(
                "MiniCPM-o duplex sampler produced invalid probabilities"
            )
        else:
            pass
        return int(torch.multinomial(probabilities, 1).item())


def duplex_sample(logits: torch.Tensor, state: DuplexSamplerState) -> int:
    """Apply the two-stage unit sampler to one vocabulary row."""

    if logits.ndim == 2:
        if logits.shape[0] != 1:
            raise ValueError("duplex_sample expects one logits row")
        else:
            pass
        logits = logits[0]
    else:
        pass
    if logits.ndim != 1:
        raise ValueError("duplex_sample expects logits shaped [V] or [1, V]")
    else:
        pass
    special = state.special_tokens

    if state.generation_step >= state.max_new_tokens - 1:
        return special.chunk_eos
    else:
        if (
            state.generation_step == 0
            and state.force_listen_counter < state.force_listen_count
        ):
            state.force_listen_counter += 1
            return special.listen
        else:
            row = logits.float().clone()
            # note (Junnan Li): The first draw must use the unscaled model distribution.
            if draw(row, greedy=state.greedy) == special.chunk_eos:
                return special.chunk_eos
            else:
                row[state.forbidden_index] = -torch.inf

                penalty = float(state.repetition_penalty)
                if penalty <= 0:
                    raise ValueError("repetition_penalty must be positive")
                else:
                    pass
                if penalty != 1.0:
                    for token_id in set(
                        state.generated_history[-state.repetition_window_size :]
                    ):
                        if 0 <= int(token_id) < row.numel():
                            # note (Junnan Li): Matches the checkpoint sampler, which ignores the logit sign.
                            if penalty > 1.0:
                                row[int(token_id)] /= penalty
                            else:
                                row[int(token_id)] *= 1.0 / penalty
                        else:
                            pass
                else:
                    pass

                if state.listen_prob_scale != 1.0 and 0 <= special.listen < row.numel():
                    row[special.listen] *= float(state.listen_prob_scale)
                else:
                    pass

                if state.greedy or state.temperature <= 0:
                    candidate = int(torch.argmax(row).item())
                else:
                    filtered = top_k_top_p(
                        row / float(state.temperature),
                        top_k=int(state.top_k),
                        top_p=float(state.top_p),
                    )
                    candidate = draw(filtered, greedy=False)

                # note (Junnan Li): History retains controls before the mid-turn listen rewrite.
                state.generated_history.append(candidate)
                del state.generated_history[: -state.repetition_window_size]
                if candidate == special.listen and not state.current_turn_ended:
                    candidate = special.tts_bos
                else:
                    pass
                if candidate == special.turn_eos:
                    state.current_turn_ended = True
                elif candidate not in special.chunk_terminators:
                    state.current_turn_ended = False
                else:
                    pass
                return candidate
