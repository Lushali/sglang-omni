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
    is_turn_ended: bool = True
    forbidden_token_index: torch.Tensor


def build_forbidden_token_index(
    special_tokens: MiniCPMOSpecialTokenIds, vocab_size: int, device: torch.device
) -> torch.Tensor:
    """Rows the second-stage sample never picks, resolved once instead of per step."""
    forbidden_token_ids = sorted(
        token_id
        for token_id in {special_tokens.chunk_eos, *special_tokens.forbidden}
        if token_id < vocab_size
    )
    return torch.tensor(forbidden_token_ids, dtype=torch.long, device=device)


def filter_top_k_top_p(
    logits: torch.Tensor, *, top_k: int, top_p: float
) -> torch.Tensor:
    filtered_logits = logits.clone()
    if 0 < top_k < filtered_logits.numel():
        threshold = torch.topk(filtered_logits, top_k).values[-1]
        filtered_logits[filtered_logits < threshold] = -torch.inf
    else:
        pass
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered_logits, descending=True)
        cumulative_probabilities = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1
        )
        should_remove = cumulative_probabilities > top_p
        should_remove[1:] = should_remove[:-1].clone()
        should_remove[0] = False
        filtered_logits[sorted_indices[should_remove]] = -torch.inf
    else:
        pass
    return filtered_logits


def sample_token_id(logits: torch.Tensor, *, greedy: bool) -> int:
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
    special_tokens = state.special_tokens

    if state.generation_step >= state.max_new_tokens - 1:
        return special_tokens.chunk_eos
    else:
        if (
            state.generation_step == 0
            and state.force_listen_counter < state.force_listen_count
        ):
            state.force_listen_counter += 1
            return special_tokens.listen
        else:
            logits_row = logits.float().clone()
            # note (Junnan Li): The first sample must use the unscaled model distribution.
            if (
                sample_token_id(logits_row, greedy=state.greedy)
                == special_tokens.chunk_eos
            ):
                return special_tokens.chunk_eos
            else:
                logits_row[state.forbidden_token_index] = -torch.inf

                penalty = float(state.repetition_penalty)
                if penalty <= 0:
                    raise ValueError("repetition_penalty must be positive")
                else:
                    pass
                if penalty != 1.0:
                    for token_id in set(
                        state.generated_history[-state.repetition_window_size :]
                    ):
                        if 0 <= int(token_id) < logits_row.numel():
                            # note (Junnan Li): Matches the checkpoint sampler, which ignores the logit sign.
                            if penalty > 1.0:
                                logits_row[int(token_id)] /= penalty
                            else:
                                logits_row[int(token_id)] *= 1.0 / penalty
                        else:
                            pass
                else:
                    pass

                if (
                    state.listen_prob_scale != 1.0
                    and 0 <= special_tokens.listen < logits_row.numel()
                ):
                    logits_row[special_tokens.listen] *= float(state.listen_prob_scale)
                else:
                    pass

                if state.greedy or state.temperature <= 0:
                    candidate_token_id = int(torch.argmax(logits_row).item())
                else:
                    filtered_logits = filter_top_k_top_p(
                        logits_row / float(state.temperature),
                        top_k=int(state.top_k),
                        top_p=float(state.top_p),
                    )
                    candidate_token_id = sample_token_id(filtered_logits, greedy=False)

                # note (Junnan Li): History retains controls before the mid-turn listen rewrite.
                state.generated_history.append(candidate_token_id)
                del state.generated_history[: -state.repetition_window_size]
                if (
                    candidate_token_id == special_tokens.listen
                    and not state.is_turn_ended
                ):
                    candidate_token_id = special_tokens.tts_bos
                else:
                    pass
                if candidate_token_id == special_tokens.turn_eos:
                    state.is_turn_ended = True
                elif candidate_token_id not in special_tokens.chunk_terminators:
                    state.is_turn_ended = False
                else:
                    pass
                return candidate_token_id
