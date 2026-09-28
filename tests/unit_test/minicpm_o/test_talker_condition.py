# SPDX-License-Identifier: Apache-2.0
"""Speech-span conditioning at the offline request boundary."""

import pytest
import torch
from torch import nn

from sglang_omni.models.minicpm_o.components.talker import build_tts_condition
from sglang_omni.models.minicpm_o.payload_types import MiniCPMOPipelineState
from sglang_omni.models.minicpm_o.talker_request import build_talker_request


@pytest.mark.parametrize("output_ids", [[1, 2], [3, 4], [3, 1, 2, 4]])
def test_request_speech_span_condition(output_ids: list[int]) -> None:
    hidden_states = [torch.arange(8, dtype=torch.float32) for _ in output_ids]
    state = MiniCPMOPipelineState(
        thinker_out={
            "output_ids": output_ids,
            "extra_model_outputs": {"hidden_states_seq": hidden_states},
        }
    )
    span = build_talker_request(state, tts_bos_token_id=3, tts_eos_token_id=4)
    embedding = nn.Embedding(8, 8)

    condition = build_tts_condition(
        span["tts_token_ids"],
        span["tts_hidden"],
        text_embedding=embedding,
        semantic_projector=nn.Identity(),
        boundary_tokens=(5, 6),
        normalize_projected_hidden=False,
    )

    boundary = embedding(torch.tensor([5, 6]))
    if output_ids == [3, 1, 2, 4]:
        expected = torch.cat(
            (
                embedding(torch.tensor([1, 2])) + torch.stack(hidden_states[1:3]),
                boundary,
            )
        )
    else:
        assert span["tts_hidden"].shape == (0,)
        expected = boundary
    torch.testing.assert_close(condition, expected)


def test_nonempty_condition_rejects_misaligned_hidden() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        build_tts_condition(
            torch.tensor([1, 2]),
            torch.empty(3, 8),
            text_embedding=nn.Embedding(8, 8),
            semantic_projector=nn.Identity(),
            boundary_tokens=(5, 6),
            normalize_projected_hidden=False,
        )
