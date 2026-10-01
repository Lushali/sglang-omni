# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o thinker model runner."""

from __future__ import annotations

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.model_runner.prefill_inputs import (
    OmniPrefillInputs,
    attach_omni_prefill_inputs,
)
from sglang_omni.models.minicpm_o.duplex_sampler import (
    DuplexSamplerState,
    build_forbidden_token_index,
    duplex_sample,
)
from sglang_omni.models.minicpm_o.special_tokens import (
    MiniCPMOSpecialTokenIds,
    resolve_special_token_ids,
)
from sglang_omni.models.minicpm_o.thinker_model_runner import (
    MiniCPMOThinkerModelRunner as OfflineThinkerModelRunner,
)
from sglang_omni.models.minicpm_o.thinker_state import DuplexUnitRequestData
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.scheduling.sglang_backend.request_data import session_prefill_rows
from sglang_omni.scheduling.types import (
    RequestOutput,
    SchedulerOutput,
    SchedulerRequest,
)


class MiniCPMOThinkerModelRunner(OfflineThinkerModelRunner):
    """Extend the shared MiniCPM-o runner with duplex sampling and media history."""

    def __init__(
        self, tp_worker: ModelWorker, output_processor: SGLangOutputProcessor
    ) -> None:
        super().__init__(tp_worker, output_processor)
        self.special_tokens: MiniCPMOSpecialTokenIds | None = None
        self.forbidden_token_index: torch.Tensor | None = None

    @staticmethod
    def is_duplex_request(request: SchedulerRequest) -> bool:
        return isinstance(request.data, DuplexUnitRequestData)

    def custom_prefill_forward(
        self,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> GenerationBatchResult | None:
        duplex = [request for request in requests if self.is_duplex_request(request)]
        if not duplex:
            return super().custom_prefill_forward(
                forward_batch, schedule_batch, requests
            )
        else:
            if len(duplex) != len(requests):
                raise RuntimeError(
                    "offline and duplex thinker requests cannot share prefill"
                )
            else:
                pass

            embed_tokens = self.embed_tokens
            last_token_id = embed_tokens.num_embeddings - 1
            rows = [
                session_prefill_rows(
                    request.data,
                    lambda token_ids: embed_tokens(token_ids.clamp(0, last_token_id)),
                    forward_batch.input_ids.device,
                )
                for request in requests
            ]
            attach_omni_prefill_inputs(
                forward_batch,
                OmniPrefillInputs(input_embeds=torch.cat(rows, dim=0)),
            )
            return None

    def sample_before_post_prefill(
        self,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> bool:
        if any(self.is_duplex_request(request) for request in requests):
            return True
        else:
            return super().sample_before_post_prefill(
                forward_batch, schedule_batch, requests
            )

    def sample_before_post_decode(
        self,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> bool:
        if any(self.is_duplex_request(request) for request in requests):
            return True
        else:
            return super().sample_before_post_decode(
                forward_batch, schedule_batch, requests
            )

    def lookahead_eligible(self, batch: ScheduleBatch) -> bool:
        # note (Junnan Li): Lookahead would sample before repetition history advances.
        return False

    def resolve_special_tokens(
        self, data: DuplexUnitRequestData
    ) -> MiniCPMOSpecialTokenIds:
        if self.special_tokens is None:
            self.special_tokens = resolve_special_token_ids(
                data.req.tokenizer,
                bad_token_ids=tuple(data.req.tokenizer.bad_token_ids),
            )
        else:
            pass
        return self.special_tokens

    def sample_next_token_ids(
        self,
        logits_output: LogitsProcessorOutput,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> torch.Tensor:
        duplex_indices = [
            index
            for index, request in enumerate(requests)
            if self.is_duplex_request(request)
        ]
        if not duplex_indices:
            return super().sample_next_token_ids(
                logits_output, forward_batch, schedule_batch, requests
            )
        else:
            logits = logits_output.next_token_logits
            original_logits = logits.clone()
            if len(duplex_indices) == len(requests):
                result = torch.empty(
                    len(requests), dtype=torch.long, device=logits.device
                )
            else:
                result = (
                    super()
                    .sample_next_token_ids(
                        logits_output, forward_batch, schedule_batch, requests
                    )
                    .clone()
                )

            for index in duplex_indices:
                data = requests[index].data
                thinker_state = data.thinker_state
                sampling = data.sampling
                special_tokens = self.resolve_special_tokens(data)
                if self.forbidden_token_index is None:
                    self.forbidden_token_index = build_forbidden_token_index(
                        special_tokens,
                        original_logits.shape[-1],
                        original_logits.device,
                    )
                else:
                    pass
                sampler_state = DuplexSamplerState(
                    special_tokens=special_tokens,
                    forbidden_token_index=self.forbidden_token_index,
                    generation_step=data.generation_steps,
                    force_listen_count=1 if data.is_listen_forced else 0,
                    force_listen_counter=0,
                    generated_history=thinker_state.generated_history,
                    is_turn_ended=thinker_state.is_turn_ended,
                    temperature=sampling.temperature,
                    top_k=sampling.top_k,
                    top_p=sampling.top_p,
                    repetition_penalty=sampling.repetition_penalty,
                    listen_prob_scale=sampling.listen_prob_scale,
                    greedy=sampling.greedy,
                    max_new_tokens=sampling.max_new_tokens_per_unit,
                    repetition_window_size=sampling.repetition_window_size,
                )
                token_id = duplex_sample(original_logits[index], sampler_state)
                thinker_state.is_turn_ended = sampler_state.is_turn_ended
                if data.is_listen_forced and data.generation_steps == 0:
                    thinker_state.force_listen_counter += 1
                else:
                    pass
                result[index] = token_id
            return result

    # note (Junnan Li): FULL capture must match decode graphs and retain talker conditioning.
    def requested_capture_hidden_mode_prefill(
        self, schedule_batch: ScheduleBatch, requests: list[SchedulerRequest]
    ) -> CaptureHiddenMode:
        return CaptureHiddenMode.FULL

    def requested_capture_hidden_mode_decode(
        self, schedule_batch: ScheduleBatch, requests: list[SchedulerRequest]
    ) -> CaptureHiddenMode:
        return CaptureHiddenMode.FULL

    def post_process_outputs(
        self,
        result: GenerationBatchResult,
        scheduler_output: SchedulerOutput,
        outputs: dict[str, RequestOutput],
    ) -> None:
        """Pair generated tokens with their next-step hidden states for the talker."""
        offline_outputs = {
            request.request_id: outputs[request.request_id]
            for request in scheduler_output.requests
            if not self.is_duplex_request(request)
        }
        if offline_outputs:
            super().post_process_outputs(result, scheduler_output, offline_outputs)
        else:
            pass
        for scheduler_request in scheduler_output.requests:
            request_output = outputs[scheduler_request.request_id]
            data = scheduler_request.data
            if isinstance(data, DuplexUnitRequestData):
                sampled_token_id = int(request_output.data)
                special_tokens = self.resolve_special_tokens(data)
                pending_token_id = data.pending_unit_token
                if pending_token_id is not None and data.generation_steps >= 2:
                    hidden_state = request_output.extra["hidden_states"]
                    hidden_state = (
                        hidden_state.reshape(-1, hidden_state.shape[-1])[-1]
                        .detach()
                        .clone()
                    )
                    data.talker_conditions.append(
                        (
                            pending_token_id,
                            hidden_state.to("cpu"),
                            pending_token_id == special_tokens.turn_eos,
                        )
                    )
                else:
                    pass
                if sampled_token_id in special_tokens.chunk_terminators:
                    data.pending_unit_token = None
                    continue
                else:
                    pass
                if data.generation_steps > 0:
                    data.generated_unit_ids.append(sampled_token_id)
                else:
                    pass
                data.pending_unit_token = sampled_token_id
                if sampled_token_id == special_tokens.turn_eos:
                    data.thinker_state.is_turn_ended = True
                elif sampled_token_id not in special_tokens.chunk_terminators:
                    data.thinker_state.is_turn_ended = False
                else:
                    pass
                continue
            else:
                pass


__all__ = [
    "MiniCPMOThinkerModelRunner",
]
