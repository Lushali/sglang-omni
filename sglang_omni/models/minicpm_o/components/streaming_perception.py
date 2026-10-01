# SPDX-License-Identifier: Apache-2.0
"""Session-resident streaming perception for MiniCPM-o native duplex."""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from typing import Literal, Protocol, TypedDict

import numpy as np
import torch
from PIL import Image
from transformers import PreTrainedTokenizerBase

from sglang_omni.models.minicpm_o.components.audio_encoder import MiniCPMOAudioEncoder
from sglang_omni.models.minicpm_o.components.whisper_encoder import AudioEncoderState
from sglang_omni.preprocessing.audio import AudioMediaIO
from sglang_omni.proto.session import ResourceUsage
from sglang_omni.scheduling.speaker_cache import estimate_cache_bytes

SAMPLE_RATE = 16000
UNIT_MS = 1000
FIRST_CHUNK_MS = 1035
IMAGE_TOKENS = 64
MAX_FRAME_PIXELS = 4096 * 4096


class ImageEncoder(Protocol):
    def __call__(
        self, *, pixel_values: list[torch.Tensor], tgt_sizes: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        pass


class ImageFeatureBatch(TypedDict):
    pixel_values: list[list[torch.Tensor]]
    tgt_sizes: list[torch.Tensor]


class StreamingConfig(TypedDict):
    effective_first_chunk_ms: float


class ProcessorAudioFeatures(TypedDict):
    audio_features: torch.Tensor
    audio_feature_lens: list[torch.Tensor]


class StreamingAudioProcessor(Protocol):
    """The checkpoint processor's streaming surface used by one session."""

    def set_streaming_mode(
        self,
        *,
        mode: str,
        chunk_ms: int,
        first_chunk_ms: int,
        cnn_redundancy_ms: int,
        enable_sliding_window: bool,
        slide_trigger_seconds: float,
        slide_stride_seconds: float,
    ) -> None:
        pass

    def process_image(
        self, images: list[Image.Image], *, max_slice_nums: int
    ) -> ImageFeatureBatch:
        pass

    def get_streaming_chunk_size(self) -> int:
        pass

    def get_streaming_config(self) -> StreamingConfig:
        pass

    def process_audio(
        self, audio: np.ndarray, *, sampling_rate: int
    ) -> ProcessorAudioFeatures:
        pass

    def process_audio_streaming(
        self, audio: np.ndarray, *, reset: bool, return_batch_feature: bool
    ) -> ProcessorAudioFeatures:
        pass


class ProcessorFactory(Protocol):
    def __call__(self) -> StreamingAudioProcessor:
        pass


class EmbeddingSpanPlan(TypedDict):
    modality: Literal["audio", "image"]
    token_start: int
    token_end: int
    embed_start: int
    embed_end: int


class PerceptionStepPlan(TypedDict):
    token_ids: list[int]
    input_embeds: torch.Tensor
    embedding_spans: list[EmbeddingSpanPlan]
    prefill_schema: list[tuple[Literal["tok", "audio", "image"], int]]


@dataclass(kw_only=True)
class AudioFeatureBatch:
    audio_features: torch.Tensor
    audio_feature_lens: torch.Tensor


def audio_feature_batch(processor_output: ProcessorAudioFeatures) -> AudioFeatureBatch:
    return AudioFeatureBatch(
        audio_features=processor_output["audio_features"],
        audio_feature_lens=torch.cat(
            [length.reshape(-1) for length in processor_output["audio_feature_lens"]]
        ),
    )


@dataclass(kw_only=True)
class MiniCPMOPerceptionState:
    """All mutable checkpoint perception state owned by one session."""

    tokenizer: PreTrainedTokenizerBase
    processor: StreamingAudioProcessor
    audio_encoder: MiniCPMOAudioEncoder
    max_slice_nums: int
    image_encoder: ImageEncoder
    audio_buffer: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32)
    )
    audio_chunk_idx: int = 0
    audio_encoder_state: AudioEncoderState | None = None
    prefix_token_ids: list[int] = field(default_factory=list)
    prefix_embeds: torch.Tensor | None = None
    prefix_schema: list[tuple[Literal["tok", "audio"], int]] = field(
        default_factory=list
    )
    is_open: bool = True

    @classmethod
    def open(
        cls,
        *,
        tokenizer: PreTrainedTokenizerBase,
        processor: StreamingAudioProcessor,
        audio_encoder: MiniCPMOAudioEncoder,
        prompt: str,
        reference_audio: bytes,
        image_encoder: ImageEncoder,
        max_slice_nums: int,
    ) -> MiniCPMOPerceptionState:
        processor.set_streaming_mode(
            mode="exact",
            chunk_ms=UNIT_MS,
            first_chunk_ms=FIRST_CHUNK_MS,
            cnn_redundancy_ms=20,
            enable_sliding_window=True,
            slide_trigger_seconds=30.0,
            slide_stride_seconds=10.0,
        )
        state = cls(
            tokenizer=tokenizer,
            processor=processor,
            audio_encoder=audio_encoder,
            image_encoder=image_encoder,
            max_slice_nums=max_slice_nums,
        )
        prompt_ids = list(
            tokenizer.encode(
                f"<|im_start|>system\n{prompt}\n", add_special_tokens=False
            )
        )
        im_end_ids = list(tokenizer.encode("<|im_end|>", add_special_tokens=False))
        state.prefix_token_ids = list(prompt_ids)
        state.prefix_token_ids.append(
            tokenizer.convert_tokens_to_ids("<|audio_start|>")
        )
        reference_waveform, _ = AudioMediaIO(target_sr=SAMPLE_RATE).load_bytes(
            reference_audio
        )
        waveform = np.asarray(reference_waveform, dtype=np.float32).reshape(-1)
        batch = audio_feature_batch(
            processor.process_audio(waveform, sampling_rate=SAMPLE_RATE)
        )
        state.prefix_embeds = audio_encoder(
            audio_features=batch.audio_features,
            audio_feature_lens=batch.audio_feature_lens,
        )["audio_embeds"]
        count = int(state.prefix_embeds.shape[0])
        state.prefix_token_ids.extend([tokenizer.unk_token_id] * count)
        state.prefix_token_ids.append(tokenizer.convert_tokens_to_ids("<|audio_end|>"))
        state.prefix_token_ids.extend(im_end_ids)
        state.prefix_schema = [
            ("tok", len(prompt_ids) + 1),
            ("audio", count),
            ("tok", 1 + len(im_end_ids)),
        ]
        return state

    def close(self) -> None:
        self.is_open = False
        self.audio_buffer = np.zeros(0, dtype=np.float32)
        self.audio_encoder_state = None
        self.prefix_embeds = None

    def held(self) -> ResourceUsage:
        if not self.is_open:
            return ResourceUsage()
        else:
            size = (
                int(self.audio_buffer.nbytes)
                + (
                    self.audio_encoder_state.nbytes
                    if self.audio_encoder_state is not None
                    else 0
                )
                + estimate_cache_bytes(self.prefix_embeds)
            )
            return ResourceUsage(slots={"perception": 1}, bytes=max(size, 1))

    def encode_audio(self, pcm: np.ndarray) -> torch.Tensor:
        need_samples = self.processor.get_streaming_chunk_size()
        # note (Junnan Li): The checkpoint front-pads the first chunk to 1035 ms so the encoder's CNN context is full.
        if self.audio_chunk_idx == 0:
            first_chunk_samples = FIRST_CHUNK_MS * SAMPLE_RATE // 1000
            padding = max(first_chunk_samples - self.audio_buffer.size - pcm.size, 0)
        else:
            padding = 0
        self.audio_buffer = np.concatenate(
            [np.zeros(padding, dtype=np.float32), self.audio_buffer, pcm]
        )
        assert self.audio_buffer.size >= need_samples, (
            self.audio_buffer.size,
            need_samples,
        )
        batch = audio_feature_batch(
            self.processor.process_audio_streaming(
                self.audio_buffer[:need_samples].copy(),
                reset=False,
                return_batch_feature=True,
            )
        )
        audio_embeds, self.audio_encoder_state = self.audio_encoder.forward_streaming(
            audio_features=batch.audio_features,
            audio_feature_lens=batch.audio_feature_lens,
            state=self.audio_encoder_state,
            prefix_extra_frames=0 if self.audio_chunk_idx == 0 else 2,
            suffix_extra_frames=2,
        )
        if self.audio_chunk_idx == 0:
            consumed_ms = int(
                self.processor.get_streaming_config()["effective_first_chunk_ms"]
            )
            consumed_samples = consumed_ms * SAMPLE_RATE // 1000
        else:
            consumed_samples = need_samples
        self.audio_buffer = self.audio_buffer[consumed_samples:].copy()
        self.audio_chunk_idx += 1
        return audio_embeds

    def encode_image(self, encoded_image: bytes) -> torch.Tensor:
        with Image.open(BytesIO(encoded_image)) as image:
            if image.format not in ("JPEG", "PNG"):
                raise ValueError("unit image must be JPEG or PNG")
            elif image.width * image.height > MAX_FRAME_PIXELS:
                raise ValueError("unit image exceeds pixel limit")
            else:
                frame = image.convert("RGB")
        processed = self.processor.process_image(
            [frame], max_slice_nums=self.max_slice_nums
        )
        image_embeds = self.image_encoder(
            pixel_values=processed["pixel_values"][0],
            tgt_sizes=processed["tgt_sizes"][0],
        )["image_embeds"]
        assert image_embeds.ndim == 2 and image_embeds.shape[0] % IMAGE_TOKENS == 0
        return image_embeds

    def build_step_plan(
        self, audio_embeds: torch.Tensor, image_embeds: tuple[torch.Tensor, ...] = ()
    ) -> PerceptionStepPlan:
        token_ids: list[int] = []
        embed_blocks: list[torch.Tensor] = []
        spans: list[EmbeddingSpanPlan] = []

        def add_embeds(
            values: torch.Tensor, modality: Literal["audio", "image"] = "audio"
        ) -> None:
            start = len(token_ids)
            count = int(values.shape[0])
            embed_start = sum(int(block.shape[0]) for block in embed_blocks)
            token_ids.extend([self.tokenizer.unk_token_id] * count)
            embed_blocks.append(values)
            spans.append(
                EmbeddingSpanPlan(
                    modality=modality,
                    token_start=start,
                    token_end=start + count,
                    embed_start=embed_start,
                    embed_end=embed_start + count,
                )
            )

        # note (Junnan Li): The system prefix is included in prefill but excluded from the unit schema.
        if self.audio_chunk_idx == 1 and self.prefix_token_ids:
            cursor = 0
            embed_cursor = 0
            for kind, count in self.prefix_schema:
                if kind == "tok":
                    token_ids.extend(self.prefix_token_ids[cursor : cursor + count])
                else:
                    assert self.prefix_embeds is not None
                    add_embeds(self.prefix_embeds[embed_cursor : embed_cursor + count])
                    embed_cursor += count
                cursor += count
        else:
            pass

        token_ids.append(self.tokenizer.convert_tokens_to_ids("<unit>"))
        schema: list[tuple[Literal["tok", "audio", "image"], int]] = [("tok", 1)]
        for frame_embeds in image_embeds:
            assert (
                frame_embeds.ndim == 2
                and frame_embeds.shape[1] == audio_embeds.shape[1]
            )
            assert (
                frame_embeds.shape[0] > 0 and frame_embeds.shape[0] % IMAGE_TOKENS == 0
            )
            for slice_index, slice_embeds in enumerate(
                frame_embeds.split(IMAGE_TOKENS)
            ):
                marker = "image" if slice_index == 0 else "slice"
                token_ids.append(self.tokenizer.convert_tokens_to_ids(f"<{marker}>"))
                schema[-1] = ("tok", schema[-1][1] + 1)
                add_embeds(slice_embeds, "image")
                token_ids.append(self.tokenizer.convert_tokens_to_ids(f"</{marker}>"))
                schema.extend([("image", IMAGE_TOKENS), ("tok", 1)])
        schema.append(("audio", int(audio_embeds.shape[0])))
        add_embeds(audio_embeds)
        return PerceptionStepPlan(
            token_ids=token_ids,
            input_embeds=torch.cat(embed_blocks, dim=0),
            embedding_spans=spans,
            prefill_schema=schema,
        )
