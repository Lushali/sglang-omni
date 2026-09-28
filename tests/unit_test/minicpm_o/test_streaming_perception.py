# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for native duplex image and audio unit plans."""

from io import BytesIO
from unittest.mock import Mock

import pytest
import torch
from PIL import Image, UnidentifiedImageError

from sglang_omni.models.minicpm_o.components.streaming_perception import (
    MiniCPMOPerceptionState,
)


@pytest.fixture
def state() -> MiniCPMOPerceptionState:
    tokenizer = Mock(unk_token_id=0)
    tokenizer.convert_tokens_to_ids.side_effect = {
        "<unit>": 1,
        "<image>": 2,
        "</image>": 3,
        "<slice>": 4,
        "</slice>": 5,
    }.__getitem__
    return MiniCPMOPerceptionState(
        tokenizer=tokenizer,
        processor=Mock(),
        audio_encoder=Mock(),
        max_slice_nums=1,
    )


@pytest.mark.parametrize(
    ("chunk_index", "reference_audio"), [(1, False), (1, True), (2, True)]
)
def test_audio_plan_without_image(
    state: MiniCPMOPerceptionState, chunk_index: int, reference_audio: bool
) -> None:
    state.audio_chunk_idx = chunk_index
    if reference_audio:
        state.prefix_token_ids = [7, 0, 0, 8]
        state.prefix_schema = [("tok", 1), ("audio", 2), ("tok", 1)]
        state.prefix_embeds = torch.full((2, 4), 5.0)
        prefix_spans = [
            dict(
                modality="audio", token_start=1, token_end=3, embed_start=0, embed_end=2
            )
        ]
    else:
        state.prefix_token_ids = [7, 8]
        state.prefix_schema = [("tok", 2)]
        prefix_spans = []
    audio = torch.full((10, 4), 9.0)
    plan = state.build_step_plan(audio)
    is_first_unit = chunk_index == 1
    prefix_ids = state.prefix_token_ids if is_first_unit else []
    spans = prefix_spans if is_first_unit else []
    embed_start = 2 if is_first_unit and reference_audio else 0
    token_start = len(prefix_ids) + 1
    assert plan["token_ids"] == prefix_ids + [1] + [0] * 10
    assert plan["embedding_spans"] == spans + [
        dict(
            modality="audio",
            token_start=token_start,
            token_end=token_start + 10,
            embed_start=embed_start,
            embed_end=embed_start + 10,
        )
    ]
    assert plan["input_embeds"].shape[0] == embed_start + 10
    assert torch.equal(plan["input_embeds"][embed_start:], audio)
    assert plan["prefill_schema"] == [("tok", 1), ("audio", 10)]
    assert plan["decode_budget"] == 20


@pytest.mark.parametrize("chunk_index", [1, 2])
def test_image_plan_follows_official_slice_order(
    state: MiniCPMOPerceptionState, chunk_index: int
) -> None:
    state.audio_chunk_idx = chunk_index
    state.prefix_token_ids = [7, 0, 0, 8]
    state.prefix_schema = [("tok", 1), ("audio", 2), ("tok", 1)]
    state.prefix_embeds = torch.full((2, 4), 5.0)
    first = torch.cat([torch.full((64, 4), float(i)) for i in (1, 2, 3)])
    second = torch.full((64, 4), 4.0)
    audio = torch.full((10, 4), 9.0)
    plan = state.build_step_plan(audio, (first, second))
    prefix = state.prefix_token_ids if chunk_index == 1 else []
    offset = len(prefix)
    assert plan["token_ids"] == (
        prefix
        + [1, 2]
        + [0] * 64
        + [3, 4]
        + [0] * 64
        + [5, 4]
        + [0] * 64
        + [5, 2]
        + [0] * 64
        + [3]
        + [0] * 10
    )
    assert plan["prefill_schema"] == [
        ("tok", 2),
        ("image", 64),
        ("tok", 2),
        ("image", 64),
        ("tok", 2),
        ("image", 64),
        ("tok", 2),
        ("image", 64),
        ("tok", 1),
        ("audio", 10),
    ]
    unit_spans = plan["embedding_spans"][-5:]
    assert [span["modality"] for span in unit_spans] == ["image"] * 4 + ["audio"]
    assert [(span["token_start"], span["token_end"]) for span in unit_spans] == [
        (offset + 2, offset + 66),
        (offset + 68, offset + 132),
        (offset + 134, offset + 198),
        (offset + 200, offset + 264),
        (offset + 265, offset + 275),
    ]
    blocks = [state.prefix_embeds] if prefix else []
    blocks += [*first.split(64), second, audio]
    assert torch.equal(plan["input_embeds"], torch.cat(blocks))
    for span, block in zip(plan["embedding_spans"], blocks, strict=True):
        assert torch.equal(
            plan["input_embeds"][span["embed_start"] : span["embed_end"]], block
        )


@pytest.mark.parametrize(("format", "max_slice_nums"), [("JPEG", 1), ("PNG", 4)])
def test_decode_image_reuses_checkpoint_processor(
    state: MiniCPMOPerceptionState, format: str, max_slice_nums: int
) -> None:
    state.max_slice_nums = max_slice_nums
    encoded = BytesIO()
    Image.new("RGB", (16, 12), "red").save(encoded, format=format)
    pixels = torch.zeros(3, 14, 28)
    sizes = torch.tensor([[1, 2]])
    state.processor.process_image.return_value = {
        "pixel_values": [[pixels]],
        "tgt_sizes": [sizes],
    }
    embeds = torch.zeros(max_slice_nums * 64, 4)
    state.image_encoder = Mock(return_value={"image_embeds": embeds})
    assert torch.equal(state.encode_image(encoded.getvalue()), embeds)
    args, kwargs = state.processor.process_image.call_args
    assert kwargs == {"max_slice_nums": max_slice_nums}
    assert args[0][0].mode == "RGB"
    assert args[0][0].size == (16, 12)
    state.image_encoder.assert_called_once_with(pixel_values=[pixels], tgt_sizes=sizes)


def encoded_image(format: str) -> bytes:
    encoded = BytesIO()
    Image.new("RGB", (16, 16)).save(encoded, format=format)
    return encoded.getvalue()


@pytest.mark.parametrize(
    ("encoded", "error", "match"),
    [
        (encoded_image("GIF"), ValueError, "JPEG or PNG"),
        (b"\x89PNG\r\n\x1a\ninvalid", UnidentifiedImageError, None),
        (encoded_image("PNG")[:45], OSError, None),
    ],
)
def test_reject_undecodable_frame_before_processor(
    state: MiniCPMOPerceptionState,
    encoded: bytes,
    error: type[Exception],
    match: str | None,
) -> None:
    with pytest.raises(error, match=match):
        state.encode_image(encoded)
    state.processor.process_image.assert_not_called()


@pytest.mark.parametrize("shape", [(63, 4), (64, 3)])
def test_reject_invalid_embedding_slot(
    state: MiniCPMOPerceptionState, shape: tuple[int, ...]
) -> None:
    with pytest.raises(AssertionError):
        state.build_step_plan(torch.zeros(10, 4), (torch.zeros(shape),))


def test_reject_pixel_limit_before_decode(
    state: MiniCPMOPerceptionState, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "sglang_omni.models.minicpm_o.components.streaming_perception.MAX_FRAME_PIXELS",
        15,
    )
    with pytest.raises(ValueError, match="pixel limit"):
        state.encode_image(encoded_image("PNG"))
    state.processor.process_image.assert_not_called()
