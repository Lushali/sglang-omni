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


@pytest.mark.parametrize("chunk_index", [1, 2])
@pytest.mark.parametrize("reference_audio", [False, True])
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
def test_image_layout_and_relay_order(
    state: MiniCPMOPerceptionState, chunk_index: int
) -> None:
    state.audio_chunk_idx = chunk_index
    state.prefix_token_ids = [7, 0, 0, 8]
    state.prefix_schema = [("tok", 1), ("audio", 2), ("tok", 1)]
    state.prefix_embeds = torch.full((2, 4), 5.0)
    image = torch.full((64, 4), 6.0)
    audio = torch.full((10, 4), 9.0)
    plan = state.build_step_plan(audio, (image,))
    prefix = state.prefix_token_ids if chunk_index == 1 else []
    assert plan["token_ids"] == prefix + [1, 2] + [0] * 64 + [3] + [0] * 10
    assert plan["prefill_schema"] == [
        ("tok", 2),
        ("image", 64),
        ("tok", 1),
        ("audio", 10),
    ]
    image_span, audio_span = plan["embedding_spans"][-2:]
    assert (image_span["token_start"], image_span["token_end"]) == (
        len(prefix) + 2,
        len(prefix) + 66,
    )
    assert (audio_span["token_start"], audio_span["token_end"]) == (
        len(prefix) + 67,
        len(prefix) + 77,
    )
    assert [span["modality"] for span in plan["embedding_spans"][-2:]] == [
        "image",
        "audio",
    ]
    blocks = [state.prefix_embeds, image, audio] if prefix else [image, audio]
    assert torch.equal(plan["input_embeds"], torch.cat(blocks))
    for span, block in zip(plan["embedding_spans"], blocks):
        assert torch.equal(
            plan["input_embeds"][span["embed_start"] : span["embed_end"]], block
        )


@pytest.mark.parametrize("format", ["JPEG", "PNG"])
def test_decode_image_reuses_checkpoint_processor(
    state: MiniCPMOPerceptionState, format: str
) -> None:
    encoded = BytesIO()
    Image.new("RGB", (16, 12), "red").save(encoded, format=format)
    pixels = torch.zeros(3, 14, 28)
    sizes = torch.tensor([[1, 2]])
    state.processor.process_image.return_value = {
        "pixel_values": [[pixels]],
        "tgt_sizes": [sizes],
    }
    state.image_encoder = Mock(return_value={"image_embeds": torch.full((64, 4), 6.0)})
    result = state.encode_image(encoded.getvalue())
    assert result.shape == (64, 4)
    args, kwargs = state.processor.process_image.call_args
    assert kwargs == {"max_slice_nums": 1}
    assert args[0][0].mode == "RGB"
    assert args[0][0].size == (16, 12)
    state.image_encoder.assert_called_once_with(pixel_values=[pixels], tgt_sizes=sizes)


@pytest.mark.parametrize("format", ["GIF", "BMP"])
def test_reject_non_jpeg_png(state: MiniCPMOPerceptionState, format: str) -> None:
    encoded = BytesIO()
    Image.new("RGB", (8, 8)).save(encoded, format=format)
    with pytest.raises(ValueError, match="JPEG or PNG"):
        state.encode_image(encoded.getvalue())
    state.processor.process_image.assert_not_called()


def test_reject_invalid_image(state: MiniCPMOPerceptionState) -> None:
    with pytest.raises(UnidentifiedImageError):
        state.encode_image(b"\x89PNG\r\n\x1a\ninvalid")
    state.processor.process_image.assert_not_called()


@pytest.mark.parametrize("shape", [(63, 4), (64, 3), (1, 64, 4)])
def test_reject_invalid_embedding_slot(
    state: MiniCPMOPerceptionState, shape: tuple[int, ...]
) -> None:
    with pytest.raises(AssertionError):
        state.build_step_plan(torch.zeros(10, 4), (torch.zeros(shape),))


def test_reject_pixel_limit_before_decode(
    state: MiniCPMOPerceptionState, monkeypatch: pytest.MonkeyPatch
) -> None:
    encoded = BytesIO()
    Image.new("RGB", (16, 16)).save(encoded, format="PNG")
    monkeypatch.setattr(
        "sglang_omni.models.minicpm_o.components.streaming_perception.MAX_FRAME_PIXELS",
        15,
    )
    with pytest.raises(ValueError, match="pixel limit"):
        state.encode_image(encoded.getvalue())
    state.processor.process_image.assert_not_called()


def test_reject_truncated_png(state: MiniCPMOPerceptionState) -> None:
    encoded = BytesIO()
    Image.new("RGB", (16, 16), "red").save(encoded, format="PNG")
    with pytest.raises(OSError):
        state.encode_image(encoded.getvalue()[:45])
    state.processor.process_image.assert_not_called()


def test_multiple_frames_and_slices_follow_official_order(
    state: MiniCPMOPerceptionState,
) -> None:
    first = torch.cat([torch.full((64, 4), float(i)) for i in (1, 2, 3)])
    second = torch.full((64, 4), 4.0)
    audio = torch.full((10, 4), 9.0)
    plan = state.build_step_plan(audio, (first, second))
    assert plan["token_ids"] == (
        [1, 2]
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
    assert torch.equal(plan["input_embeds"], torch.cat([first, second, audio]))
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


def test_hd_slice_count_reaches_processor(state: MiniCPMOPerceptionState) -> None:
    state.max_slice_nums = 4
    state.processor.process_image.return_value = {
        "pixel_values": [[torch.zeros(3, 14, 28)]],
        "tgt_sizes": [torch.tensor([[1, 2]])],
    }
    embeds = torch.zeros(5 * 64, 4)
    state.image_encoder = Mock(return_value={"image_embeds": embeds})
    encoded = BytesIO()
    Image.new("RGB", (32, 32)).save(encoded, format="PNG")
    assert torch.equal(state.encode_image(encoded.getvalue()), embeds)
    assert state.processor.process_image.call_args.kwargs["max_slice_nums"] == 4
