# SPDX-License-Identifier: Apache-2.0
"""Execute native relay methods with CPU backend stubs, without optional runtimes."""

import ast
import collections
import logging
import sys
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Protocol, TypedDict
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from PIL import Image

from sglang_omni.models.minicpm_o.components.streaming_perception import (
    IMAGE_TOKENS,
    MiniCPMOPerceptionState,
)
from sglang_omni.models.minicpm_o.duplex_sampler import (
    DuplexSamplerState,
    duplex_sample,
    forbidden_token_index,
)
from sglang_omni.models.minicpm_o.native_config import (
    MiniCPMODuplexPipelineConfig,
    MiniCPMODuplexSampling,
)
from sglang_omni.models.minicpm_o.special_tokens import (
    REQUIRED_SPECIAL_TOKENS,
    resolve_special_token_ids,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.scheduling.types import ARRequestData


@pytest.fixture
def relay(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    module = ModuleType("native_vision_test")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    module.__dict__.update(
        Path=Path,
        collections=collections,
        dataclass=dataclass,
        field=field,
        Protocol=Protocol,
        TypedDict=TypedDict,
        ARRequestData=ARRequestData,
        IMAGE_TOKENS=IMAGE_TOKENS,
        np=np,
        torch=torch,
        array=array,
        SessionHooks=object,
        ARSessionAdapter=object,
        OfflineThinkerModelRunner=object,
        MiniCPMOPerceptionState=MiniCPMOPerceptionState,
        resolve_special_token_ids=resolve_special_token_ids,
        SamplingParams=Mock(side_effect=Mock),
        Req=Mock(
            side_effect=lambda request_id, text, ids, params, **kwargs: Mock(
                rid=request_id,
                origin_input_ids=ids,
                sampling_params=params,
                input_embeds=None,
                multimodal_inputs=None,
            )
        ),
        SessionParams=SimpleNamespace,
        TokenizedGenerateReqInput=SimpleNamespace,
        Capabilities=Mock(),
        RealtimeDeployment=Mock(),
        Image=Image,
        MiniCPMODuplexSampling=MiniCPMODuplexSampling,
        logger=logging.getLogger("native_vision_test"),
    )
    root = Path(__file__).resolve().parents[3] / "sglang_omni"
    sources = {
        "scheduling/sglang_backend/request_data.py": {
            "EmbeddingSpan",
            "SGLangARRequestData",
            "splice_embedding_spans",
        },
        "models/minicpm_o/thinker_state.py": {
            "MiniCPMOThinkerSessionState",
            "DuplexUnitRequestData",
        },
        "models/minicpm_o/native_stages.py": {
            "PerceptionHooks",
            "create_perception_scheduler",
        },
        "models/minicpm_o/session_adapters.py": {
            "ThinkerAdapter",
            "build_realtime_deployment",
        },
        "models/minicpm_o/native_thinker_model_runner.py": {
            "MiniCPMOThinkerModelRunner",
        },
        "scheduling/sglang_backend/ar_session.py": {"ARSessionBridge"},
    }
    for filename, names in sources.items():
        tree = ast.parse((root / filename).read_text())
        # Optional runtime imports are replaced, while method bodies execute unchanged.
        tree.body = [
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            )
        ] + [
            node
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
        ]
        exec(
            compile(ast.fix_missing_locations(tree), filename, "exec"), module.__dict__
        )
    return module


@pytest.fixture
def perception() -> MiniCPMOPerceptionState:
    tokenizer = Mock(unk_token_id=0, bad_token_ids=[])
    tokenizer.convert_tokens_to_ids.side_effect = dict(
        zip(REQUIRED_SPECIAL_TOKENS, range(1, 17))
    ).__getitem__
    state = MiniCPMOPerceptionState(
        tokenizer=tokenizer, processor=Mock(), audio_encoder=Mock(), max_slice_nums=1
    )
    state.encode_audio = Mock(return_value=torch.full((10, 4), 9.0))
    state.encode_image = Mock(return_value=torch.full((64, 4), 6.0))
    return state


@pytest.mark.parametrize("greedy", [True, False])
@pytest.mark.parametrize("use_runner", [True, False])
def test_checkpoint_bad_tokens_are_not_sampled(
    relay: ModuleType, greedy: bool, use_runner: bool
) -> None:
    """Both duplex initialization paths retain the checkpoint sampling mask."""
    tokenizer = Mock(unk_token_id=0, bad_token_ids=[7, 8, 94])
    tokenizer.convert_tokens_to_ids.side_effect = dict(
        zip(REQUIRED_SPECIAL_TOKENS, range(100, 116))
    ).__getitem__
    if use_runner:
        runner = relay.MiniCPMOThinkerModelRunner.__new__(
            relay.MiniCPMOThinkerModelRunner
        )
        runner.special_tokens = None
        special = runner.special_for_data(
            SimpleNamespace(req=Mock(tokenizer=tokenizer))
        )
    else:
        special = relay.ThinkerAdapter(tokenizer, 128).special
    logits = torch.full((128,), -torch.inf)
    logits[7] = 100.0
    logits[special.tts_pad] = 99.0
    logits[42] = 0.0
    state = DuplexSamplerState(
        special_tokens=special,
        forbidden_index=forbidden_token_index(special, 128, torch.device("cpu")),
        temperature=0.7,
        top_k=1,
        top_p=0.8,
        repetition_penalty=1.0,
        listen_prob_scale=1.0,
        greedy=greedy,
    )
    assert duplex_sample(logits, state) == 42


@pytest.mark.parametrize("has_image", [False, True])
@pytest.mark.parametrize("first_unit", [False, True])
def test_append_and_thinker_splice(
    relay: ModuleType,
    perception: MiniCPMOPerceptionState,
    has_image: bool,
    first_unit: bool,
) -> None:
    identity = SessionIdentity("vision")
    hooks = relay.PerceptionHooks(
        perception.tokenizer, Mock(), perception.audio_encoder
    )
    hooks.states[identity] = perception
    pcm = np.arange(16000, dtype="<i2").tobytes()
    chunk = TimedChunk(
        "audio", 0, 1000, 0, {"pcm": pcm, "images": [b"frame"]} if has_image else pcm
    )
    payload = StagePayload(
        "unit", OmniRequest(None, params=MiniCPMODuplexSampling().model_dump()), None
    )
    result = hooks.append(chunk, payload, SimpleNamespace(session_identity=identity))
    assert result is payload
    np.testing.assert_array_equal(
        perception.encode_audio.call_args.args[0],
        np.arange(16000, dtype=np.float32) / 32768,
    )
    if has_image:
        perception.encode_image.assert_called_once_with(b"frame")
    else:
        perception.encode_image.assert_not_called()
    adapter = relay.ThinkerAdapter(perception.tokenizer, 100)
    adapter.open(identity, payload.request)
    adapter.states[identity].prefix_pending = first_unit
    request = adapter.build(identity, chunk, payload)
    assert request.prefill_schema == payload.data["prefill_schema"]
    assert len(request.unit_embedding_spans) == (2 if has_image else 1)
    prefix_length = 0 if first_unit else 1
    for actual, planned in zip(
        request.unit_embedding_spans, payload.data["embedding_spans"]
    ):
        assert (actual.start, actual.end) == (
            planned["token_start"] + prefix_length,
            planned["token_end"] + prefix_length,
        )
    rows = relay.splice_embedding_spans(
        torch.full((len(request.input_ids), 4), -1.0), 0, request.unit_embedding_spans
    )
    assert torch.equal(rows[-10:], torch.full((10, 4), 9.0))
    if has_image:
        assert torch.equal(
            rows[prefix_length + 2 : prefix_length + 66], torch.full((64, 4), 6.0)
        )
        assert torch.equal(rows[prefix_length + 66], torch.full((4,), -1.0))
    else:
        pass


@pytest.mark.parametrize("finish", ["complete", "abort"])
def test_image_audio_commit_atomically(
    relay: ModuleType, perception: MiniCPMOPerceptionState, finish: str
) -> None:
    identity = SessionIdentity("vision")
    payload = StagePayload(
        "unit",
        OmniRequest(None, params=MiniCPMODuplexSampling().model_dump()),
        perception.build_step_plan(
            torch.full((10, 4), 9.0), (torch.full((64, 4), 6.0),)
        ),
    )
    adapter = relay.ThinkerAdapter(perception.tokenizer, 100)
    adapter.open(identity, payload.request)
    request = adapter.build(identity, TimedChunk("audio", 0, 1000, 0, b""), payload)
    history = relay.EmbeddingSpan(start=1, end=3, input_embeds=torch.ones(2, 4))
    unit = SimpleNamespace(
        session_identity=identity, session_request=None, embedding_spans=[]
    )
    session = SimpleNamespace(unit=unit, embedding_spans=[history])
    native = Mock(session_id=identity.id)
    native.create_req.return_value = Mock(
        origin_input_ids=[7] * 12 + list(request.req.origin_input_ids), to_finish=None
    )
    bridge = relay.ARSessionBridge.__new__(relay.ARSessionBridge)
    bridge.drain = Mock()
    bridge.sessions = {identity.id: session}
    bridge.units_by_request_id = {"unit": unit}
    bridge.bridge_scheduler = Mock()
    bridge.bridge_scheduler.session_controller.get.return_value = native
    bridge.create_session_request(payload, request)
    assert session.embedding_spans == [history]
    assert [(span.start, span.end) for span in unit.embedding_spans] == [
        (14, 78),
        (79, 89),
    ]
    assert request.session_embedding_spans == [history, *unit.embedding_spans]
    if finish == "complete":
        bridge.complete("unit")
        assert session.embedding_spans == [history, *unit.embedding_spans]
        native.abort_req.assert_not_called()
    else:
        bridge.release_append_unit("unit")
        assert session.embedding_spans == [history]
        native.abort_req.assert_called_once_with()
    assert session.unit is None


@pytest.mark.parametrize(
    "error",
    [
        ValueError("bad frame"),
        OSError("truncated"),
        Image.DecompressionBombError("big"),
    ],
)
def test_undecodable_frame_runs_unit_on_audio(
    relay: ModuleType, perception: MiniCPMOPerceptionState, error: Exception
) -> None:
    identity = SessionIdentity("vision")
    hooks = relay.PerceptionHooks(
        perception.tokenizer, Mock(), perception.audio_encoder
    )
    hooks.states[identity] = perception
    perception.encode_image.side_effect = error
    payload = StagePayload(
        "unit", OmniRequest(None, params=MiniCPMODuplexSampling().model_dump()), None
    )
    hooks.append(
        TimedChunk("audio", 0, 1000, 0, {"pcm": b"\0\0", "images": [b"bad"]}),
        payload,
        SimpleNamespace(session_identity=identity),
    )
    perception.encode_audio.assert_called_once()
    assert [span["modality"] for span in payload.data["embedding_spans"]] == ["audio"]


def test_deployment_grants_image_with_default_limit(relay: ModuleType) -> None:
    relay.build_realtime_deployment(
        Mock(), MiniCPMODuplexPipelineConfig(model_path="unused")
    )
    kwargs = relay.Capabilities.call_args.kwargs
    assert relay.RealtimeDeployment.call_args.args[0] is relay.Capabilities.return_value
    assert kwargs["input_modalities"] == ("audio", "image")
    assert "max_image_bytes" not in kwargs
    assert kwargs["image_frames_per_unit"] == (4, 3, 2, 2, 1, 1, 1, 1, 1)
    assert kwargs["default_max_slice_nums"] == 1


def test_encoders_share_stage_device(relay: ModuleType, tmp_path: Path) -> None:
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"reference")
    relay.AutoTokenizer = Mock()
    relay.AutoProcessor = Mock()
    relay.MiniCPMOAudioEncoder = Mock()
    relay.MiniCPMOImageEncoder = Mock()
    relay.resolve_concrete_device = Mock(return_value="cpu")
    relay.SessionScheduler = Mock()
    relay.create_perception_scheduler(
        "checkpoint",
        device="cpu",
        dtype="float32",
        reference_audio=str(reference),
        max_open_sessions=2,
    )
    relay.MiniCPMOAudioEncoder.assert_called_once_with(
        "checkpoint", device="cpu", dtype="float32"
    )
    relay.MiniCPMOImageEncoder.assert_called_once_with(
        "checkpoint", device="cpu", dtype="float32"
    )
    hooks = relay.SessionScheduler.call_args.args[0]
    assert hooks.image_encoder is relay.MiniCPMOImageEncoder.return_value


def test_empty_eos_does_not_encode(
    relay: ModuleType, perception: MiniCPMOPerceptionState
) -> None:
    hooks = relay.PerceptionHooks(
        perception.tokenizer, Mock(), perception.audio_encoder
    )
    payload = StagePayload(
        "unit", OmniRequest(None, params=MiniCPMODuplexSampling().model_dump()), None
    )
    hooks.append(TimedChunk("audio", 0, 0, 1, None, eos=True), payload, Mock())
    assert payload.data is None
    perception.encode_audio.assert_not_called()
    perception.encode_image.assert_not_called()


def test_bad_frame_does_not_drop_valid_siblings(
    relay: ModuleType, perception: MiniCPMOPerceptionState
) -> None:
    identity = SessionIdentity("vision")
    hooks = relay.PerceptionHooks(
        perception.tokenizer, Mock(), perception.audio_encoder
    )
    hooks.states[identity] = perception
    first = torch.full((64, 4), 3.0)
    last = torch.full((64, 4), 7.0)
    perception.encode_image.side_effect = [first, ValueError("bad frame"), last]
    payload = StagePayload("unit", OmniRequest(None), None)
    hooks.append(
        TimedChunk(
            "audio", 0, 1000, 0, {"pcm": b"\0\0", "images": [b"first", b"bad", b"last"]}
        ),
        payload,
        SimpleNamespace(session_identity=identity),
    )
    spans = payload.data["embedding_spans"]
    assert [span["modality"] for span in spans] == ["image", "image", "audio"]
    assert torch.equal(payload.data["input_embeds"][:128], torch.cat([first, last]))
