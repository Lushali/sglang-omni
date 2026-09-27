# SPDX-License-Identifier: Apache-2.0
"""Per-session references reach the model without replacing deployment defaults."""

import base64
import io
import wave
from unittest.mock import AsyncMock, Mock

import pytest

from sglang_omni.client.client import Client
from sglang_omni.models.minicpm_o.components.tts_runtime import MiniCPMOVocoderRuntime
from sglang_omni.models.minicpm_o.native_config import MiniCPMODuplexPipelineConfig
from sglang_omni.models.minicpm_o.native_stages import PerceptionHooks, SpeechHooks
from sglang_omni.models.minicpm_o.session_adapters import build_realtime_deployment
from sglang_omni.proto.request import OmniRequest
from sglang_omni.proto.session import SessionIdentity
from sglang_omni.serve.realtime.schema import SessionConfiguration, SessionUpdateRequest


@pytest.mark.asyncio
async def test_reference_bytes_reach_session_request() -> None:
    client = AsyncMock(spec=Client)
    client.open_session.return_value = SessionIdentity("voice")
    deployment = build_realtime_deployment(
        client, MiniCPMODuplexPipelineConfig(model_path="unused")
    )
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(bytes(320))
    reference = output.getvalue()
    config: SessionConfiguration = {
        "audio": {"input": {"format": {"type": "audio/pcm", "rate": 16000}}},
        "sglang": {
            "reference_audio": {
                "media_type": "audio/wav",
                "data": base64.b64encode(reference).decode(),
            },
            "tts_reference_audio": {
                "media_type": "audio/wav",
                "data": base64.b64encode(reference).decode(),
            },
        },
    }
    adapter = deployment.adapter_factory()
    await adapter.open(
        "voice", SessionUpdateRequest(session=config).session, AsyncMock()
    )
    try:
        request = client.open_session.call_args.args[0]
        assert request.params["reference_audio"] == reference
        assert request.params["tts_reference_audio"] == reference
    finally:
        await adapter.close()


def test_perception_reference_does_not_replace_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_open = Mock()
    monkeypatch.setattr(
        "sglang_omni.models.minicpm_o.native_stages.MiniCPMOPerceptionState.open",
        state_open,
    )
    hooks = PerceptionHooks(Mock(), Mock(), Mock(), reference_audio=b"default")
    hooks.open(
        SessionIdentity("custom"),
        OmniRequest(None, params={"reference_audio": b"custom"}),
    )
    hooks.open(SessionIdentity("default"), OmniRequest(None))
    assert [call.kwargs["reference_audio"] for call in state_open.call_args_list] == [
        b"custom",
        b"default",
    ]


def test_speech_reference_precedence_is_per_session() -> None:
    runtime = Mock()
    hooks = SpeechHooks(runtime, b"default")
    for session_id, params in (
        ("separate", {"reference_audio": b"input", "tts_reference_audio": b"output"}),
        ("shared", {"reference_audio": b"input"}),
        ("default", {}),
    ):
        hooks.open(SessionIdentity(session_id), OmniRequest(None, params=params))
    assert [
        call.kwargs["prompt_wav"] for call in runtime.open_session.call_args_list
    ] == [b"output", b"input", b"default"]


def test_reference_prompt_uses_memory_stream() -> None:
    token2wav = Mock()
    token2wav.prepare_prompt.side_effect = ValueError("invalid audio")
    runtime = MiniCPMOVocoderRuntime(token2wav)
    with pytest.raises(ValueError, match="invalid audio"):
        runtime.open_session("voice", prompt_wav=b"invalid audio")
    stream = token2wav.prepare_prompt.call_args.args[0]
    assert stream.read() == b"invalid audio"
