# SPDX-License-Identifier: Apache-2.0
"""Reference audio is bounded and immutable after native session admission."""

import base64
import io
import json
import wave

import pytest

from sglang_omni.serve.realtime.negotiation import SessionNegotiation
from sglang_omni.serve.realtime.reference_audio import MAX_REFERENCE_AUDIO_BYTES
from sglang_omni.serve.realtime.schema import JsonValue
from sglang_omni.serve.realtime.types import Capabilities, ProtocolError, RuntimeLimits
from tests.unit_test.serve.test_realtime_duplex_session import (
    ScriptedAdapter,
    build_test_client,
    send_event,
)


def wav_reference(rate: int = 16000, frames: int = 160) -> str:
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(bytes(frames * 2))
    return base64.b64encode(output.getvalue()).decode("ascii")


@pytest.mark.parametrize("field", ["reference_audio", "tts_reference_audio"])
def test_reference_is_negotiated_and_frozen(field: str) -> None:
    negotiation = SessionNegotiation(
        model="test",
        capabilities=Capabilities(supports_reference_audio=True),
        limits=RuntimeLimits(),
    )
    reference = {"media_type": "audio/wav", "data": wav_reference()}
    config, granted = negotiation.negotiate(
        {}, "CREATED", {"sglang": {field: reference}}
    )
    assert config["sglang"][field].data == base64.b64decode(reference["data"])
    assert granted["supports_reference_audio"] is True
    with pytest.raises(ProtocolError, match="frozen"):
        negotiation.negotiate(
            config,
            "OPEN",
            {
                "sglang": {
                    field: {"media_type": "audio/wav", "data": wav_reference(frames=80)}
                }
            },
        )


@pytest.mark.parametrize(
    "data",
    [
        "%%%%",
        "AAAA",
        "/tmp/reference.wav",
        "https://example.com/reference.wav",
        base64.b64encode(bytes(MAX_REFERENCE_AUDIO_BYTES + 1)).decode("ascii"),
        5,
        None,
        ["AAAA"],
    ],
)
def test_invalid_reference_is_rejected(data: JsonValue) -> None:
    negotiation = SessionNegotiation(
        model="test",
        capabilities=Capabilities(supports_reference_audio=True),
        limits=RuntimeLimits(),
    )
    with pytest.raises(ProtocolError) as error:
        negotiation.negotiate(
            {},
            "CREATED",
            {"sglang": {"reference_audio": {"media_type": "audio/wav", "data": data}}},
        )
    assert error.value.code == "invalid_request"


def test_reference_is_rejected_without_capability() -> None:
    negotiation = SessionNegotiation(
        model="test", capabilities=Capabilities(), limits=RuntimeLimits()
    )
    with pytest.raises(ProtocolError) as error:
        negotiation.negotiate(
            {},
            "CREATED",
            {
                "sglang": {
                    "reference_audio": {
                        "media_type": "audio/wav",
                        "data": wav_reference(),
                    }
                }
            },
        )
    assert error.value.code == "not_applicable"


@pytest.mark.parametrize("field", ["reference_audio", "tts_reference_audio"])
@pytest.mark.parametrize("rate,frames", [(1, 100), (8000, 240001), (16000, 0)])
def test_unbounded_or_empty_reference_is_rejected(
    field: str, rate: int, frames: int
) -> None:
    negotiation = SessionNegotiation(
        model="test",
        capabilities=Capabilities(supports_reference_audio=True),
        limits=RuntimeLimits(),
    )
    with pytest.raises(ProtocolError) as error:
        negotiation.negotiate(
            {},
            "CREATED",
            {
                "sglang": {
                    field: {
                        "media_type": "audio/wav",
                        "data": wav_reference(rate, frames),
                    }
                }
            },
        )
    assert error.value.code == "invalid_request"


@pytest.mark.parametrize("damage", ["zero_rate", "header_only", "chunk_overflow"])
def test_malformed_wav_is_rejected(damage: str) -> None:
    audio = bytearray(base64.b64decode(wav_reference()))
    if damage == "zero_rate":
        audio[24:28] = bytes(4)
    elif damage == "chunk_overflow":
        audio = bytearray(
            b"RIFF" + (12).to_bytes(4, "little") + b"WAVEJUNK" + b"\xff" * 4
        )
    else:
        audio = bytearray(b"RIFF" + (4).to_bytes(4, "little") + b"WAVE")
    negotiation = SessionNegotiation(
        model="test",
        capabilities=Capabilities(supports_reference_audio=True),
        limits=RuntimeLimits(),
    )
    with pytest.raises(ProtocolError):
        negotiation.negotiate(
            {},
            "CREATED",
            {
                "sglang": {
                    "reference_audio": {
                        "media_type": "audio/wav",
                        "data": base64.b64encode(audio).decode(),
                    }
                }
            },
        )


@pytest.mark.parametrize("rate,frames", [(8000, 240000), (48000, 160)])
def test_reference_rate_and_duration_boundaries_are_supported(
    rate: int, frames: int
) -> None:
    negotiation = SessionNegotiation(
        model="test",
        capabilities=Capabilities(supports_reference_audio=True),
        limits=RuntimeLimits(),
    )
    reference = wav_reference(rate, frames)
    config, _ = negotiation.negotiate(
        {},
        "CREATED",
        {"sglang": {"reference_audio": {"media_type": "audio/wav", "data": reference}}},
    )
    assert config["sglang"]["reference_audio"].data == base64.b64decode(reference)


def test_updates_never_echo_references_and_do_not_decode_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = wav_reference(8000, 240000)
    with build_test_client(
        ScriptedAdapter(), capabilities=Capabilities(supports_reference_audio=True)
    ).websocket_connect("/v1/realtime") as websocket:
        websocket.receive_json()
        send_event(
            websocket,
            "session.update",
            session={
                "sglang": {
                    field: {"media_type": "audio/wav", "data": reference}
                    for field in ("reference_audio", "tts_reference_audio")
                }
            },
        )
        updated = websocket.receive_json()
        assert updated["type"] == "session.updated"
        assert "reference_audio" not in updated["session"]["sglang"]
        assert "tts_reference_audio" not in updated["session"]["sglang"]
        assert len(json.dumps(updated)) < 4096

        def unexpected_decode(*args: str, **kwargs: bool) -> bytes:
            raise AssertionError("unchanged references must not be decoded again")

        monkeypatch.setattr(base64, "b64decode", unexpected_decode)
        send_event(websocket, "session.update", session={"output_modalities": ["text"]})
        updated = websocket.receive_json()
        assert updated["type"] == "session.updated"
        assert len(json.dumps(updated)) < 4096


def test_invalid_reference_does_not_open_adapter_and_session_can_retry() -> None:
    adapter = ScriptedAdapter()
    with build_test_client(
        adapter, capabilities=Capabilities(supports_reference_audio=True)
    ).websocket_connect("/v1/realtime") as websocket:
        websocket.receive_json()
        send_event(
            websocket,
            "session.update",
            session={
                "sglang": {
                    "reference_audio": {
                        "media_type": "audio/wav",
                        "data": wav_reference(1, 100),
                    }
                }
            },
        )
        error = websocket.receive_json()
        assert error["error"]["code"] == "invalid_request"
        assert error["sglang"]["fatal"] is False
        assert adapter.output_sink is None
        send_event(websocket, "session.update", session={})
        assert websocket.receive_json()["type"] == "session.updated"


@pytest.mark.parametrize("damage", ["placeholder_sizes", "odd_tail"])
def test_header_sizes_yield_to_samples_present(damage: str) -> None:
    audio = bytearray(base64.b64decode(wav_reference(frames=160)))
    if damage == "placeholder_sizes":
        audio[4:8] = audio[40:44] = (0xFFFFFFFF).to_bytes(4, "little")
        frames = 160
    else:
        audio.pop()
        audio[4:8] = (len(audio) - 8).to_bytes(4, "little")
        frames = 159
    negotiation = SessionNegotiation(
        model="test",
        capabilities=Capabilities(supports_reference_audio=True),
        limits=RuntimeLimits(),
    )
    config, _ = negotiation.negotiate(
        {},
        "CREATED",
        {
            "sglang": {
                "reference_audio": {
                    "media_type": "audio/wav",
                    "data": base64.b64encode(audio).decode(),
                }
            }
        },
    )
    assert config["sglang"]["reference_audio"].data == base64.b64decode(
        wav_reference(frames=frames)
    )


def test_later_format_chunks_cannot_override_validated_audio() -> None:
    audio = bytearray(base64.b64decode(wav_reference()))
    later_format = bytearray(audio[12:36])
    later_format[12:16] = (1).to_bytes(4, "little")
    audio.extend(later_format)
    audio[4:8] = (len(audio) - 8).to_bytes(4, "little")
    negotiation = SessionNegotiation(
        model="test",
        capabilities=Capabilities(supports_reference_audio=True),
        limits=RuntimeLimits(),
    )
    config, _ = negotiation.negotiate(
        {},
        "CREATED",
        {
            "sglang": {
                "reference_audio": {
                    "media_type": "audio/wav",
                    "data": base64.b64encode(audio).decode(),
                }
            }
        },
    )
    assert config["sglang"]["reference_audio"].data == base64.b64decode(wav_reference())
