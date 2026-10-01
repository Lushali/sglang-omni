# SPDX-License-Identifier: Apache-2.0
"""Reference audio is bounded and immutable after native session admission."""

import base64
import io
import json
import wave

import pytest

from sglang_omni.serve.realtime.negotiation import SessionNegotiation
from sglang_omni.serve.realtime.reference_audio import MAX_REFERENCE_AUDIO_BYTES
from sglang_omni.serve.realtime.schema import JsonValue, SessionConfiguration
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


WAV = base64.b64decode(wav_reference())
NEGOTIATION = SessionNegotiation(
    model="test",
    capabilities=Capabilities(supports_reference_audio=True),
    limits=RuntimeLimits(),
)


def negotiate_reference(field: str, data: JsonValue) -> SessionConfiguration:
    config, _ = NEGOTIATION.negotiate(
        {}, "CREATED", {"sglang": {field: {"media_type": "audio/wav", "data": data}}}
    )
    return config


@pytest.mark.parametrize(
    "data",
    [
        pytest.param("%%%%", id="not_base64"),
        pytest.param(
            base64.b64encode(bytes(MAX_REFERENCE_AUDIO_BYTES + 1)).decode("ascii"),
            id="over_byte_limit",
        ),
        pytest.param(
            base64.b64encode(b"RIFF" + (4).to_bytes(4, "little") + b"WAVE").decode(),
            id="header_only",
        ),
        pytest.param(
            base64.b64encode(
                b"RIFF" + (12).to_bytes(4, "little") + b"WAVEJUNK" + b"\xff" * 4
            ).decode(),
            id="chunk_overflow",
        ),
        pytest.param(wav_reference(7999, 160), id="rate_below_8k"),
        pytest.param(wav_reference(8000, 240001), id="over_30_seconds"),
        pytest.param(wav_reference(16000, 0), id="empty"),
    ],
)
def test_invalid_reference_is_rejected(data: JsonValue) -> None:
    with pytest.raises(ProtocolError) as error:
        negotiate_reference("reference_audio", data)
    assert error.value.code == "invalid_request"


@pytest.mark.parametrize(
    ("audio", "expected"),
    [
        pytest.param(
            base64.b64decode(wav_reference(8000, 240000)),
            base64.b64decode(wav_reference(8000, 240000)),
            id="30_seconds_at_8k",
        ),
        pytest.param(
            WAV[:4] + b"\xff" * 4 + WAV[8:40] + b"\xff" * 4 + WAV[44:],
            WAV,
            id="placeholder_sizes",
        ),
        pytest.param(
            WAV[:4] + (len(WAV) - 9).to_bytes(4, "little") + WAV[8:-1],
            base64.b64decode(wav_reference(frames=159)),
            id="odd_tail",
        ),
        pytest.param(
            WAV[:4]
            + (len(WAV) + 16).to_bytes(4, "little")
            + WAV[8:]
            + WAV[12:24]
            + (1).to_bytes(4, "little")
            + WAV[28:36],
            WAV,
            id="later_format_chunk",
        ),
    ],
)
def test_reference_is_normalized_to_canonical_wav(
    audio: bytes, expected: bytes
) -> None:
    config = negotiate_reference(
        "tts_reference_audio", base64.b64encode(audio).decode()
    )
    assert config["sglang"]["tts_reference_audio"].data == expected


def test_reference_is_never_echoed() -> None:
    reference = {"media_type": "audio/wav", "data": wav_reference(8000, 240000)}
    with build_test_client(
        ScriptedAdapter(), capabilities=Capabilities(supports_reference_audio=True)
    ).websocket_connect("/v1/realtime") as websocket:
        websocket.receive_json()
        send_event(
            websocket,
            "session.update",
            session={
                "sglang": {
                    "reference_audio": reference,
                    "tts_reference_audio": reference,
                }
            },
        )
        updated = websocket.receive_json()
    assert updated["type"] == "session.updated"
    assert len(json.dumps(updated)) < 4096
