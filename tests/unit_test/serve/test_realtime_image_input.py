# SPDX-License-Identifier: Apache-2.0
"""Per-unit image protocol contracts and audio wire compatibility."""

import base64
import json
from dataclasses import replace

import pytest
from pydantic import ValidationError

from sglang_omni.serve.realtime.runtime import SessionRuntime
from sglang_omni.serve.realtime.schema import CLIENT_EVENT, JsonObject
from sglang_omni.serve.realtime.types import Capabilities, ProtocolError, RuntimeLimits
from tests.unit_test.serve.test_realtime_coordinator_adapter import (
    SESSION_CONFIG,
    RecordingSink,
    SessionCoordinator,
    build_adapter,
    build_unit,
)
from tests.unit_test.serve.test_realtime_duplex_session import (
    UNIT_BYTES,
    ScriptedAdapter,
    append_audio,
    build_test_client,
    open_session,
    receive_until,
    send_event,
)

JPEG = b"\xff\xd8frame"
PNG = b"\x89PNGtest"


def image_event(t_ms: float = 0.0, image: bytes = JPEG) -> JsonObject:
    return dict(
        type="sglang.input_image.append",
        event_id="frame",
        image=base64.b64encode(image).decode(),
        sglang=dict(t_ms=t_ms),
    )


@pytest.mark.parametrize("t_ms", [0.0, 19.999, 20.0, 40])
def test_image_schema_accepts_media_time(t_ms: float) -> None:
    event = CLIENT_EVENT.validate_python(image_event(t_ms))
    assert event.sglang.t_ms == t_ms


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"t_ms": "0"},
        {"t_ms": True},
        {"t_ms": None},
        {"t_ms": float("inf")},
        {"t_ms": float("-inf")},
        {"t_ms": float("nan")},
        {"t_ms": 0.0, "seq": 1},
    ],
)
def test_image_schema_rejects_invalid_metadata(metadata: JsonObject) -> None:
    with pytest.raises(ValidationError):
        CLIENT_EVENT.validate_python({**image_event(), "sglang": metadata})


@pytest.mark.parametrize("fields", [{"image": b"abc"}, {"extra": 1}, {"event_id": ""}])
def test_image_schema_rejects_invalid_event(fields: JsonObject) -> None:
    with pytest.raises(ValidationError):
        CLIENT_EVENT.validate_python({**image_event(), **fields})


@pytest.mark.parametrize("image", [JPEG, PNG])
def test_frame_binding_missing_frame_and_accounting(image: bytes) -> None:
    adapter = ScriptedAdapter()
    with build_test_client(
        adapter, capabilities=Capabilities(input_modalities=("audio", "image"))
    ).websocket_connect("/v1/realtime") as websocket:
        open_session(websocket)
        websocket.send_json(image_event(39.999, image))
        accepted = websocket.receive_json()
        assert accepted["type"] == "sglang.input_image.accepted"
        assert accepted["unit_id"] == "unit_1"
        assert accepted["client_event_id"] == "frame"
        append_audio(websocket, b"\0" * UNIT_BYTES * 2, 0)
        receive_until(websocket, "sglang.unit.done")
        receive_until(websocket, "sglang.unit.done")
        send_event(websocket, "sglang.input_audio.end")
        drained = receive_until(websocket, "sglang.input_audio.drained")[-1]
    assert [unit.image for unit in adapter.units] == [None, image, None]
    assert [
        drained[key]
        for key in ("accepted_end_ms", "consumed_ms", "discarded_ms", "padding_ms")
    ] == [40.0, 40.0, 0.0, 0.0]


@pytest.mark.parametrize(
    ("scenario", "code", "message"),
    [
        ("late", "invalid_state", "frame unit already cut"),
        ("duplicate", "invalid_state", "unit already has a frame"),
        ("ahead", "buffer_overflow", None),
        ("ended", "invalid_state", "input has ended"),
        ("unsupported", "not_supported", None),
        ("negative", "invalid_state", "frame unit already cut"),
    ],
)
def test_binding_rejections_are_nonfatal(
    scenario: str, code: str, message: str | None
) -> None:
    capabilities = Capabilities(
        input_modalities=("audio",) if scenario == "unsupported" else ("audio", "image")
    )
    with build_test_client(
        ScriptedAdapter(), capabilities=capabilities
    ).websocket_connect("/v1/realtime") as websocket:
        open_session(websocket)
        if scenario == "late":
            append_audio(websocket, b"\0" * UNIT_BYTES, 0)
            receive_until(websocket, "sglang.unit.done")
        elif scenario == "duplicate":
            websocket.send_json(image_event())
            websocket.receive_json()
        elif scenario == "ended":
            send_event(websocket, "sglang.input_audio.end")
            receive_until(websocket, "sglang.input_audio.drained")
        else:
            pass
        websocket.send_json(
            image_event(
                60.0 if scenario == "ahead" else -0.1 if scenario == "negative" else 0.0
            )
        )
        error = websocket.receive_json()
        assert error["error"]["code"] == code
        assert error["error"]["event_id"] == "frame"
        assert error["sglang"]["fatal"] is False
        if message is not None:
            assert error["error"]["message"] == message
        else:
            pass
        send_event(websocket, "session.update", session={})
        assert websocket.receive_json()["type"] == "session.updated"


def test_lookahead_uses_ceiling_of_accepted_audio_time() -> None:
    with build_test_client(
        ScriptedAdapter(),
        capabilities=Capabilities(input_modalities=("audio", "image")),
    ).websocket_connect("/v1/realtime") as websocket:
        open_session(websocket)
        websocket.send_json(image_event(40.0))
        assert websocket.receive_json()["unit_id"] == "unit_2"
        append_audio(websocket, b"\0\0", 0)
        websocket.receive_json()
        websocket.send_json(image_event(60.0))
        assert websocket.receive_json()["unit_id"] == "unit_3"
        websocket.send_json(image_event(80.0))
        assert websocket.receive_json()["error"]["code"] == "buffer_overflow"


@pytest.mark.parametrize(
    ("encoded", "code"),
    [
        ("!" * 13, "buffer_overflow"),
        ("!!!!", "invalid_request"),
        ("é", "invalid_request"),
        (base64.b64encode(b"GIF89a").decode(), "invalid_request"),
        ("", "invalid_request"),
        (base64.b64encode(b"\xff\xd8" + b"x" * 7).decode(), "buffer_overflow"),
    ],
)
def test_invalid_image_bytes_are_nonfatal(encoded: str, code: str) -> None:
    with build_test_client(
        ScriptedAdapter(),
        capabilities=Capabilities(
            input_modalities=("audio", "image"), max_image_bytes=8
        ),
    ).websocket_connect("/v1/realtime") as websocket:
        open_session(websocket)
        websocket.send_json({**image_event(), "image": encoded})
        error = websocket.receive_json()
        assert error["error"]["code"] == code
        assert error["sglang"]["fatal"] is False
        websocket.send_json(image_event(image=PNG))
        assert websocket.receive_json()["type"] == "sglang.input_image.accepted"


@pytest.mark.asyncio
async def test_clear_and_close_drop_pending_frames_without_audio_accounting() -> None:
    adapter = ScriptedAdapter()
    runtime = SessionRuntime(
        "test",
        Capabilities(input_modalities=("audio", "image")),
        lambda: adapter,
        RuntimeLimits(),
    )
    await runtime.update({}, "update")
    await runtime.append_image(JPEG, 0.0, "frame")
    await runtime.clear("clear")
    assert runtime.pending_frames == {}
    await runtime.append_image(PNG, 0.0, "replacement")
    await runtime.close("client_closed")
    assert runtime.pending_frames == {}
    assert (
        runtime.accepted_samples,
        runtime.consumed_samples,
        runtime.discarded_samples,
        runtime.padding_samples,
    ) == (0, 0, 0, 0)


def test_capabilities_default_and_image_format() -> None:
    defaults = Capabilities().to_granted_capabilities()
    assert defaults["input_modalities"] == ["audio"]
    assert "input_image_format" not in defaults
    image_capabilities = Capabilities(
        input_modalities=("audio", "image"), max_image_bytes=100
    )
    granted = image_capabilities.to_granted_capabilities()
    assert granted == {
        **defaults,
        "input_modalities": ["audio", "image"],
        "input_image_format": {
            "types": ["image/jpeg", "image/png"],
            "max_bytes": 100,
            "max_per_unit": 1,
        },
    }
    assert Capabilities().max_image_bytes == 512 * 1024
    client = build_test_client(ScriptedAdapter(), capabilities=image_capabilities)
    assert (
        client.get("/v1/realtime/capabilities").json()["input_image_format"]
        == granted["input_image_format"]
    )
    with client.websocket_connect("/v1/realtime") as websocket:
        updated = open_session(websocket)
        assert (
            updated["session"]["sglang"]["granted"]["input_image_format"]
            == granted["input_image_format"]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("image", [None, JPEG])
async def test_adapter_payload_preserves_audio_and_bundles_image(
    image: bytes | None,
) -> None:
    coordinator = SessionCoordinator([])
    adapter = build_adapter(coordinator)
    await adapter.open("session", SESSION_CONFIG, RecordingSink())
    unit = replace(build_unit(2, is_eos=True), image=image)
    assert await adapter.process(unit) == unit.real_samples
    await adapter.close()
    chunk = coordinator.appended[0]
    assert chunk.payload == (
        unit.pcm if image is None else {"pcm": unit.pcm, "image": image}
    )
    assert (chunk.modality, chunk.format, chunk.seq, chunk.eos) == (
        "audio",
        "pcm16",
        2,
        True,
    )


def test_audio_only_session_events_carry_no_image_fields() -> None:
    with build_test_client(ScriptedAdapter()).websocket_connect(
        "/v1/realtime"
    ) as websocket:
        events = [websocket.receive_json()]
        send_event(websocket, "session.update", session={})
        events.append(websocket.receive_json())
        append_audio(websocket, b"\0" * UNIT_BYTES, 0)
        events.extend([websocket.receive_json(), websocket.receive_json()])
        send_event(websocket, "sglang.input_audio.end")
        events.extend([websocket.receive_json() for _ in range(3)])
        send_event(websocket, "session.close")
        events.append(websocket.receive_json())
    assert [event["type"] for event in events] == [
        "session.created",
        "session.updated",
        "sglang.input_audio.accepted",
        "sglang.unit.done",
        "sglang.input_audio.ended",
        "sglang.unit.done",
        "sglang.input_audio.drained",
        "session.closed",
    ]
    granted = events[1]["session"]["sglang"]["granted"]
    assert granted["input_modalities"] == ["audio"]
    assert "input_image_format" not in granted
    assert "image" not in json.dumps(events)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("image", "code"),
    [(b"GIF89a", "invalid_request"), (b"\xff\xd8" + b"x" * 7, "buffer_overflow")],
)
async def test_runtime_validates_bytes_without_protocol(
    image: bytes, code: str
) -> None:
    adapter = ScriptedAdapter()
    runtime = SessionRuntime(
        "test",
        Capabilities(input_modalities=("audio", "image"), max_image_bytes=8),
        lambda: adapter,
        RuntimeLimits(),
    )
    await runtime.update({}, "update")
    with pytest.raises(ProtocolError) as error:
        await runtime.append_image(image, 0.0, "frame")
    assert error.value.code == code
    assert runtime.pending_frames == {}
    await runtime.close("client_closed")


def test_clear_prevents_frame_from_reaching_next_unit() -> None:
    adapter = ScriptedAdapter()
    with build_test_client(
        adapter, capabilities=Capabilities(input_modalities=("audio", "image"))
    ).websocket_connect("/v1/realtime") as websocket:
        open_session(websocket)
        websocket.send_json(image_event())
        websocket.receive_json()
        append_audio(websocket, b"\0" * (UNIT_BYTES // 2), 0)
        websocket.receive_json()
        send_event(websocket, "input_audio_buffer.clear")
        assert websocket.receive_json()["sglang"]["discarded_ms"] == 10.0
        append_audio(websocket, b"\0" * UNIT_BYTES, 1)
        receive_until(websocket, "sglang.unit.done")
    assert adapter.units[0].image is None
