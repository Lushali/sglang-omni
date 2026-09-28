# SPDX-License-Identifier: Apache-2.0
"""Per-unit image protocol contracts and audio wire compatibility."""

import base64
import json
from dataclasses import replace

import pytest
from pydantic import ValidationError

from sglang_omni.serve.realtime.schema import CLIENT_EVENT, JsonObject
from sglang_omni.serve.realtime.types import Capabilities
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
FRAMES_BY_SLICES = (4, 3, 2, 2, 1, 1, 1, 1, 1)


def image_event(t_ms: float = 0.0, image: bytes = JPEG) -> JsonObject:
    return dict(
        type="sglang.input_image.append",
        event_id="frame",
        image=base64.b64encode(image).decode(),
        sglang=dict(t_ms=t_ms),
    )


@pytest.mark.parametrize(
    "fields",
    [
        {"sglang": {}},
        {"sglang": {"t_ms": "0"}},
        {"sglang": {"t_ms": float("nan")}},
        {"sglang": {"t_ms": 0.0, "seq": 1}},
        {"image": b"abc"},
        {"extra": 1},
    ],
)
def test_image_schema_rejects_invalid_event(fields: JsonObject) -> None:
    with pytest.raises(ValidationError):
        CLIENT_EVENT.validate_python({**image_event(), **fields})


def test_frame_binding_missing_frame_and_accounting() -> None:
    adapter = ScriptedAdapter()
    with build_test_client(
        adapter, capabilities=Capabilities(input_modalities=("audio", "image"))
    ).websocket_connect("/v1/realtime") as websocket:
        open_session(websocket)
        websocket.send_json(image_event(39.999))
        accepted = websocket.receive_json()
        assert accepted["type"] == "sglang.input_image.accepted"
        assert accepted["unit_id"] == "unit_1"
        assert accepted["client_event_id"] == "frame"
        append_audio(websocket, b"\0" * UNIT_BYTES * 2, 0)
        receive_until(websocket, "sglang.unit.done")
        receive_until(websocket, "sglang.unit.done")
        send_event(websocket, "sglang.input_audio.end")
        drained = receive_until(websocket, "sglang.input_audio.drained")[-1]
    assert [unit.images for unit in adapter.units] == [(), (JPEG,), ()]
    assert [
        drained[key]
        for key in ("accepted_end_ms", "consumed_ms", "discarded_ms", "padding_ms")
    ] == [40.0, 40.0, 0.0, 0.0]


@pytest.mark.parametrize(
    ("scenario", "code", "message"),
    [
        ("late", "invalid_state", "frame unit already cut"),
        ("ended", "invalid_state", "input has ended"),
        ("unsupported", "not_supported", "image input is not granted"),
    ],
)
def test_binding_rejections_are_nonfatal(
    scenario: str, code: str, message: str
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
        elif scenario == "ended":
            send_event(websocket, "sglang.input_audio.end")
            receive_until(websocket, "sglang.input_audio.drained")
        else:
            pass
        websocket.send_json(image_event())
        error = websocket.receive_json()
        assert error["error"]["code"] == code
        assert error["error"]["event_id"] == "frame"
        assert error["sglang"]["fatal"] is False
        assert error["error"]["message"] == message
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
        (base64.b64encode(b"GIF89a").decode(), "invalid_request"),
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


def test_image_capability_is_granted_over_http_and_session() -> None:
    image_format = {
        "types": ["image/jpeg", "image/png"],
        "max_bytes": 100,
        "max_per_unit": 1,
        "max_slice_nums": 1,
    }
    client = build_test_client(
        ScriptedAdapter(),
        capabilities=Capabilities(
            input_modalities=("audio", "image"), max_image_bytes=100
        ),
    )
    assert (
        client.get("/v1/realtime/capabilities").json()["input_image_format"]
        == image_format
    )
    with client.websocket_connect("/v1/realtime") as websocket:
        granted = open_session(websocket)["session"]["sglang"]["granted"]
    assert granted["input_modalities"] == ["audio", "image"]
    assert granted["input_image_format"] == image_format


@pytest.mark.asyncio
@pytest.mark.parametrize("image", [None, JPEG])
async def test_adapter_payload_preserves_audio_and_bundles_image(
    image: bytes | None,
) -> None:
    coordinator = SessionCoordinator([])
    adapter = build_adapter(coordinator)
    await adapter.open("session", SESSION_CONFIG, RecordingSink())
    unit = replace(build_unit(2, is_eos=True), images=() if image is None else (image,))
    assert await adapter.process(unit) == unit.real_samples
    await adapter.close()
    chunk = coordinator.appended[0]
    assert chunk.payload == (
        unit.pcm if image is None else {"pcm": unit.pcm, "images": [image]}
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


def test_clear_drops_frames_and_moves_frame_origin() -> None:
    adapter = ScriptedAdapter()
    with build_test_client(
        adapter, capabilities=Capabilities(input_modalities=("audio", "image"))
    ).websocket_connect("/v1/realtime") as websocket:
        open_session(websocket)
        websocket.send_json(image_event(0.0, JPEG))
        assert websocket.receive_json()["unit_id"] == "unit_0"
        for sequence in range(10):
            append_audio(websocket, b"\0" * (UNIT_BYTES // 2), sequence)
            send_event(websocket, "input_audio_buffer.clear")
            cleared = receive_until(websocket, "input_audio_buffer.cleared")[-1]
            assert cleared["sglang"]["discarded_ms"] == 10.0
        websocket.send_json(image_event(80.0, JPEG))
        assert websocket.receive_json()["error"]["code"] == "invalid_state"
        websocket.send_json(image_event(160.0, JPEG))
        assert websocket.receive_json()["error"]["code"] == "buffer_overflow"
        websocket.send_json(image_event(140.0, JPEG))
        assert websocket.receive_json()["unit_id"] == "unit_2"
        websocket.send_json(image_event(100.0, PNG))
        assert websocket.receive_json()["unit_id"] == "unit_0"
        append_audio(websocket, b"\0" * UNIT_BYTES, 10)
        receive_until(websocket, "sglang.unit.done")
    assert adapter.units[0].images == (PNG,)


@pytest.mark.parametrize(
    ("frames", "expected"),
    [
        ([(15.0, PNG), (5.0, JPEG)], (JPEG, PNG)),
        ([(0.0, PNG), (0.0, JPEG)], (PNG, JPEG)),
    ],
)
def test_unit_frames_follow_slice_grant_and_media_time(
    frames: list[tuple[float, bytes]], expected: tuple[bytes, ...]
) -> None:
    adapter = ScriptedAdapter()
    capabilities = Capabilities(
        input_modalities=("audio", "image"),
        image_frames_per_unit=FRAMES_BY_SLICES,
    )
    with build_test_client(adapter, capabilities=capabilities).websocket_connect(
        "/v1/realtime"
    ) as websocket:
        websocket.receive_json()
        send_event(
            websocket, "session.update", session={"sglang": {"max_slice_nums": 4}}
        )
        granted = websocket.receive_json()["session"]["sglang"]["granted"]
        assert granted["input_image_format"]["max_per_unit"] == 2
        for t_ms, image in frames:
            websocket.send_json(image_event(t_ms, image))
            assert websocket.receive_json()["type"] == "sglang.input_image.accepted"
        websocket.send_json(image_event(10.0, JPEG))
        assert websocket.receive_json()["error"]["code"] == "buffer_overflow"
        append_audio(websocket, b"\0" * UNIT_BYTES, 0)
        receive_until(websocket, "sglang.unit.done")
    assert adapter.units[0].images == expected
