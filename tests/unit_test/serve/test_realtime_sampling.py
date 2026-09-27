# SPDX-License-Identifier: Apache-2.0
"""Sampling configuration is negotiated before a native session opens."""

import pytest

from sglang_omni.serve.realtime.negotiation import SessionNegotiation
from sglang_omni.serve.realtime.schema import JsonObject
from sglang_omni.serve.realtime.types import Capabilities, ProtocolError, RuntimeLimits


@pytest.mark.parametrize(
    "sampling",
    [
        {"greedy": False, "temperature": 0.7, "top_p": 0.8, "top_k": 20},
        {"force_listen_count": 0, "repetition_penalty": 1.1, "listen_prob_scale": 0.5},
    ],
)
def test_sampling_is_preserved_and_frozen(sampling: JsonObject) -> None:
    negotiation = SessionNegotiation(
        model="test",
        capabilities=Capabilities(sampling_parameters=tuple(sampling)),
        limits=RuntimeLimits(),
    )
    config, granted = negotiation.negotiate(
        {}, "CREATED", {"sglang": {"sampling": sampling}}
    )
    assert config["sglang"]["sampling"] == sampling
    assert granted["sampling_parameters"] == list(sampling)
    with pytest.raises(ProtocolError, match="frozen"):
        negotiation.negotiate(
            config, "OPEN", {"sglang": {"sampling": {"temperature": 0.2}}}
        )


def test_deployment_rejects_unsupported_sampling() -> None:
    negotiation = SessionNegotiation(
        model="test", capabilities=Capabilities(), limits=RuntimeLimits()
    )
    with pytest.raises(ProtocolError) as error:
        negotiation.negotiate(
            {}, "CREATED", {"sglang": {"sampling": {"greedy": False}}}
        )
    assert error.value.code == "not_applicable"
    assert error.value.param == "session.sglang.sampling"


@pytest.mark.parametrize(
    "sampling",
    [
        {"temperature": -1.0},
        {"temperature": float("nan")},
        {"temperature": float("inf")},
        {"temperature": "0.7"},
        {"top_p": 0.0},
        {"top_p": 1.1},
        {"top_k": -2},
        {"top_k": True},
        {"repetition_penalty": 0.0},
        {"listen_prob_scale": -1.0},
        {"force_listen_count": -1},
        {"force_listen_count": 1.5},
        {"greedy": "false"},
        {"length_penalty": 1.1},
    ],
)
def test_invalid_sampling_is_rejected(sampling: JsonObject) -> None:
    negotiation = SessionNegotiation(
        model="test", capabilities=Capabilities(), limits=RuntimeLimits()
    )
    with pytest.raises(ProtocolError) as error:
        negotiation.negotiate({}, "CREATED", {"sglang": {"sampling": sampling}})
    assert error.value.code == "invalid_request"
