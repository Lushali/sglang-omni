# SPDX-License-Identifier: Apache-2.0
"""Deployment sampling defaults and per-session overrides reach the session request."""

from unittest.mock import AsyncMock

import pytest

from sglang_omni.client.client import Client
from sglang_omni.models.minicpm_o.native_config import (
    MiniCPMODuplexPipelineConfig,
    MiniCPMODuplexSampling,
)
from sglang_omni.models.minicpm_o.session_adapters import build_realtime_deployment
from sglang_omni.proto.session import SessionIdentity
from sglang_omni.serve.realtime.negotiation import SessionNegotiation
from sglang_omni.serve.realtime.schema import JsonObject, SamplingConfig

DEFAULT_SAMPLING = {
    "greedy": True,
    "temperature": 0.7,
    "top_k": 100,
    "top_p": 0.8,
    "repetition_penalty": 1.05,
    "listen_prob_scale": 1.0,
    "force_listen_count": 3,
}


async def open_session_params(
    config: MiniCPMODuplexPipelineConfig, sampling: SamplingConfig
) -> JsonObject:
    client = AsyncMock(spec=Client)
    client.open_session.return_value = SessionIdentity("session", 0)
    deployment = build_realtime_deployment(client, config)
    negotiation = SessionNegotiation(
        model="test", capabilities=deployment.capabilities, limits=deployment.limits
    )
    session, _ = negotiation.negotiate(
        {}, "CREATED", {"instructions": "be brief", "sglang": {"sampling": sampling}}
    )
    adapter = deployment.adapter_factory()
    await adapter.open("session", session, AsyncMock())
    try:
        client.open_session.assert_awaited_once()
        return client.open_session.call_args.args[0].params
    finally:
        await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sampling",
    [
        {},
        {
            "temperature": 0.4,
            "top_k": 20,
            "top_p": 0.6,
            "repetition_penalty": 1.2,
            "listen_prob_scale": 0.5,
            "greedy": False,
            "force_listen_count": 0,
        },
    ],
)
async def test_native_session_receives_sampling(sampling: SamplingConfig) -> None:
    params = await open_session_params(
        MiniCPMODuplexPipelineConfig(model_path="unused"), sampling
    )
    assert params == {"instructions": "be brief", **DEFAULT_SAMPLING, **sampling}


@pytest.mark.asyncio
async def test_deployment_defaults_apply_and_session_override_stays_local() -> None:
    config = MiniCPMODuplexPipelineConfig(
        model_path="unused",
        sampling=MiniCPMODuplexSampling(greedy=False, temperature=0.3),
    )
    overridden = await open_session_params(config, {"temperature": 0.9})
    plain = await open_session_params(config, {})
    assert overridden["temperature"] == 0.9
    assert plain["temperature"] == 0.3
    assert plain["greedy"] is False
    assert plain["force_listen_count"] == 3
    assert config.sampling.temperature == 0.3


def test_capabilities_list_every_sampling_default() -> None:
    deployment = build_realtime_deployment(
        AsyncMock(spec=Client), MiniCPMODuplexPipelineConfig(model_path="unused")
    )
    assert set(deployment.capabilities.sampling_parameters) == set(DEFAULT_SAMPLING)
