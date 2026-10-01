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
    "greedy": False,
    "temperature": 0.7,
    "top_k": 20,
    "top_p": 0.8,
    "repetition_penalty": 1.05,
    "listen_prob_scale": 1.0,
    "force_listen_count": 3,
    "max_new_tokens_per_unit": 20,
    "repetition_window_size": 512,
    "talker_temperature": 0.8,
    "talker_repetition_penalty": 1.05,
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
async def test_session_sampling_overrides_deployment_defaults_locally() -> None:
    config = MiniCPMODuplexPipelineConfig(
        model_path="unused",
        sampling=MiniCPMODuplexSampling(greedy=True, temperature=0.3),
    )
    override = {
        "temperature": 0.4,
        "top_k": 50,
        "top_p": 0.6,
        "repetition_penalty": 1.2,
        "listen_prob_scale": 0.5,
        "greedy": False,
        "force_listen_count": 0,
        "max_new_tokens_per_unit": 8,
        "repetition_window_size": 64,
        "talker_temperature": 0.6,
        "talker_repetition_penalty": 1.1,
    }
    overridden = await open_session_params(config, override)
    plain = await open_session_params(config, {})
    assert overridden == {"instructions": "be brief", **override, "max_slice_nums": 1}
    assert plain == {
        "instructions": "be brief",
        **DEFAULT_SAMPLING,
        "greedy": True,
        "temperature": 0.3,
        "max_slice_nums": 1,
    }
    assert config.sampling.temperature == 0.3
