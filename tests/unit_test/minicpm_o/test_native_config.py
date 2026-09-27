# SPDX-License-Identifier: Apache-2.0
"""Native config loading must not import checkpoint Python through HF blob links."""

from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from transformers import AutoConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.runtime import (
    apply_typed_stage_kwargs,
    resolve_stage_typed_kwargs,
)
from sglang_omni.models.minicpm_o import engine_builder, native_stages, stages
from sglang_omni.models.minicpm_o.components import audio_encoder, image_encoder
from sglang_omni.models.minicpm_o.hf_config import MiniCPMOConfig
from sglang_omni.models.minicpm_o.session_adapters import build_realtime_deployment
from sglang_omni.scheduling import sglang_backend
from sglang_omni.scheduling.session import SessionHooks


class ConfigLoaded(Exception):
    """Stop at the configuration boundary before allocating any model or GPU."""


@pytest.fixture
def snapshot(tmp_path: Path) -> Path:
    config = {
        "model_type": "minicpmo",
        "architectures": ["MiniCPMO"],
        "auto_map": {"AutoConfig": "configuration_minicpmo.MiniCPMOConfig"},
        "attention_bias": False,
        "hidden_size": 64,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "num_hidden_layers": 1,
        "vision_config": {"hidden_size": 32},
        "audio_config": {"d_model": 32},
        "tts_config": {"hidden_size": 16},
    }
    files = {
        "config.json": json.dumps(config),
        "configuration_minicpmo.py": "from .modeling_navit_siglip import Config\n",
        "modeling_navit_siglip.py": "class Config: pass\n",
    }
    blobs = tmp_path / "blobs"
    snapshot = tmp_path / "snapshots" / "revision"
    blobs.mkdir()
    snapshot.mkdir(parents=True)
    for name, contents in files.items():
        blob = blobs / hashlib.sha256(contents.encode()).hexdigest()
        blob.write_text(contents)
        (snapshot / name).symlink_to(blob)
    return snapshot


@pytest.mark.parametrize("encoder", ["image", "audio"])
def test_encoder_loads_native_config_from_snapshot_links(
    encoder: str, snapshot: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def stop_before_weights(*args):
        raise ConfigLoaded

    if encoder == "image":
        monkeypatch.setattr(image_encoder, "init_sglang_tp", stop_before_weights)
        constructor = image_encoder.MiniCPMOImageEncoder
    else:
        monkeypatch.setattr(audio_encoder, "audio_config_object", stop_before_weights)
        constructor = audio_encoder.MiniCPMOAudioEncoder

    with pytest.raises(ConfigLoaded):
        constructor(str(snapshot), device="cpu")


def test_native_config_preserves_component_dictionaries(snapshot: Path) -> None:
    config = MiniCPMOConfig.from_pretrained(snapshot)
    raw = json.loads((snapshot / "config.json").read_text())
    for name in ("vision_config", "audio_config", "tts_config"):
        assert getattr(config, name) == raw[name]
    assert image_encoder.vision_config_object(config).hidden_size == 32
    assert audio_encoder.audio_config_object(config).d_model == 32
    assert config.get_text_config().hidden_size == 64


@pytest.mark.parametrize("stage", ["thinker", "talker"])
@pytest.mark.parametrize("trust_override", [None, False, True])
def test_engine_factory_resolves_native_config_before_server_args(
    stage: str,
    trust_override: bool | None,
    snapshot: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = dict(
        CONFIG_MAPPING._extra_content
    )  # noqa: leading-underscore  # upstream name
    mapping.pop("minicpmo", None)
    monkeypatch.setattr(CONFIG_MAPPING, "_extra_content", mapping)
    monkeypatch.setattr(stages, "resolved_view", lambda args: args)

    def build_overrides(*, server_args_overrides=None, **defaults):
        return {**defaults, **(server_args_overrides or {})}

    def build_server_args(model_path, **kwargs):
        trust = kwargs.get("trust_remote_code", True)
        if trust_override is True:
            assert trust is True, "An explicit remote-code override must be preserved"
        else:
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust)
            assert isinstance(config, MiniCPMOConfig)
        raise ConfigLoaded

    monkeypatch.setattr(stages, "build_generation_batch_overrides", build_overrides)
    monkeypatch.setattr(
        stages, "validate_generation_batch_policy", lambda **kwargs: None
    )
    monkeypatch.setattr(stages, "build_sglang_server_args", build_server_args)
    factory = getattr(stages, f"create_sglang_{stage}_executor_from_config")
    overrides = {} if trust_override is None else {"trust_remote_code": trust_override}
    with pytest.raises(ConfigLoaded):
        factory(str(snapshot), server_args_overrides=overrides)


@pytest.mark.parametrize(
    ("settings", "sessions", "state_bytes", "thinker", "talker"),
    [
        ("", 2, 4 << 30, 4, 32),
        ("max_sessions: 8\n", 8, 16 << 30, 9, 32),
        (
            "max_sessions: 64\nspeech_state_bytes_per_session: 1024\n",
            64,
            65536,
            65,
            65,
        ),
    ],
)
def test_duplex_yaml_session_limits(
    settings: str,
    sessions: int,
    state_bytes: int,
    thinker: int,
    talker: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference_path = tmp_path / "reference.wav"
    reference_path.write_bytes(b"reference")
    config_path = tmp_path / "duplex.yaml"
    config_path.write_text(
        "config_cls: MiniCPMODuplexPipelineConfig\n"
        f"model_path: unused\nreference_audio: {reference_path}\n" + settings
    )
    config = ConfigManager.from_file(str(config_path)).config
    hooks = SessionHooks()
    monkeypatch.setattr(native_stages.AutoTokenizer, "from_pretrained", Mock())
    monkeypatch.setattr(native_stages, "MiniCPMOAudioEncoder", Mock())
    monkeypatch.setattr(native_stages, "MiniCPMOImageEncoder", Mock())
    monkeypatch.setattr(native_stages, "PerceptionHooks", Mock(return_value=hooks))
    monkeypatch.setattr(
        native_stages,
        "MiniCPMOCode2Wav",
        Mock(return_value=Mock(default_prompt_wav=str(reference_path))),
    )
    monkeypatch.setattr(native_stages, "MiniCPMOVocoderRuntime", Mock())
    monkeypatch.setattr(native_stages, "SpeechHooks", Mock(return_value=hooks))
    perception = native_stages.create_perception_scheduler(
        config.model_path, device="cpu", **config.stage_factory_kwargs("perception")
    )
    speech = native_stages.create_speech_scheduler(
        config.model_path, device="cpu", **config.stage_factory_kwargs("speech")
    )
    for scheduler in (perception, speech):
        assert scheduler.max_open_sessions == sessions
        assert scheduler.max_concurrency == 1
    assert speech.max_state_bytes == state_bytes
    assert build_realtime_deployment(Mock(), config).max_connections == sessions
    assert config.stage_factory_kwargs("thinker")["server_args_overrides"] == {
        "max_running_requests": thinker
    }
    assert config.stage_factory_kwargs("talker")["server_args_overrides"] == {
        "max_running_requests": talker
    }


@pytest.mark.parametrize("stage_name", ["thinker", "talker"])
def test_duplex_engine_override_wins(stage_name: str, tmp_path: Path) -> None:
    config_path = tmp_path / "duplex.yaml"
    config_path.write_text(
        "config_cls: MiniCPMODuplexPipelineConfig\nmodel_path: unused\n"
        "max_sessions: 8\nstages:\n"
        f"  {stage_name}:\n    engine:\n      max_running_requests: 3\n"
    )
    config = ConfigManager.from_file(str(config_path)).config
    stage = config.stage_named(stage_name)
    kwargs = apply_typed_stage_kwargs(
        native_stages.create_thinker_scheduler,
        config.stage_factory_kwargs(stage_name),
        resolve_stage_typed_kwargs(stage),
        stage_name=stage_name,
    )
    assert kwargs["server_args_overrides"]["max_running_requests"] == 3


@pytest.mark.parametrize("context_length", [None, 32768])
def test_native_thinker_context_length(
    context_length: int | None, snapshot: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(engine_builder, "get_tokenizer", Mock())
    server_args = Mock(side_effect=ConfigLoaded)
    monkeypatch.setattr(sglang_backend, "build_sglang_server_args", server_args)
    builder = engine_builder.MiniCPMOThinkerEngineBuilder()
    overrides = {} if context_length is None else {"context_length": context_length}
    with pytest.raises(ConfigLoaded):
        builder.build(str(snapshot), device="cpu", server_args_overrides=overrides)
    assert server_args.call_args.kwargs["context_length"] == (context_length or 8192)


def test_native_engine_factories_declare_the_placement_fraction() -> None:
    for factory in (
        native_stages.create_thinker_scheduler,
        native_stages.create_talker_scheduler,
    ):
        assert "total_gpu_memory_fraction" in inspect.signature(factory).parameters


def test_minicpmo_configs_load_without_sglang(tmp_path: Path) -> None:
    config_path = tmp_path / "duplex.yaml"
    script = """
import sys
from pathlib import Path
sys.modules["sglang"] = None
from sglang_omni.config.manager import ConfigManager
for name in ("MiniCPMODuplexPipelineConfig", "MiniCPMOPipelineConfig", "MiniCPMOSpeechPipelineConfig"):
    Path(sys.argv[1]).write_text(f"config_cls: {name}\\nmodel_path: unused\\n")
    config = ConfigManager.from_file(sys.argv[1]).config
    assert type(config).__name__ == name
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(config_path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
