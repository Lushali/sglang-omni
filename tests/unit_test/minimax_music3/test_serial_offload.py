# SPDX-License-Identifier: Apache-2.0
"""Serial offload coordinator behind --stage-offload-components ar,dit."""

from __future__ import annotations

import pytest
import torch
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.models.minimax_music3.acoustic import MiniMaxMusic3AcousticScheduler
from sglang_omni.models.minimax_music3.engine_builder import MiniMaxMusic3EngineBuilder
from sglang_omni.models.minimax_music3.scheduler import MiniMaxMusic3Scheduler
from sglang_omni.models.minimax_music3.serial_offload import (
    STALL_REPORT_SECONDS,
    SerialOffloadCoordinator,
    StageResidency,
    get_coordinator,
)
from sglang_omni.scheduling.generation_batch_policy import (
    build_generation_batch_overrides,
)


@pytest.mark.parametrize("enable_serial_offload", [False, True])
def test_offload_enforces_one_cfg_pair_and_eager_execution(
    enable_serial_offload: bool,
) -> None:
    builder = MiniMaxMusic3EngineBuilder(enable_serial_offload=enable_serial_offload)
    overrides = build_generation_batch_overrides(
        server_args_overrides={
            "max_running_requests": 16,
            "disable_cuda_graph": False,
            "cuda_graph_backend_decode": "full",
            "cuda_graph_backend_prefill": "full",
            "cuda_graph_config": {
                "decode": {"backend": "full"},
                "prefill": {"backend": "full"},
            },
            "enable_torch_compile": True,
            "disable_overlap_schedule": False,
        },
        **builder.generation_defaults(dtype="bfloat16"),
    )

    builder.adjust_overrides(overrides)

    if enable_serial_offload:
        assert builder.max_running_requests == 1
        assert overrides["max_running_requests"] == 2
        assert overrides["disable_cuda_graph"] is True
        assert overrides["cuda_graph_backend_decode"] == "disabled"
        assert overrides["cuda_graph_backend_prefill"] == "disabled"
        assert "cuda_graph_config" not in overrides
        assert overrides["enable_torch_compile"] is False
        assert overrides["disable_overlap_schedule"] is True
    else:
        assert builder.max_running_requests == 16
        assert overrides["max_running_requests"] == 32
        assert overrides["disable_cuda_graph"] is False
        assert overrides["cuda_graph_backend_decode"] == "full"
        assert overrides["cuda_graph_backend_prefill"] == "full"
        assert "cuda_graph_config" in overrides
        assert overrides["enable_torch_compile"] is True
        assert overrides["disable_overlap_schedule"] is False


def registered_coordinator() -> SerialOffloadCoordinator:
    coordinator = SerialOffloadCoordinator()
    coordinator.register_ar(torch.nn.Linear(2, 2), torch.device("cpu"))
    return coordinator


def test_get_coordinator_returns_a_process_wide_singleton() -> None:
    assert get_coordinator() is get_coordinator()


def test_disabled_coordinator_never_blocks_admission_and_handoffs_are_noops() -> None:
    coordinator = SerialOffloadCoordinator()

    assert coordinator.enabled is False
    assert coordinator.ar_can_admit() is True
    coordinator.begin_dit_handoff("req-1")
    coordinator.end_dit_handoff("req-1")
    assert coordinator.ar_can_admit() is True


def test_register_ar_enables_the_coordinator_and_starts_ar_active() -> None:
    coordinator = registered_coordinator()

    assert coordinator.enabled is True
    assert coordinator.ar_can_admit() is True


def test_begin_dit_handoff_moves_ar_off_the_gpu_and_blocks_admission() -> None:
    coordinator = registered_coordinator()
    assert coordinator.try_acquire_ar("req-1")

    coordinator.begin_dit_handoff("req-1")

    assert coordinator.ar_can_admit() is False


def test_end_dit_handoff_restores_ar_and_reopens_admission() -> None:
    coordinator = registered_coordinator()
    assert coordinator.try_acquire_ar("req-1")
    coordinator.begin_dit_handoff("req-1")

    coordinator.end_dit_handoff("req-1")

    assert coordinator.ar_can_admit() is True


def test_handoff_calls_are_idempotent() -> None:
    coordinator = registered_coordinator()
    assert coordinator.try_acquire_ar("req-1")

    coordinator.begin_dit_handoff("req-1")
    coordinator.begin_dit_handoff("req-1")
    assert coordinator.ar_can_admit() is False

    coordinator.end_dit_handoff("req-1")
    coordinator.end_dit_handoff("req-1")
    assert coordinator.ar_can_admit() is True


def test_one_owner_excludes_other_requests_until_acoustic_retires() -> None:
    coordinator = registered_coordinator()
    assert coordinator.try_acquire_ar("req-1")
    assert not coordinator.try_acquire_ar("req-2")
    coordinator.begin_dit_handoff("req-1")
    assert not coordinator.try_acquire_ar("req-2")
    with pytest.raises(RuntimeError, match="does not own"):
        coordinator.begin_dit_handoff("req-2")

    coordinator.end_dit_handoff("req-1")
    assert coordinator.try_acquire_ar("req-2")


def test_end_for_a_request_that_never_handed_off_does_not_wake_ar() -> None:
    coordinator = registered_coordinator()
    assert coordinator.try_acquire_ar("req-1")
    coordinator.begin_dit_handoff("req-1")

    coordinator.end_dit_handoff("req-unknown")

    assert coordinator.ar_can_admit() is False


def test_a_stalled_handoff_is_reported_once_and_never_force_woken(
    caplog: pytest.LogCaptureFixture,
) -> None:
    coordinator = registered_coordinator()
    assert coordinator.try_acquire_ar("req-1")
    coordinator.begin_dit_handoff("req-1")
    coordinator.paused_at_seconds -= STALL_REPORT_SECONDS + 1.0

    with caplog.at_level("ERROR"):
        assert coordinator.ar_can_admit() is False
        assert coordinator.ar_can_admit() is False

    stall_records = [r for r in caplog.records if "off the GPU for" in r.message]
    assert len(stall_records) == 1
    assert "req-1" in stall_records[0].message


def test_handoff_without_registration_raises_if_force_enabled() -> None:
    """Defensive guard for a caller that enables without registering."""
    coordinator = SerialOffloadCoordinator()
    coordinator.is_enabled = True

    with pytest.raises(RuntimeError, match="never registered"):
        coordinator.begin_dit_handoff("req-1")
    with pytest.raises(RuntimeError, match="never registered"):
        coordinator.end_dit_handoff("req-1")


def test_residency_sleep_and_wake_preserve_weights_and_state() -> None:
    module = torch.nn.Linear(2, 2)
    expected = module.weight.detach().clone()
    residency = StageResidency({"module": module}, torch.device("cpu"))

    residency.sleep()
    assert residency.resident is False
    residency.wake()

    assert residency.resident is True
    assert torch.equal(module.weight, expected)


def test_residency_reuses_one_host_copy_instead_of_recopying_each_sleep() -> None:
    """The weights are immutable, so only the first sleep may snapshot them."""
    module = torch.nn.Linear(2, 2)
    residency = StageResidency({"module": module}, torch.device("cpu"))

    residency.sleep()
    snapshot = residency.host_weights[("module", "weight")]
    residency.wake()
    residency.sleep()

    assert residency.host_weights[("module", "weight")] is snapshot


def test_a_host_built_module_is_asleep_and_never_snapshots_from_the_gpu() -> None:
    module = torch.nn.Linear(2, 2)
    residency = StageResidency({"module": module}, torch.device("cpu"), resident=False)

    assert residency.resident is False
    assert residency.host_weights[("module", "weight")] is not module.weight
    assert (
        residency.host_weights[("module", "weight")].data_ptr()
        == module.weight.data_ptr()
    )

    residency.wake()
    assert residency.resident is True


def test_residency_keeps_tied_weights_tied_across_a_round_trip() -> None:
    module = torch.nn.Linear(4, 4)
    tied = torch.nn.Linear(4, 4)
    tied.weight = module.weight
    parent = torch.nn.Sequential(module, tied)
    residency = StageResidency({"module": parent}, torch.device("cpu"))

    residency.sleep()
    residency.wake()

    assert module.weight is tied.weight
    assert module.weight.data_ptr() == tied.weight.data_ptr()


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_serial_offload_round_trip_moves_weights_between_devices() -> None:
    coordinator = SerialOffloadCoordinator()
    device = torch.device("cuda:0")
    model = torch.nn.Linear(4, 4).to(device)
    coordinator.register_ar(model, device)

    assert coordinator.try_acquire_ar("req-1")
    coordinator.begin_dit_handoff("req-1")
    assert next(model.parameters()).device.type == "cpu"

    coordinator.end_dit_handoff("req-1")
    assert next(model.parameters()).device == device


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_host_built_residency_keeps_canonical_copy_on_cpu() -> None:
    device = torch.device("cuda:0")
    model = torch.nn.Linear(4, 4)
    expected = model.weight.detach().clone()
    residency = StageResidency({"module": model}, device, resident=False)
    host_weight = residency.host_weights[("module", "weight")]

    residency.wake()
    assert model.weight.device == device
    assert host_weight.device.type == "cpu"

    residency.sleep()
    assert model.weight.device.type == "cpu"
    assert model.weight.data_ptr() == host_weight.data_ptr()
    assert torch.equal(model.weight, expected)

    residency.wake()
    assert model.weight.device == device
    assert host_weight.device.type == "cpu"
    assert torch.equal(model.weight.cpu(), expected)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_stage_group_wakes_once_and_keeps_allocator_blocks_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda:0")
    dit = torch.nn.Linear(4, 4)
    dav = torch.nn.Linear(4, 4)
    residency = StageResidency(
        {"dit": dit, "dav": dav}, device, resident=False, label="dit/dav"
    )
    synchronize_calls: list[torch.device] = []
    empty_cache_calls = 0
    synchronize = torch.cuda.synchronize
    empty_cache = torch.cuda.empty_cache

    def tracked_synchronize(target: torch.device) -> None:
        synchronize_calls.append(target)
        synchronize(target)

    def tracked_empty_cache() -> None:
        nonlocal empty_cache_calls
        empty_cache_calls += 1
        empty_cache()

    monkeypatch.setattr(torch.cuda, "synchronize", tracked_synchronize)
    monkeypatch.setattr(torch.cuda, "empty_cache", tracked_empty_cache)

    residency.wake()
    assert next(dit.parameters()).device == device
    assert next(dav.parameters()).device == device
    assert synchronize_calls == [device]

    residency.sleep()
    assert next(dit.parameters()).device.type == "cpu"
    assert next(dav.parameters()).device.type == "cpu"
    assert empty_cache_calls == 0


def test_ar_abort_releases_only_its_owner_before_handoff() -> None:
    coordinator = registered_coordinator()
    assert coordinator.try_acquire_ar("req-1")
    coordinator.cancel_ar("unknown")
    assert not coordinator.try_acquire_ar("req-2")
    coordinator.cancel_ar("req-1")
    assert coordinator.try_acquire_ar("req-2")
    coordinator.begin_dit_handoff("req-2")
    coordinator.cancel_ar("req-2")
    assert not coordinator.try_acquire_ar("req-3")


class RecordingAcousticDecoder:
    serial_offload = True

    def __init__(self) -> None:
        self.release_count = 0

    def offload_to_cpu(self) -> None:
        self.release_count += 1


def test_late_acoustic_cleanup_cannot_release_a_new_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = registered_coordinator()
    monkeypatch.setattr(
        "sglang_omni.models.minimax_music3.acoustic.get_coordinator",
        lambda: coordinator,
    )
    decoder = RecordingAcousticDecoder()
    scheduler = MiniMaxMusic3AcousticScheduler(decoder)
    assert coordinator.try_acquire_ar("old")
    coordinator.begin_dit_handoff("old")
    scheduler.clear_stream_state("old")
    assert decoder.release_count == 1

    assert coordinator.try_acquire_ar("new")
    coordinator.begin_dit_handoff("new")
    scheduler.clear_stream_state("old")
    scheduler.clear_stream_state("unknown")
    assert decoder.release_count == 1
    coordinator.require_acoustic("new")
    assert not coordinator.try_acquire_ar("next")
    scheduler.clear_stream_state("new")
    assert decoder.release_count == 2
    assert coordinator.try_acquire_ar("next")


@pytest.mark.parametrize("operation", ["sleep", "wake", "release"])
def test_failed_transition_prevents_further_admission(
    monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    coordinator = registered_coordinator()
    assert coordinator.try_acquire_ar("req-1")

    def fail() -> None:
        raise RuntimeError("transfer failed")

    if operation == "sleep":
        monkeypatch.setattr(coordinator.ar_residency, "sleep", fail)
        with pytest.raises(RuntimeError, match="transfer failed"):
            coordinator.begin_dit_handoff("req-1")
    else:
        coordinator.begin_dit_handoff("req-1")
        if operation == "wake":
            monkeypatch.setattr(coordinator.ar_residency, "wake", fail)
            release_acoustic = None
        else:
            release_acoustic = fail
        with pytest.raises(RuntimeError, match="transfer failed"):
            coordinator.end_dit_handoff("req-1", release_acoustic=release_acoustic)

    with pytest.raises(RuntimeError, match="transition failed"):
        coordinator.try_acquire_ar("req-2")


def test_acoustic_compute_requires_a_completed_handoff() -> None:
    coordinator = registered_coordinator()
    assert coordinator.try_acquire_ar("req-1")
    with pytest.raises(RuntimeError, match="does not own"):
        coordinator.require_acoustic("req-1")
    coordinator.begin_dit_handoff("req-1")
    coordinator.require_acoustic("req-1")
    with pytest.raises(RuntimeError, match="does not own"):
        coordinator.require_acoustic("req-2")


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_sleep_waits_for_compute_on_another_cuda_stream() -> None:
    device = torch.device("cuda:0")
    model = torch.nn.Linear(256, 256, bias=False).to(device)
    expected = model.weight.detach().cpu().clone()
    residency = StageResidency({"module": model}, device)
    residency.sleep()
    residency.wake()
    compute_stream = torch.cuda.Stream(device=device)
    completed = torch.cuda.Event()
    with torch.cuda.stream(compute_stream):
        torch.cuda._sleep(
            10_000_000
        )  # noqa: leading-underscore  # upstream CUDA test API
        output = model(torch.eye(256, device=device))
        completed.record()

    residency.sleep()

    assert completed.query()
    torch.testing.assert_close(output.cpu(), expected.T)


def test_scheduler_admits_only_the_owning_cfg_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = registered_coordinator()
    monkeypatch.setattr(
        "sglang_omni.models.minimax_music3.scheduler.get_coordinator",
        lambda: coordinator,
    )
    scheduler = MiniMaxMusic3Scheduler.__new__(MiniMaxMusic3Scheduler)
    scheduler.max_prefill_tokens = 100
    monkeypatch.setattr(scheduler, "get_num_allocatable_reqs", lambda count: 4)
    queue = [
        Req(
            rid=request_id,
            origin_input_text="",
            origin_input_ids=[1, 2],
            sampling_params=SamplingParams(max_new_tokens=1),
        )
        for request_id in ("first", "first-uncond", "second", "second-uncond")
    ]
    running_batch = ScheduleBatch.__new__(ScheduleBatch)
    running_batch.reqs = []

    assert scheduler.pair_admission_limit(queue, running_batch) == 2
    assert scheduler.pair_admission_limit(queue[2:], running_batch) == 0
    coordinator.begin_dit_handoff("first")
    assert scheduler.pair_admission_limit(queue[2:], running_batch) == 0
    coordinator.end_dit_handoff("first")
    assert scheduler.pair_admission_limit(queue[2:], running_batch) == 2
