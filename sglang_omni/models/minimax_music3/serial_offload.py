# SPDX-License-Identifier: Apache-2.0
"""Serial GPU residency for MiniMax Music 3 AR and DIT/DAV."""

from __future__ import annotations

import logging
import threading
import time

import torch

logger = logging.getLogger(__name__)

STALL_REPORT_SECONDS = 900.0
BYTES_PER_GIB = 1024.0**3


def owned_tensors(
    modules: dict[str, torch.nn.Module],
) -> list[tuple[tuple[str, str], torch.Tensor]]:
    """List stage tensors once so shared weights retain one canonical host copy."""
    owned_weights: list[tuple[tuple[str, str], torch.Tensor]] = []
    seen_tensor_ids: set[int] = set()
    for module_name, module in modules.items():
        for tensor_name, tensor in (
            *module.named_parameters(),
            *module.named_buffers(),
        ):
            if id(tensor) in seen_tensor_ids:
                continue
            else:
                pass
            seen_tensor_ids.add(id(tensor))
            owned_weights.append(((module_name, tensor_name), tensor))
    return owned_weights


def gpu_memory_note(device: torch.device) -> str:
    """Report allocator and device occupancy; offload leaves the KV cache resident."""
    if device.type != "cuda":
        return "cpu"
    else:
        pass
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return (
        f"free={free_bytes / BYTES_PER_GIB:.2f}/{total_bytes / BYTES_PER_GIB:.2f}GiB "
        f"torch_allocated={torch.cuda.memory_allocated(device) / BYTES_PER_GIB:.2f}GiB "
        f"torch_reserved={torch.cuda.memory_reserved(device) / BYTES_PER_GIB:.2f}GiB"
    )


class StageResidency:
    """Reuse canonical CPU weights while dropping and restoring GPU replicas."""

    def __init__(
        self,
        modules: dict[str, torch.nn.Module],
        device: torch.device,
        *,
        resident: bool = True,
        label: str = "stage",
    ) -> None:
        if not modules:
            raise ValueError("StageResidency requires at least one module")
        else:
            pass
        self.modules: dict[str, torch.nn.Module] = dict(modules)
        self.device: torch.device = torch.device(device)
        self.is_resident: bool = bool(resident)
        self.label: str = label
        self.host_weights: dict[tuple[str, str], torch.Tensor] = {}
        if not self.is_resident:
            # Note (Akazaakane): Detach the host handle so wake cannot replace its storage.
            self.host_weights = {
                name: tensor.detach() for name, tensor in owned_tensors(self.modules)
            }
        else:
            pass
        weight_bytes = sum(
            tensor.numel() * tensor.element_size()
            for _, tensor in owned_tensors(self.modules)
        )
        logger.info(
            f"MiniMax Music 3 residency {label}: {weight_bytes / BYTES_PER_GIB:.2f}GiB of "
            f"weights, starts {'resident' if self.is_resident else 'offloaded'} "
            f"({gpu_memory_note(self.device)})"
        )

    @property
    def resident(self) -> bool:
        return self.is_resident

    def sleep(self) -> None:
        """Drop the GPU replica; a cheap no-op once already asleep."""
        if not self.is_resident:
            return
        else:
            pass
        owned_weights = owned_tensors(self.modules)
        if not self.host_weights:
            self.host_weights = {
                name: tensor.detach().to("cpu", copy=True)
                for name, tensor in owned_weights
            }
        else:
            pass
        for name, tensor in owned_weights:
            tensor.data = self.host_weights[name]
        self.is_resident = False
        logger.info(
            f"MiniMax Music 3 residency {self.label} -> host "
            f"({gpu_memory_note(self.device)})"
        )

    def wake(self) -> None:
        """Refill the GPU replica from the host copy; no-op once resident."""
        if self.is_resident:
            return
        else:
            pass
        for name, tensor in owned_tensors(self.modules):
            tensor.data = self.host_weights[name].to(
                self.device, copy=True, non_blocking=True
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        else:
            pass
        self.is_resident = True
        logger.info(
            f"MiniMax Music 3 residency {self.label} -> gpu "
            f"({gpu_memory_note(self.device)})"
        )


class SerialOffloadCoordinator:
    """Coordinate request-scoped AR and DIT/DAV GPU handoffs."""

    def __init__(self) -> None:
        self.lock: threading.Lock = threading.Lock()
        self.is_enabled: bool = False
        self.is_ar_active: bool = True
        self.ar_residency: StageResidency | None = None
        self.outstanding_request_ids: set[str] = set()
        self.paused_at_seconds: float | None = None
        self.has_reported_stall: bool = False

    @property
    def enabled(self) -> bool:
        return self.is_enabled

    def register_ar(self, model: torch.nn.Module, device: torch.device) -> None:
        with self.lock:
            self.ar_residency = StageResidency({"ar": model}, device, label="ar")
            self.is_enabled = True
            self.is_ar_active = True
        logger.info(
            f"MiniMax Music 3 serial offload enabled device={device}; AR "
            "starts GPU-resident, DIT/DAV starts offloaded"
        )

    def ar_can_admit(self) -> bool:
        """Whether AR may admit a new request onto the GPU right now."""
        if not self.is_enabled:
            return True
        else:
            pass
        with self.lock:
            if self.is_ar_active:
                return True
            else:
                pass
            self.report_stall_locked()
            return False

    def begin_dit_handoff(self, request_id: str) -> None:
        """Hand the GPU to DIT/DAV for *request_id* and take AR off it."""
        if not self.is_enabled:
            return
        else:
            pass
        with self.lock:
            self.require_ar_locked()
            self.outstanding_request_ids.add(request_id)
            if not self.is_ar_active:
                return
            else:
                pass
            self.ar_residency.sleep()
            self.is_ar_active = False
            self.paused_at_seconds = time.monotonic()
            self.has_reported_stall = False
        logger.info(
            f"MiniMax Music 3 serial offload: AR -> CPU (DIT/DAV's turn, "
            f"request={request_id})"
        )

    def end_dit_handoff(self, request_id: str) -> None:
        """Retire a handoff and restore AR once no requests remain outstanding."""
        if not self.is_enabled:
            return
        else:
            pass
        with self.lock:
            self.require_ar_locked()
            self.outstanding_request_ids.discard(request_id)
            if self.is_ar_active or self.outstanding_request_ids:
                return
            else:
                pass
            self.ar_residency.wake()
            self.is_ar_active = True
            self.paused_at_seconds = None
            self.has_reported_stall = False
        logger.info(
            f"MiniMax Music 3 serial offload: AR -> GPU (AR's turn, "
            f"request={request_id})"
        )

    def require_ar_locked(self) -> None:
        if self.ar_residency is None:
            raise RuntimeError(
                "MiniMax Music 3 serial offload is enabled but the AR "
                "backbone was never registered"
            )
        else:
            pass

    def report_stall_locked(self) -> None:
        """Name the outstanding requests once AR has been parked too long."""
        if self.paused_at_seconds is None or self.has_reported_stall:
            return
        else:
            pass
        elapsed_seconds = time.monotonic() - self.paused_at_seconds
        if elapsed_seconds < STALL_REPORT_SECONDS:
            return
        else:
            pass
        self.has_reported_stall = True
        logger.error(
            f"MiniMax Music 3 serial offload: AR has been off the GPU for "
            f"{elapsed_seconds:.0f}s and is still waiting on "
            f"{sorted(self.outstanding_request_ids)}; AR admits nothing until DIT/DAV "
            "retires them, so this server needs a restart if the requests "
            "are gone"
        )


COORDINATOR = SerialOffloadCoordinator()


def get_coordinator() -> SerialOffloadCoordinator:
    return COORDINATOR


__all__ = [
    "STALL_REPORT_SECONDS",
    "StageResidency",
    "SerialOffloadCoordinator",
    "get_coordinator",
]
