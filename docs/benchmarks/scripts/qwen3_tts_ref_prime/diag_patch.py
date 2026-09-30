# SPDX-License-Identifier: Apache-2.0
"""Diagnostic-only: time the vocoder's initial path per request (never merged)."""
import sys
from pathlib import Path

p = Path(sys.argv[1]) / "sglang_omni/models/qwen3_tts/streaming_vocoder.py"
s = p.read_text()


def sub(old: str, new: str) -> None:
    global s
    assert s.count(old) == 1, old[:80]
    s = s.replace(old, new)


sub(
    "import torch\n\n",
    "import torch\n\nfrom sglang_omni.profiler.event_recorder import emit as diag_emit\n\n",
)
sub(
    """        state.initial_pending = True
        self.initial_queue.put((request_id, state))
""",
    """        state.initial_pending = True
        diag_emit(request_id=request_id, stage="vocoder", event_name="diag_initial_enqueued",
                  metadata={"total_frames": state.total_frames, "ref_frames": state.ref_frames})
        self.initial_queue.put((request_id, state))
""",
)
sub(
    """    def run_initial_batch(self, batch: list[tuple[str, Qwen3TTSStreamState]]) -> None:
""",
    """    def run_initial_batch(self, batch: list[tuple[str, Qwen3TTSStreamState]]) -> None:
        for request_id, _ in batch:
            diag_emit(request_id=request_id, stage="vocoder", event_name="diag_batch_start",
                      metadata={"batch": len(batch)})
""",
)
sub(
    """        for cohort in self.group_decode_plans(planned_incremental):
            for group in self.split_incremental_group_for_graph(
                cohort,
                runner=self.initial_incremental_decode_graphs,
""",
    """        for request_id, _, plan in planned_incremental:
            diag_emit(request_id=request_id, stage="vocoder", event_name="diag_plan_built",
                      metadata={"fresh": plan.fresh_frames,
                                "context": int(getattr(plan, "context_frames", 0)),
                                "trim": plan.reference_trim_frames})
        for cohort in self.group_decode_plans(planned_incremental):
            for group in self.split_incremental_group_for_graph(
                cohort,
                runner=self.initial_incremental_decode_graphs,
""",
)
sub(
    """        pending = self.launch_incremental_group(group, stream=stream)
        if pending is None:
            return None
        else:
            pass
        return self.finish_incremental_group(pending)
""",
    """        pending = self.launch_incremental_group(group, stream=stream)
        if pending is None:
            return None
        else:
            pass
        if stream is not None:
            stream.synchronize()
        for request_id, _, _ in group:
            diag_emit(request_id=request_id, stage="vocoder", event_name="diag_decode_done")
        result = self.finish_incremental_group(pending)
        for request_id, _, _ in group:
            diag_emit(request_id=request_id, stage="vocoder", event_name="diag_resolved")
        return result
""",
)
if "def advance_incremental_context" in s:
    sub(
        """        self.advance_incremental_context(plans, incremental)
        width = plans[0].fresh_frames
""",
        """        self.advance_incremental_context(plans, incremental)
        if stream is not None:
            stream.synchronize()
        diag_emit(request_id="cohort", stage="vocoder", event_name="diag_context_done",
                  metadata={"rows": len(plans), "context": [p.context_frames for p in plans]})
        width = plans[0].fresh_frames
""",
    )
sub(
    """                state.initial_pending = False
                if not self.is_aborted(request_id):
                    self.mark_stream_emitted(request_id)
                    self.outbox.put(self.stream_chunk_message(request_id, delta))
""",
    """                state.initial_pending = False
                if not self.is_aborted(request_id):
                    self.mark_stream_emitted(request_id)
                    diag_emit(request_id=request_id, stage="vocoder", event_name="diag_commit_put")
                    self.outbox.put(self.stream_chunk_message(request_id, delta))
""",
)
p.write_text(s)
print("patched", p)
