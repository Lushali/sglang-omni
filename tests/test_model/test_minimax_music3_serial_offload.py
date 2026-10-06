# SPDX-License-Identifier: Apache-2.0
"""Opt-in Music3 HTTP parity and repeated serial-offload integration test."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import requests
import soundfile as sf

from benchmarks.benchmarker.utils import start_server_from_cmd, stop_server
from sglang_omni.utils.connection import find_available_port

CHECKPOINT_ENV = "MINIMAX_MUSIC3_TEST_CHECKPOINT"
REPEATED_REQUESTS_ENV = "MINIMAX_MUSIC3_TEST_REPEATED_REQUESTS"
STARTUP_TIMEOUT_SECONDS = 600
REQUEST_TIMEOUT_SECONDS = 600
HANDOFF_TIMEOUT_SECONDS = 30
SAMPLE_RATE = 32_000
FRAME_RATE = 25
MAX_RELATIVE_RMS_DIFFERENCE = 0.005
MIN_WAVEFORM_CORRELATION = 0.9999


@pytest.mark.accelerator
def test_music3_serial_offload_http_parity_and_lifecycle(tmp_path: Path) -> None:
    checkpoint = os.environ.get(CHECKPOINT_ENV)
    if not checkpoint:
        pytest.skip(f"Set {CHECKPOINT_ENV} to run the Music3 HTTP integration test")
    else:
        pass
    repeated_requests = int(os.environ.get(REPEATED_REQUESTS_ENV, "20"))
    payload = {
        "model": "minimax-music3",
        "input": "[Verse]\nCity lights are calling out my name",
        "instructions": "A dreamy synthwave track with analog pads and a bassline at 110 BPM",
        "seed": 42,
    }
    resident_audio: dict[int, bytes] = {}
    offload_audio: dict[int, bytes] = {}
    metrics: dict[str, list[dict[str, int | float | str]]] = {}

    for offload in (False, True):
        mode = "offload" if offload else "resident"
        port = find_available_port(host="127.0.0.1")
        log_path = tmp_path / f"{mode}.log"
        command = [
            sys.executable,
            "-m",
            "sglang_omni.cli",
            "serve",
            "--model-path",
            checkpoint,
            "--model-name",
            payload["model"],
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--dit_dav.process",
            "minimax_music3_ar",
            "--minimax_music3_ar.engine.max_running_requests",
            "1",
            "--minimax_music3_ar.engine.disable_cuda_graph",
            "true",
            "--dit_dav.factory.compile_acoustic",
            "false",
        ]
        if offload:
            command.extend(["--stage-offload-components", "ar,dit"])
        else:
            pass
        server = start_server_from_cmd(
            command,
            log_path,
            port,
            timeout=STARTUP_TIMEOUT_SECONDS,
            env={"CUDA_VISIBLE_DEVICES": "0"},
        )
        metrics[mode] = []

        def generate_audio(frames: int) -> bytes:
            started_at_seconds = time.perf_counter()
            response = requests.post(
                f"http://127.0.0.1:{port}/v1/audio/speech",
                json={**payload, "max_new_tokens": frames},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            assert response.status_code == 200, response.text
            waveform, sample_rate = sf.read(
                io.BytesIO(response.content), always_2d=True
            )
            assert sample_rate == SAMPLE_RATE
            assert waveform.shape[1] == 2
            assert np.isfinite(waveform).all()
            assert np.max(np.abs(waveform)) > 0
            duration_seconds = waveform.shape[0] / sample_rate
            assert abs(duration_seconds - frames / FRAME_RATE) < 0.1
            metrics[mode].append(
                {
                    "frames": frames,
                    "duration_seconds": duration_seconds,
                    "latency_seconds": time.perf_counter() - started_at_seconds,
                    "sha256": hashlib.sha256(response.content).hexdigest(),
                }
            )
            return response.content

        try:
            for frames in (100, 250):
                audio = generate_audio(frames)
                (tmp_path / f"{mode}-{frames}.wav").write_bytes(audio)
                if offload:
                    reference_waveform, _ = sf.read(io.BytesIO(resident_audio[frames]))
                    offload_waveform, _ = sf.read(io.BytesIO(audio))
                    assert reference_waveform.shape == offload_waveform.shape
                    difference = offload_waveform - reference_waveform
                    relative_rms_difference = float(
                        np.sqrt(np.mean(difference**2) / np.mean(reference_waveform**2))
                    )
                    waveform_correlation = float(
                        np.corrcoef(
                            reference_waveform.ravel(), offload_waveform.ravel()
                        )[0, 1]
                    )
                    metrics[mode][-1][
                        "relative_rms_difference"
                    ] = relative_rms_difference
                    metrics[mode][-1]["waveform_correlation"] = waveform_correlation
                    assert relative_rms_difference < MAX_RELATIVE_RMS_DIFFERENCE
                    assert waveform_correlation > MIN_WAVEFORM_CORRELATION
                    offload_audio[frames] = audio
                    assert generate_audio(frames) == audio
                else:
                    resident_audio[frames] = audio

            if offload:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    submitted = [executor.submit(generate_audio, 100) for _ in range(2)]
                    for future in submitted:
                        assert future.result() == offload_audio[100]
                repeated_audio = generate_audio(25)
                for _ in range(repeated_requests - 1):
                    assert generate_audio(25) == repeated_audio
                handoff_deadline_seconds = time.monotonic() + HANDOFF_TIMEOUT_SECONDS
                while log_path.read_text().count("serial offload: AR -> GPU") < len(
                    metrics[mode]
                ):
                    assert time.monotonic() < handoff_deadline_seconds
                    time.sleep(0.1)
            else:
                pass
        finally:
            stop_server(server)
            (tmp_path / "metrics.json").write_text(json.dumps(metrics, indent=2))

        if offload:
            log_text = log_path.read_text()
            transitions = re.findall(
                r"serial offload: AR -> (CPU|GPU) .*?request=([^\)]+)", log_text
            )
            assert len(transitions) == 2 * len(metrics[mode])
            for index in range(0, len(transitions), 2):
                assert transitions[index][0] == "CPU"
                assert transitions[index + 1][0] == "GPU"
                assert transitions[index][1] == transitions[index + 1][1]
            assert "dit/dav -> gpu" in log_text
            assert "dit/dav -> host" in log_text
            assert "Traceback" not in log_text
        else:
            pass
