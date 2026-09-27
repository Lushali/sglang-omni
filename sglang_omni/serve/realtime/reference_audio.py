# SPDX-License-Identifier: Apache-2.0
"""Validate uploaded session reference audio as canonical PCM16 WAV."""

import io
import wave

MAX_REFERENCE_AUDIO_BYTES = 1024 * 1024
MAX_REFERENCE_AUDIO_SECONDS = 30


def normalize_reference_wav(audio: bytes) -> bytes:
    # Note (Junnan Li): Streamed WAV writers leave the size fields as placeholders,
    # so the samples actually present decide the length, not the header.
    try:
        with wave.open(io.BytesIO(audio), "rb") as reference:
            rate = reference.getframerate()
            channels = reference.getnchannels()
            width = reference.getsampwidth()
            if not 8000 <= rate <= 48000 or channels not in (1, 2) or width != 2:
                raise ValueError(
                    "reference requires PCM16 WAV, 8-48 kHz, mono or stereo"
                )
            else:
                pcm = reference.readframes(rate * MAX_REFERENCE_AUDIO_SECONDS + 1)
    except (wave.Error, EOFError, RuntimeError) as exc:
        raise ValueError("invalid PCM WAV reference") from exc
    frames = len(pcm) // (channels * width)
    if not 0 < frames <= rate * MAX_REFERENCE_AUDIO_SECONDS:
        raise ValueError("reference audio must be nonempty and at most 30 seconds")
    else:
        pass
    # Note (Junnan Li): Canonical chunks keep downstream WAV readers on the
    # validated format.
    normalized = io.BytesIO()
    with wave.open(normalized, "wb") as reference:
        reference.setparams((channels, width, rate, frames, "NONE", "not compressed"))
        reference.writeframes(pcm[: frames * channels * width])
    return normalized.getvalue()
