# MiniCPM-o Reference Audio

On the speech pipeline, pass an explicit speaker reference in
`audio.ref_audio` on `/v1/chat/completions`:

```python
import base64
from pathlib import Path

from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="unused")
reference = base64.b64encode(Path("reference.wav").read_bytes()).decode("ascii")
response = client.chat.completions.create(
    model="MiniCPM-o-4_5",
    messages=[{"role": "user", "content": "Please say hello."}],
    modalities=["text", "audio"],
    audio={
        "format": "wav",
        "ref_audio": f"data:audio/wav;base64,{reference}",
    },
)
```

`stage_params.code2wav.ref_audio` is an alternative, with higher priority than
`audio.ref_audio`. Both accept `prompt_wav` as an alias. The Python pipeline
client can also supply `ref_audio` through `extra_params`. References must be
base64 audio data URIs, inline `{data, media_type}` descriptors, or encoded audio
bytes for the Python client. Paths and HTTP URLs are not fetched by this stage;
read or download the file on the client before sending it.

The reference conditions Token2wav's speaker embedding, prompt tokens, and mel
features. Audio supplied in chat messages remains understanding input and is not
automatically used as the speaker reference. Without an explicit reference,
Token2wav uses the checkpoint's `assets/HT_ref_audio.wav` when available.

## Native full-duplex sampling

For the native full-duplex pipeline, configure sampling in `session.update`
before sending the first audio packet:

```json
{
  "type": "session.update",
  "event_id": "sampling-1",
  "session": {
    "sglang": {
      "sampling": {
        "greedy": false,
        "temperature": 0.7,
        "top_k": 20,
        "top_p": 0.8,
        "repetition_penalty": 1.05,
        "listen_prob_scale": 1.0,
        "force_listen_count": 3
      }
    }
  }
}
```

`sglang.granted.sampling_parameters` lists the fields supported by the deployment.
Unsupported fields are rejected. Sampling settings are fixed once the session
opens; start a new session to change them after audio input has begun.

Omitted fields use the deployment defaults from the `sampling` section of the pipeline config (see `examples/full_duplex/minicpmo.yaml`), which ship as `greedy=false`, `temperature=0.7`, `top_k=20`, `top_p=0.8`, `repetition_penalty=1.05`, `listen_prob_scale=1.0`, and `force_listen_count=3`. Values sent in `session.update` apply only to that session. Set `greedy=false` to enable temperature/top-k/top-p
sampling; `temperature=0` selects the second-stage argmax. The initial
chunk-end draw follows `greedy` and uses the unscaled distribution.
`force_listen_count=0` disables the initial forced-listen units.

Temperature and listen probability scale must be nonnegative, top-p must be in
`(0, 1]`, repetition penalty must be positive, and forced-listen count must be a
nonnegative integer. Top-k accepts `-1` or `0` to disable filtering and positive
integers to enable it. These settings control the Thinker duplex sampler, not
the Talker sampling policy. `length_penalty` is not implemented and is rejected.
The official demo adapter must forward these fields explicitly to use them.

## Native full-duplex reference audio

Deployments advertising `sglang.granted.supports_reference_audio=true` accept per-session WAV references before the first audio append:

```python
reference = base64.b64encode(Path("reference.wav").read_bytes()).decode("ascii")
update = {
    "type": "session.update",
    "event_id": "voice-1",
    "session": {
        "sglang": {
            "reference_audio": {"media_type": "audio/wav", "data": reference}
        }
    },
}
```

`reference_audio` supplies the Perception system-prompt audio and the default speaker conditioning for Speech. An optional `tts_reference_audio` object with the same structure overrides only Speech. Omitting the fields uses the deployment's `reference_audio`, or the checkpoint reference when none is configured.

Each reference must be a base64-encoded PCM16 WAV file of at most 1 MiB: 8–48 kHz, mono or stereo, nonempty and at most 30 seconds. Header size fields are ignored and the length is taken from the samples present, so streamed WAV output with placeholder sizes is accepted. Paths and URLs are not accepted. Invalid references are rejected during negotiation.

References are frozen once the session opens and apply only to that session. They are input-only and never echoed in `session.updated`.

## Native duplex video and HD slices

The native deployment accepts several frames per audio unit. Send one `sglang.input_image.append` event per frame before the unit is cut by audio input. Frames in a unit are ordered by `sglang.t_ms`, with equal timestamps kept in arrival order, and all frames precede the unit's audio. Each frame is limited to 512 KiB encoded bytes and 4096 × 4096 pixels. `sglang.t_ms` is audio media time; after `input_audio_buffer.clear`, frames at the cleared time are rejected as stale.

Before the first audio packet, a session can request HD slicing:

```json
{
  "type": "session.update",
  "event_id": "vision-1",
  "session": {"sglang": {"max_slice_nums": 4}}
}
```

Each frame is encoded as one 64-embedding overview tile plus, when slicing, up to `max_slice_nums` 64-embedding crops; the processor picks the actual grid from the image size. The per-unit frame cap therefore shrinks as the slice count grows, and `sglang.granted.input_image_format` reports the negotiated `max_per_unit` together with the deployment's `max_slice_nums` limit. Extra frames are rejected before vision encoding. The setting is frozen once the session opens.

The limits and the default slice count come from the `vision` section of the pipeline config (see `examples/full_duplex/minicpmo.yaml`): `max_frames_per_unit`, `max_tiles_per_unit`, the session default `max_slice_nums` and the highest value a session may request, `max_slice_nums_limit`. These bound per-unit vision work, not the session context, which images, audio, prompt and generated tokens all consume.
