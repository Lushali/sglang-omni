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

The vocoder caches only the most recently used reference by audio content. A
different reference, including switching back to the default, rebuilds the
conditioning. Invalid references fail instead of silently using the default.
Audio output remains non-streaming.

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

Omitted fields use the deployment defaults from the `sampling` section of the pipeline config (see `examples/full_duplex/minicpmo.yaml`), which ship as `greedy=true`, `temperature=0.7`, `top_k=100`, `top_p=0.8`, `repetition_penalty=1.05`, `listen_prob_scale=1.0`, and `force_listen_count=3`. Values sent in `session.update` apply only to that session. Set `greedy=false` to enable temperature/top-k/top-p
sampling; `temperature=0` selects the second-stage argmax. The initial
chunk-end draw follows `greedy` and uses the unscaled distribution.
`force_listen_count=0` disables the initial forced-listen units.

Temperature and listen probability scale must be nonnegative, top-p must be in
`(0, 1]`, repetition penalty must be positive, and forced-listen count must be a
nonnegative integer. Top-k accepts `-1` or `0` to disable filtering and positive
integers to enable it. These settings control the Thinker duplex sampler, not
the Talker sampling policy. `length_penalty` is not implemented and is rejected.
The official demo adapter must forward these fields explicitly to use them.
