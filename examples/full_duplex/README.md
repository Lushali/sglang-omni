# Full-duplex audio examples

Run commands from the repository root. The native audio route uses `/v1/realtime`
with `session.update`, `input_audio_buffer.append`, `sglang.input_audio.end`, and
`session.close`. Wait for `sglang.input_audio.drained` before closing after input EOS.

## MiniCPM-o

Set `model_path` in `minicpmo.yaml` to a prepared MiniCPM-o 4.5 checkpoint, then run:

```bash
sgl-omni serve --config examples/full_duplex/minicpmo.yaml --enable-realtime
```

The `MiniCPMODuplexPipelineConfig` is also available as the `session` model variant.
The existing `text` and `speech` variants keep their ordinary request pipelines.
An optional top-level `reference_audio` path supplies the reference for both
perception and speech; the default is the checkpoint's `assets/HT_ref_audio.wav`.
Set `max_sessions` and `speech_state_bytes_per_session` in `minicpmo.yaml` to
adjust session capacity and the speech memory budget per session (defaults: 2
and 2 GiB). Perception and speech each process one unit at a time. Engine request
limits are `max(engine default, max_sessions + 1)`, reserving one slot for retained
KV; engine defaults are 4 for thinker and 32 for talker. Explicit `engine` overrides
take precedence.

The native path accepts mono PCM16 at 16 kHz and emits 24 kHz audio and text over
`/v1/realtime`. It processes one-second units with session-resident encoder,
thinker KV, sampler history, TTS, and vocoder state. Empty input EOS reaches the
speech stage to flush pending audio without an additional thinker request.

This integration uses the current shared native protocol: open, append, and close.
It does not expose the historical epoch/cancel or `sglang.microturn.done` events.
Video input is not wired into this audio path.

The thinker defaults to 8192 tokens; optionally set `stages.thinker.engine.context_length`
up to the checkpoint's `max_position_embeddings` (40960 for
MiniCPM-o 4.5), with KV storage approximately 144 KiB/token, or 4.5 GiB per full 32k session.

When accumulated history plus a new unit exceeds the effective input limit, the
server emits a fatal `context_exhausted` error naming the thinker context length,
then closes the session; it does not truncate history or continue with later units.
The effective limit can be lower than the configured context length due to KV capacity.
