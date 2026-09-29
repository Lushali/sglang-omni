#!/usr/bin/env bash
# note (luojiaxuan): output parity of the Qwen3-TTS prefill graph backends with
# same-config restart controls. Every arm is a fresh server on one GPU; the
# arms are interleaved (a round, then b round) so drift over the session hits
# both copies of a backend. Each arm serves the same 56 SeedTTS EN prompts to
# the CustomVoice speaker Ryan, streaming, open loop at 2 rps with a fixed
# arrival seed and a fixed request seed. Temperature 0 does not make Qwen3-TTS
# deterministic (its runs differ even between restarts of one config), so only
# the seeded run is compared.
set -euo pipefail
: "${REPO:?checkout to serve from}" "${OUT:?output root}" "${PY:?python}"
MODEL=${MODEL:-Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice}
PORT=${PORT:-18231}
ARMS=${ARMS:-"eager-a:disabled breakable-a:breakable full-a:full eager-b:disabled breakable-b:breakable full-b:full"}
cd "$REPO"

wait_gpu_free() {
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1024 ]; do
    sleep 2
  done
}

run_arm() {
  local arm=$1 backend=$2 dir="$OUT/$1"
  mkdir -p "$dir"
  echo "$(date -u +%FT%TZ) start $arm backend=$backend" | tee -a "$OUT/events.log"
  setsid "$PY" -m sglang_omni.cli serve --model-path "$MODEL" --host 127.0.0.1 --port "$PORT" \
    --tts_engine.engine.cuda_graph_backend_prefill "$backend" > "$dir/server.log" 2>&1 &
  local server=$!
  until curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; do
    if ! kill -0 "$server" 2> /dev/null; then
      echo "$(date -u +%FT%TZ) FAIL $arm server exited" | tee -a "$OUT/events.log"
      tail -40 "$dir/server.log"
      exit 1
    fi
    sleep 5
  done
  grep -a "prefill" "$dir/server.log" | grep -ai "graph\|capture\|backend" | tail -5 > "$dir/prefill-capture.txt" || true
  "$PY" -m benchmarks.eval.benchmark_tts_seedtts --model "$MODEL" --port "$PORT" \
    --use-existing-server --generate-only --no-ref-audio --voice Ryan \
    --max-samples 56 --stream --request-rate 2 --arrival-seed 7 --seed 1234 --save-audio \
    --disable-tqdm --output-dir "$dir/seeded" > "$dir/seeded.log" 2>&1
  echo "$(date -u +%FT%TZ) done $arm seeded" | tee -a "$OUT/events.log"
  kill -- "-$server"
  wait "$server" || true
  wait_gpu_free
}

for spec in $ARMS; do
  run_arm "${spec%%:*}" "${spec##*:}"
done
echo "$(date -u +%FT%TZ) PARITY_DONE" | tee -a "$OUT/events.log"
