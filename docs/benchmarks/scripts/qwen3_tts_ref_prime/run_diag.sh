#!/usr/bin/env bash
# note (luojiaxuan): diagnostic run of the vocoder's initial path with timing
# events, main against the priming branch on Base ICL plus main on x-vector.
# Each arm: fresh server, warmup, a 1 rps pass that encodes the references,
# then the same prompts again with request events on.
set -euo pipefail
: "${OUT:?}" "${PY:?}" "${RUN_ROOT:?}"
MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base
PORT=18234
ARMS="diag-main-icl:diag-main:--ref-format=references diag-prime-icl:diag-prime:--ref-format=references diag-main-xvec:diag-main:--ref-format=references,--no-ref-text"
wait_gpu_free() {
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1024 ]; do sleep 2; done
}
bench() {
  local out=$1 offset=$2; shift 2
  "$PY" -m benchmarks.eval.benchmark_tts_seedtts --model "$MODEL" --port "$PORT" \
    --use-existing-server --generate-only --stream --request-rate 1 --max-concurrency 64 \
    --warmup 0 --arrival-seed 0 --seed 1234 --sample-offset "$offset" --max-samples 60 \
    --disable-tqdm --output-dir "$out" "$@" > "$out.log" 2>&1
}
for spec in $ARMS; do
  IFS=: read -r arm tree client <<< "$spec"
  IFS=, read -r -a client_args <<< "$client"
  repo="$RUN_ROOT/$tree"
  dir="$OUT/$arm"
  mkdir -p "$dir"
  cd "$repo"
  echo "$(date -u +%FT%TZ) start $arm $(git rev-parse --short HEAD)" | tee -a "$OUT/events.log"
  PYTHONPATH="$repo" setsid "$PY" -m sglang_omni.cli serve --model-path "$MODEL" --host 127.0.0.1 \
    --port "$PORT" --vocoder.process vocoder --tts_engine.gpu_memory_fraction 0.85 \
    --vocoder.gpu_memory_fraction 0.10 > "$dir/server.log" 2>&1 &
  server=$!
  until curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; do
    kill -0 "$server" 2> /dev/null || { echo "$(date -u +%FT%TZ) FAIL $arm server exited" | tee -a "$OUT/events.log"; exit 1; }
    sleep 5
  done
  bench "$dir/warmup" 1000 "${client_args[@]}"
  bench "$dir/r1-cold" 100 "${client_args[@]}"
  curl -sf -X POST "http://127.0.0.1:$PORT/start_request_profile" -H 'Content-Type: application/json' \
    -d "{\"run_id\": \"$arm\", \"event_dir\": \"$dir/events\"}" > /dev/null
  bench "$dir/r1-hot" 100 "${client_args[@]}"
  curl -sf -X POST "http://127.0.0.1:$PORT/stop_request_profile" -H 'Content-Type: application/json' \
    -d "{\"run_id\": \"$arm\", \"event_dir\": \"$dir/events\"}" > /dev/null
  echo "$(date -u +%FT%TZ) done $arm" | tee -a "$OUT/events.log"
  kill -- "-$server"; wait "$server" || true; wait_gpu_free
done
echo "$(date -u +%FT%TZ) DIAG_DONE" | tee -a "$OUT/events.log"
