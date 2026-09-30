#!/usr/bin/env bash
# note (luojiaxuan): where Base first audio goes, against CustomVoice, in the CI
# deployment shape (vocoder in its own process). Each arm is a fresh server,
# warmed on prompts that are not measured. The engine caches reference
# encodings per clip, so each 1 rps run takes prompts no earlier run used:
# a plain run for clean first playable, a run with request events on for the
# per-stage breakdown, then the events run's prompts again, whose references
# are now cached, so cold minus hot is the reference encoding on the path.
# The 20 rps run gives the client the worker's admission cap of 64, as the CI
# latency stage does. Arrivals and request seeds are fixed. The client's warmup
# burst is off: it defaults to the concurrency cap, would land in the request
# events and would encode the measured prompts' references ahead of them.
set -euo pipefail
: "${REPO:?checkout to serve from}" "${OUT:?output root}" "${PY:?python}"
PORT=${PORT:-18232}
BASE=Qwen/Qwen3-TTS-12Hz-1.7B-Base
CV=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
# name:model:client-args:with-r20
ARMS=${ARMS:-"base-icl:$BASE:--ref-format=references:1 base-xvec:$BASE:--ref-format=references,--no-ref-text:0 cv:$CV:--no-ref-audio,--voice=Ryan:1"}
cd "$REPO"

wait_gpu_free() {
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1024 ]; do
    sleep 2
  done
}

bench() {
  local model=$1 out=$2 rate=$3 offset=$4 samples=$5
  shift 5
  "$PY" -m benchmarks.eval.benchmark_tts_seedtts --model "$model" --port "$PORT" \
    --use-existing-server --generate-only --stream --request-rate "$rate" \
    --max-concurrency 64 --warmup 0 --arrival-seed 0 --seed 1234 \
    --sample-offset "$offset" --max-samples "$samples" --disable-tqdm \
    --output-dir "$out" "$@" > "$out.log" 2>&1
}

events() {
  curl -sf -X POST "http://127.0.0.1:$PORT/$1_request_profile" -H 'Content-Type: application/json' \
    -d "{\"run_id\": \"$2\", \"event_dir\": \"$3\"}" > /dev/null
}

for spec in $ARMS; do
  IFS=: read -r arm model client r20 <<< "$spec"
  IFS=, read -r -a client_args <<< "$client"
  dir="$OUT/$arm"
  mkdir -p "$dir"
  echo "$(date -u +%FT%TZ) start $arm" | tee -a "$OUT/events.log"
  setsid "$PY" -m sglang_omni.cli serve --model-path "$model" --host 127.0.0.1 --port "$PORT" \
    --vocoder.process vocoder --tts_engine.gpu_memory_fraction 0.85 \
    --vocoder.gpu_memory_fraction 0.10 > "$dir/server.log" 2>&1 &
  server=$!
  until curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; do
    if ! kill -0 "$server" 2> /dev/null; then
      echo "$(date -u +%FT%TZ) FAIL $arm server exited" | tee -a "$OUT/events.log"
      tail -40 "$dir/server.log"
      exit 1
    fi
    sleep 5
  done
  bench "$model" "$dir/warmup" 4 1000 40 "${client_args[@]}"
  bench "$model" "$dir/r1-plain" 1 0 60 "${client_args[@]}"
  events start "$arm-cold" "$dir/events-cold"
  bench "$model" "$dir/r1-events-cold" 1 60 60 "${client_args[@]}"
  events stop "$arm-cold" "$dir/events-cold"
  events start "$arm-hot" "$dir/events-hot"
  bench "$model" "$dir/r1-events-hot" 1 60 60 "${client_args[@]}"
  events stop "$arm-hot" "$dir/events-hot"
  if [ "$r20" = 1 ]; then
    bench "$model" "$dir/r20-plain" 20 0 1088 "${client_args[@]}"
  else
    :
  fi
  echo "$(date -u +%FT%TZ) done $arm" | tee -a "$OUT/events.log"
  kill -- "-$server"
  wait "$server" || true
  wait_gpu_free
done
echo "$(date -u +%FT%TZ) RUN_DONE" | tee -a "$OUT/events.log"
