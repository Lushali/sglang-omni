#!/usr/bin/env bash
# note (luojiaxuan): main against the reference-priming branch on Base voice
# cloning (ICL), with same-config restart controls. Every arm is a fresh server
# in the CI deployment shape (vocoder in its own process); arms are interleaved
# (a round, then b round). Each arm runs, in order: a warmup on prompts that are
# not measured; the seeded parity run (56 prompts, 2 rps, audio saved); a 1 rps
# run on prompts no earlier run of this server used, so their references are
# encoded on the path; the same prompts again with request events on, now with
# cached references; and 20 rps with the worker's admission cap of 64. The
# client's warmup burst is off in every measured run, so request events hold
# only timed requests and references stay cold until a run encodes them.
set -euo pipefail
: "${MAIN:?main checkout}" "${PRIME:?branch checkout}" "${OUT:?output root}" "${PY:?python}"
MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base
PORT=${PORT:-18233}
ARMS=${ARMS:-"main-a:MAIN prime-a:PRIME main-b:MAIN prime-b:PRIME"}

wait_gpu_free() {
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1024 ]; do
    sleep 2
  done
}

bench() {
  local out=$1 rate=$2 offset=$3 samples=$4
  shift 4
  "$PY" -m benchmarks.eval.benchmark_tts_seedtts --model "$MODEL" --port "$PORT" \
    --use-existing-server --generate-only --stream --ref-format references \
    --request-rate "$rate" --max-concurrency 64 --warmup 0 --seed 1234 \
    --sample-offset "$offset" --max-samples "$samples" --disable-tqdm \
    --output-dir "$out" "$@" > "$out.log" 2>&1
}

events() {
  curl -sf -X POST "http://127.0.0.1:$PORT/$1_request_profile" -H 'Content-Type: application/json' \
    -d "{\"run_id\": \"$2\", \"event_dir\": \"$3\"}" > /dev/null
}

for spec in $ARMS; do
  IFS=: read -r arm tree <<< "$spec"
  repo=${!tree}
  dir="$OUT/$arm"
  mkdir -p "$dir"
  cd "$repo"
  echo "$(date -u +%FT%TZ) start $arm $(git rev-parse --short HEAD)" | tee -a "$OUT/events.log"
  PYTHONPATH="$repo" setsid "$PY" -m sglang_omni.cli serve --model-path "$MODEL" --host 127.0.0.1 \
    --port "$PORT" --vocoder.process vocoder --tts_engine.gpu_memory_fraction 0.85 \
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
  bench "$dir/warmup" 4 1000 40 --arrival-seed 0
  bench "$dir/seeded" 2 0 56 --arrival-seed 7 --save-audio
  bench "$dir/r1-cold" 1 100 60 --arrival-seed 0
  events start "$arm-hot" "$dir/events-hot"
  bench "$dir/r1-hot" 1 100 60 --arrival-seed 0
  events stop "$arm-hot" "$dir/events-hot"
  bench "$dir/r20" 20 0 1088 --arrival-seed 0
  echo "$(date -u +%FT%TZ) done $arm" | tee -a "$OUT/events.log"
  kill -- "-$server"
  wait "$server" || true
  wait_gpu_free
done
echo "$(date -u +%FT%TZ) AB_DONE" | tee -a "$OUT/events.log"
