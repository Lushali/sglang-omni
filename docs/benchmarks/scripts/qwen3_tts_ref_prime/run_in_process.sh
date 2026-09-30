#!/usr/bin/env bash
# note (luojiaxuan): main against the priming branch with the vocoder in the
# default placement (same process as the engine). Clean arms interleave a/b
# with a warmup, a 1 rps pass that encodes the references, the same prompts
# again with request events on, and 20 rps; diag arms add the vocoder timing
# events and run only the two 1 rps passes.
set -euo pipefail
: "${RUN_ROOT:?}" "${OUT:?}" "${PY:?}"
MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base
PORT=18235
ARMS=${ARMS:-"inproc-main-a:sglang-omni:full inproc-prime-a:prime:full inproc-main-b:sglang-omni:full inproc-prime-b:prime:full diag-inproc-main:diag-main:diag diag-inproc-prime:diag-prime:diag"}
wait_gpu_free() {
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1024 ]; do sleep 2; done
}
bench() {
  local out=$1 rate=$2 offset=$3 samples=$4
  "$PY" -m benchmarks.eval.benchmark_tts_seedtts --model "$MODEL" --port "$PORT" \
    --use-existing-server --generate-only --stream --ref-format references \
    --request-rate "$rate" --max-concurrency 64 --warmup 0 --arrival-seed 0 --seed 1234 \
    --sample-offset "$offset" --max-samples "$samples" --disable-tqdm \
    --output-dir "$out" > "$out.log" 2>&1
}
events() {
  curl -sf -X POST "http://127.0.0.1:$PORT/$1_request_profile" -H 'Content-Type: application/json' \
    -d "{\"run_id\": \"$2\", \"event_dir\": \"$3\"}" > /dev/null
}
for spec in $ARMS; do
  IFS=: read -r arm tree mode <<< "$spec"
  repo="$RUN_ROOT/$tree"
  dir="$OUT/$arm"
  mkdir -p "$dir"
  cd "$repo"
  echo "$(date -u +%FT%TZ) start $arm $(git rev-parse --short HEAD)" | tee -a "$OUT/events.log"
  PYTHONPATH="$repo" setsid "$PY" -m sglang_omni.cli serve --model-path "$MODEL" --host 127.0.0.1 \
    --port "$PORT" > "$dir/server.log" 2>&1 &
  server=$!
  until curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; do
    kill -0 "$server" 2> /dev/null || { echo "$(date -u +%FT%TZ) FAIL $arm server exited" | tee -a "$OUT/events.log"; exit 1; }
    sleep 5
  done
  bench "$dir/warmup" 4 1000 40
  bench "$dir/r1-cold" 1 100 60
  events start "$arm-hot" "$dir/events-hot"
  bench "$dir/r1-hot" 1 100 60
  events stop "$arm-hot" "$dir/events-hot"
  if [ "$mode" = full ]; then
    bench "$dir/r20" 20 0 1088
  else
    :
  fi
  echo "$(date -u +%FT%TZ) done $arm" | tee -a "$OUT/events.log"
  kill -- "-$server"; wait "$server" || true; wait_gpu_free
done
echo "$(date -u +%FT%TZ) INPROC_DONE" | tee -a "$OUT/events.log"
