#!/usr/bin/env bash
# note (luojiaxuan): container command for the #1998 output parity check:
# clone the pinned commit, build the CI venv, run the target unit tests, then
# run.sh over the six backend arms and compare.py over their audio.
set -uo pipefail
umask 077
: "${RUN_ROOT:?}" "${TARGET_SHA:?}"
mkdir -p "$RUN_ROOT/home/.cache" /github /root/.cache
ln -sfn "$RUN_ROOT/home" /github/home
ln -sfn /data/cache/huggingface /root/.cache/huggingface
exec > >(tee -a "$RUN_ROOT/driver.log") 2>&1
ev() { echo "$(date -u +%FT%TZ) $*" | tee -a "$RUN_ROOT/events.log"; }
fail() { ev "FAIL $*"; exit 1; }
export HOME=/github/home OMNI_CI_HOME=/github/home/ci HF_HOME=/root/.cache/huggingface
export XDG_CACHE_HOME=/github/home/ci/.cache HF_ENDPOINT=https://huggingface.co HF_HUB_DISABLE_XET=0
export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple UV_CACHE_DIR=/github/home/.cache/uv
export TORCHINDUCTOR_CACHE_DIR=/github/home/ci/.torchinductor SGLANG_CACHE_DIR=/github/home/sglang-cache
export FLASHINFER_WORKSPACE_BASE=/root FLASHINFER_JIT_DEBUG=0 FLASHINFER_DISABLE_VERSION_CHECK=1 SGLANG_OMNI_AUTO_CLONE=0

ev "driver start"
REPO="$RUN_ROOT/sglang-omni"
[ -d "$REPO" ] || git clone -q "$RUN_ROOT/code.bundle" "$REPO" || fail "clone"
cd "$REPO" && git checkout -q "$TARGET_SHA" || fail "checkout $TARGET_SHA"
ev "source $(git rev-parse HEAD)"
bash .github/scripts/reconcile_omni_ci_env.sh omni || fail "env reconcile"
PY="$OMNI_CI_HOME/omni/bin/python"
"$PY" -c 'import torch; print("CUDA_SMOKE_OK", torch.cuda.device_count(), torch.cuda.get_device_name(0))' > "$RUN_ROOT/cuda-smoke.log" 2>&1 || fail "cuda smoke"
ev "$(tail -1 "$RUN_ROOT/cuda-smoke.log")"
"$PY" -m pytest -q -p no:cacheprovider tests/unit_test/scheduling/test_prefill_graph_policy.py \
  tests/unit_test/models/test_model_capabilities.py tests/unit_test/qwen3_tts/test_pipeline.py \
  tests/unit_test/fun_asr/test_pipeline.py > "$RUN_ROOT/unit-tests.txt" 2>&1 || fail "unit tests: $(tail -1 "$RUN_ROOT/unit-tests.txt")"
ev "unit tests $(tail -1 "$RUN_ROOT/unit-tests.txt")"
"$PY" -m benchmarks.dataset.prepare --dataset seedtts > "$RUN_ROOT/dataset.log" 2>&1 || fail "dataset"
ev "dataset ready"
REPO="$REPO" OUT="$RUN_ROOT/parity" PY="$PY" bash "$RUN_ROOT/run.sh" || fail "run.sh"
"$PY" "$RUN_ROOT/compare.py" "$RUN_ROOT/parity" > "$RUN_ROOT/parity.md" 2>&1 || fail "compare"
ev "DRIVER_DONE"
