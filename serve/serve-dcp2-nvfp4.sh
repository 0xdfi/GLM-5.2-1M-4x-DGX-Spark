#!/usr/bin/env bash
# DCP2 + NVFP4 candidate serve — adapts the proven stock DCP2/320K recipe
# (TP4/DCP2, CUDA graphs capture 16, 8192 batch, MTP-k5, async) with the
# candidate's NVFP4 compact KV (368-byte) instead of stock FP8 (656-byte).
# NVFP4 is ~58% of FP8, so 320K DCP2 KV ~= 5.1 GB (vs stock 8.8 GB) -> room for graphs.
set -euo pipefail
HEAD_NAME="${HEAD_NAME:-glm-exp1-head}"
HEAD_IP="${HEAD_IP:-192.168.100.10}"
RAY_PORT="${RAY_PORT:-26479}"
HOST="${HOST:-192.168.100.10}"
PORT="${PORT:-8210}"
HS_IFACE="${HS_IFACE:-enp1s0f1np1}"
LOG_FILE="${LOG_FILE:-/exp1-evidence/runtime-dcp2-nvfp4.log}"
RAY_SESSION_ID="${RAY_SESSION_ID:?RAY_SESSION_ID required}"
DCP_SIZE="${DCP_SIZE:-2}"
DCP_COMM_BACKEND="${DCP_COMM_BACKEND:-ag_rs}"
NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-320000}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-6000000000}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-3}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
LONG_PREFILL="${LONG_PREFILL:-8192}"
CUDAGRAPH="${CUDAGRAPH:-16}"

INDEX_TOPK_PATTERN='FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS'
HF_OVERRIDES="{\"use_index_cache\":true,\"index_topk_pattern\":\"${INDEX_TOPK_PATTERN}\"}"
SPEC='{"model":"/models","method":"mtp","num_speculative_tokens":5,"moe_backend":"flashinfer_cutlass","draft_attention_backend":"B12X_MLA_SPARSE","draft_sample_method":"probabilistic"}'

ARGS=(
  python3 -m vllm.entrypoints.openai.api_server
  --model /models --tokenizer /models --served-model-name glm-5.2
  --trust-remote-code --download-dir /models --load-format auto
  --quantization compressed-tensors --distributed-executor-backend ray
  --tensor-parallel-size 4
  --decode-context-parallel-size "${DCP_SIZE}"
  --dcp-comm-backend "${DCP_COMM_BACKEND}" --dcp-kv-cache-interleave-size 1 --pipeline-parallel-size 1
  --gpu-memory-utilization 0.88
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --generation-config vllm
  --override-generation-config '{"temperature":1.0,"top_p":0.95,"top_k":40}'
  --hf-overrides "${HF_OVERRIDES}"
  --port "${PORT}" --host "${HOST}"
  --no-enable-log-requests --no-enable-prefix-caching
  --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}"
  --kv-cache-dtype nvfp4_ds_mla
  --attention-backend B12X_MLA_SPARSE --moe-backend flashinfer_cutlass
  --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice
  --speculative-config "${SPEC}"
  --long-prefill-token-threshold "${LONG_PREFILL}"
  --async-scheduling
)
# Speed mode: ENFORCE_EAGER=1 -> eager (frees graph memory for max KV/context);
# else capture CUDA graphs (faster decode, more memory).
if [[ "${ENFORCE_EAGER:-0}" == "1" ]]; then
  ARGS+=(--enforce-eager)
else
  ARGS+=(--max-cudagraph-capture-size "${CUDAGRAPH}")
fi

if [[ "${RENDER_ONLY:-0}" == "1" ]]; then printf '%q ' "${ARGS[@]}"; printf '\n'; exit 0; fi

docker exec "${HEAD_NAME}" bash -lc \
  "test \"\$(cat /tmp/exp1-ray-session-id)\" = '${RAY_SESSION_ID}'; ray status --address=${HEAD_IP}:${RAY_PORT} | grep -q '/4.0 GPU'"

printf -v esc '%q ' "${ARGS[@]}"
launch="set -euo pipefail; test \"\$(cat /tmp/exp1-ray-session-id)\" = '${RAY_SESSION_ID}'; printf '%s\\n' '${esc}' > '${LOG_FILE}'; exec ${esc} >> '${LOG_FILE}' 2>&1"
docker exec -d \
  -e SAFETENSORS_FAST_GPU=1 -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_DEVICE_MAX_CONNECTIONS=32 \
  -e CUTE_DSL_ARCH=sm_121a -e TORCH_CUDA_ARCH_LIST=12.1a -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e NCCL_SOCKET_IFNAME="${HS_IFACE}" -e GLOO_SOCKET_IFNAME="${HS_IFACE}" -e NCCL_IB_DISABLE=0 \
  -e NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS}" -e NCCL_MIN_NCHANNELS="${NCCL_MAX_NCHANNELS}" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_USE_B12X_SPARSE_INDEXER=1 \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB="${SPARSE_LOGITS_MB:-256}" \
  -e GLM52_PAGED_MQA_TOPK_CHUNK_SIZE="${TOPK_CHUNK:-8192}" \
  `# NOTE: do NOT enable GLM52_PAGED_MQA_TRITON / GLM52_MQA_LOGITS_TRITON / GLM52_B12X_MLA here —` \
  `# tested 2026-07-21: they REGRESS decode (26.5 -> 7.9 @128K) on our image, which lacks the` \
  `# patched sparse_attn_indexer + tuned Triton kernels those flags assume. Our default path is faster.` \
  `# ^ sparse-indexer per-step cost caps (credit: XanuNetworks) — chunk the topk scan +` \
  `# cap logits memory so decode does not crawl O(context) at depth.` \
  `# ^ Marlin MoE atomic-add — proven-kept speed lever (credit: tonyd2wild Speed-Night 2).` \
  `# Capture-size lever: for concurrency (MAX_NUM_SEQS>1), align --max-cudagraph-capture-size to a` \
  `# multiple of (num_speculative_tokens+1)=6 that covers MAX_NUM_SEQS*6, else cN decode runs piecewise.` \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=0 -e USES_B12X=True -e RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}" \
  "${HEAD_NAME}" bash -lc "${launch}"
echo "Started DCP2/NVFP4 serve: ctx=${MAX_MODEL_LEN} kvbytes=${KV_CACHE_MEMORY_BYTES} graphs=${CUDAGRAPH} batch=${MAX_NUM_BATCHED_TOKENS}"
