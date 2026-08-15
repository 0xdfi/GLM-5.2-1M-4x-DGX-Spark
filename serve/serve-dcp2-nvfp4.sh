#!/usr/bin/env bash
# Legacy exp1 DCP launcher. Historical evidence reproduction only.
# This is not O14 Balanced and contains no selectable capacity or private defaults.
set -euo pipefail

: "${LEGACY_EXP1_ACK:?Set LEGACY_EXP1_ACK=I_UNDERSTAND_THIS_IS_NOT_AN_O14_PROFILE}"
[[ "${LEGACY_EXP1_ACK}" == "I_UNDERSTAND_THIS_IS_NOT_AN_O14_PROFILE" ]] || {
  echo "Refusing: incorrect LEGACY_EXP1_ACK" >&2
  exit 2
}

: "${HEAD_NAME:?container name required}"
: "${HEAD_IP:?Ray head address required}"
: "${RAY_PORT:?Ray port required}"
: "${API_HOST:?API bind address required}"
: "${API_PORT:?API port required}"
: "${SOCKET_IFACE:?socket interface required}"
: "${NCCL_IB_HCA:?RDMA HCA list required}"
: "${LOG_FILE:?runtime log path required}"
: "${SESSION_FILE:?Ray session identity file required}"
: "${RAY_SESSION_ID:?Ray session identity required}"
: "${MODEL_PATH:?model/tokenizer path required}"
: "${DCP_SIZE:?legacy DCP size required}"
: "${MAX_MODEL_LEN:?legacy max model length required}"
: "${KV_CACHE_MEMORY_BYTES:?legacy KV bytes required}"
: "${MAX_NUM_SEQS:?legacy max sequences required}"
: "${MAX_NUM_BATCHED_TOKENS:?legacy batch token limit required}"

DCP_COMM_BACKEND="${DCP_COMM_BACKEND:-ag_rs}"
NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-4}"
LONG_PREFILL="${LONG_PREFILL:-8192}"
CUDAGRAPH="${CUDAGRAPH:-16}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
INDEX_TOPK_PATTERN='FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS'
HF_OVERRIDES="{\"use_index_cache\":true,\"index_topk_pattern\":\"${INDEX_TOPK_PATTERN}\"}"
SPEC="{\"model\":\"${MODEL_PATH}\",\"method\":\"mtp\",\"num_speculative_tokens\":5,\"moe_backend\":\"flashinfer_cutlass\",\"draft_attention_backend\":\"B12X_MLA_SPARSE\",\"draft_sample_method\":\"probabilistic\"}"

ARGS=(
  python3 -m vllm.entrypoints.openai.api_server
  --model "${MODEL_PATH}" --tokenizer "${MODEL_PATH}" --served-model-name glm-5.2
  --trust-remote-code --download-dir "${MODEL_PATH}" --load-format auto
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
  --port "${API_PORT}" --host "${API_HOST}"
  --no-enable-log-requests
  --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}"
  --kv-cache-dtype nvfp4_ds_mla
  --attention-backend B12X_MLA_SPARSE --moe-backend flashinfer_cutlass
  --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice
  --speculative-config "${SPEC}"
  --long-prefill-token-threshold "${LONG_PREFILL}"
  --async-scheduling
)

if [[ "${ENFORCE_EAGER:-0}" == "1" ]]; then
  ARGS+=(--enforce-eager)
else
  ARGS+=(--max-cudagraph-capture-size "${CUDAGRAPH}")
fi
if [[ "${ENABLE_PREFIX_CACHING}" == "1" ]]; then
  ARGS+=(--enable-prefix-caching)
else
  ARGS+=(--no-enable-prefix-caching)
fi

if [[ "${RENDER_ONLY:-0}" == "1" ]]; then
  printf '%q ' "${ARGS[@]}"
  printf '\n'
  exit 0
fi

printf -v session_file_q '%q' "${SESSION_FILE}"
printf -v session_id_q '%q' "${RAY_SESSION_ID}"
printf -v log_file_q '%q' "${LOG_FILE}"
docker exec "${HEAD_NAME}" bash -lc \
  "test \"\$(cat ${session_file_q})\" = ${session_id_q}; ray status --address=${HEAD_IP}:${RAY_PORT} | grep -q '/4.0 GPU'"

printf -v command_q '%q ' "${ARGS[@]}"
launch="set -euo pipefail; test \"\$(cat ${session_file_q})\" = ${session_id_q}; printf '%s\\n' '${command_q}' > ${log_file_q}; exec ${command_q} >> ${log_file_q} 2>&1"
docker exec -d \
  -e SAFETENSORS_FAST_GPU=1 -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_DEVICE_MAX_CONNECTIONS=32 \
  -e CUTE_DSL_ARCH=sm_121a -e TORCH_CUDA_ARCH_LIST=12.1a -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e NCCL_SOCKET_IFNAME="${SOCKET_IFACE}" -e GLOO_SOCKET_IFNAME="${SOCKET_IFACE}" \
  -e NCCL_IB_HCA="${NCCL_IB_HCA}" -e NCCL_IB_DISABLE=0 \
  -e NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS}" -e NCCL_MIN_NCHANNELS="${NCCL_MAX_NCHANNELS}" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_USE_B12X_SPARSE_INDEXER=1 \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB="${SPARSE_LOGITS_MB:-256}" \
  -e GLM52_PAGED_MQA_TOPK_CHUNK_SIZE="${TOPK_CHUNK:-8192}" \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=0 -e USES_B12X=True -e RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}" \
  "${HEAD_NAME}" bash -lc "${launch}"
printf 'Started legacy exp1 DCP%s/NVFP4 serve: ctx=%s kvbytes=%s graphs=%s batch=%s\n' \
  "${DCP_SIZE}" "${MAX_MODEL_LEN}" "${KV_CACHE_MEMORY_BYTES}" "${CUDAGRAPH}" "${MAX_NUM_BATCHED_TOKENS}"
