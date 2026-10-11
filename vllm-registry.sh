#!/bin/bash
# Registry of vLLM-served models for the DGX Spark router.
#
# Each key maps to a backend configuration consumed by launch-vllm.sh and
# install-vllm.sh. Only one vLLM container runs at a time (these models are too
# large to co-reside in 128GB unified memory), so the launcher always tears down
# the previous container before starting a new one.
#
# GB10 / Blackwell SM12.1 notes baked into the env defaults below:
#   VLLM_NVFP4_GEMM_BACKEND=marlin     -> stable NVFP4 GEMM path on GB10
#   VLLM_FLASHINFER_MOE_BACKEND=latency-> throughput MoE kernels are SM120-only
#   VLLM_USE_FLASHINFER_MOE_FP4=0      -> avoid the unstable FP4 MoE fastpath
# These are hardware properties shared by every NVFP4 MoE model on this box.
#
# To add a model: add its key to vllm_keys() and a case arm in vllm_config().

vllm_keys() {
    echo "qwen3.8-flash-next"
}

# Populate VR_* globals for the given model key. Returns nonzero for unknown keys.
vllm_config() {
    local key="$1"

    # ---- defaults (overridable per model below) ----
    VR_KEY="$key"
    VR_LOCAL_DIR=""
    VR_DRAFT_REPO=""
    VR_DRAFT_LOCAL_DIR=""
    VR_SPEC_TOKENS=0
    VR_IMAGE="nvcr.io/nvidia/vllm:26.03.post1-py3"
    VR_MAXLEN=32768
    VR_KV_DTYPE="fp8"
    VR_GPU_UTIL="0.85"
    VR_MAXSEQS=1
    VR_REASONING_PARSER=""      # vLLM built-in reasoning parser name ("" = none)
    VR_TOOL_PARSER=""           # vLLM tool-call parser name ("" = none)
    VR_USE_SUPERV3=0            # 1 => mount + load the super_v3 reasoning plugin
    VR_DOCKER_ARGS=()           # extra `docker run` flags (mounts, shm) for special models
    VR_ARGS=(--trust-remote-code --tensor-parallel-size 1 --disable-uvicorn-access-log)
    VR_ENV=(
        VLLM_NVFP4_GEMM_BACKEND=marlin
        VLLM_FLASHINFER_MOE_BACKEND=latency
        VLLM_USE_FLASHINFER_MOE_FP4=0
        VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    )

    case "$key" in



        qwen3.8-flash-next)
            # Qwen3.8-Flash-Next (qwen4_exp, 125B/6B active) abliterated NVFP4 build.
            # The checkpoint is 171 GiB; its 95 GiB bf16 n-gram (PLE) table is served
            # from pageable host memory backed by /swap-ple.img (VLLM_PLE_CPU_OFFLOAD),
            # leaving ~77 GiB of weights resident. Runtime and flags come from
            # ~/qwen3.8-flash-next-dgx-spark (orcarouter profile), which built the
            # patched image and wrote the config override mounted below.
            # KV is capped at 16 GiB (profile default 24) to keep ~10 GiB headroom
            # above the launcher's MemAvailable watchdog. Cold start is ~16 min.
            VR_REPO="orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4"
            VR_LOCAL_DIR="/home/th0rgal/models/qwen3.8-flash-next-orcarouter"
            VR_SERVED="qwen3.8-flash-next"
            VR_IMAGE="vllm-skinny-tp1:v1"
            VR_MAXLEN=262144
            VR_MAXSEQS=2
            VR_KV_DTYPE="auto"      # QSA requires a BF16 main KV cache
            VR_REASONING_PARSER="qwen3"
            VR_TOOL_PARSER="qwen3_coder"
            VR_ARGS+=(
                --distributed-executor-backend mp   # PLE offload hangs at TP=1 without it
                --kv-cache-memory 10737418240   # 10 GiB (was 16) to leave room for vllm-asr (Cohere Transcribe, port 8002)
                --max-num-batched-tokens 8192 --enable-chunked-prefill
                --no-async-scheduling --no-enable-prefix-caching --no-enable-flashinfer-autotune
                --limit-mm-per-prompt '{"image":4}'
                --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
            )
            VR_ENV=(
                VLLM_TARGET_DEVICE=cuda
                CUTE_DSL_ARCH=sm_121a
                VLLM_PLE_CPU_OFFLOAD=1
                VLLM_PLE_OFFLOAD_READY_TIMEOUT=1800
                FLASHINFER_DISABLE_VERSION_CHECK=1
                VLLM_QSA_DET_TOPK=0
                VLLM_QSA_EXACT_TOPK=0
            )
            VR_DOCKER_ARGS=(
                --shm-size=32g
                -v "/home/th0rgal/.local/state/qwen38-spark/config.vllm.json:${VR_LOCAL_DIR}/config.json:ro"
                -v "/home/th0rgal/.cache/flashinfer:/root/.cache/flashinfer"
                -v "/home/th0rgal/.cache/vllm-qwen38:/root/.cache/vllm"
            )
            ;;


        *)
            return 1
            ;;
    esac
    return 0
}
