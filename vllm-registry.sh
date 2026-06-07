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
    echo "nemotron-3-super qwen3.6 gemma-4"
}

# Populate VR_* globals for the given model key. Returns nonzero for unknown keys.
vllm_config() {
    local key="$1"

    # ---- defaults (overridable per model below) ----
    VR_KEY="$key"
    VR_IMAGE="nvcr.io/nvidia/vllm:26.03.post1-py3"
    VR_MAXLEN=32768
    VR_KV_DTYPE="fp8"
    VR_GPU_UTIL="0.85"
    VR_MAXSEQS=1
    VR_REASONING_PARSER=""      # vLLM built-in reasoning parser name ("" = none)
    VR_TOOL_PARSER=""           # vLLM tool-call parser name ("" = none)
    VR_USE_SUPERV3=0            # 1 => mount + load the super_v3 reasoning plugin
    VR_ARGS=(--trust-remote-code --tensor-parallel-size 1 --disable-uvicorn-access-log)
    VR_ENV=(
        VLLM_NVFP4_GEMM_BACKEND=marlin
        VLLM_FLASHINFER_MOE_BACKEND=latency
        VLLM_USE_FLASHINFER_MOE_FP4=0
        VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    )

    case "$key" in
        nemotron-3-super)
            VR_REPO="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
            VR_SERVED="nemotron-3-super"
            VR_USE_SUPERV3=1
            VR_REASONING_PARSER="super_v3"
            VR_TOOL_PARSER="qwen3_coder"
            ;;
        qwen3.6)
            VR_REPO="nvidia/Qwen3.6-35B-A3B-NVFP4"
            VR_SERVED="qwen3.6"
            # 26.03's Qwen loader lacks the NVFP4 MoE input-scale params and dies
            # with KeyError on w2_input_scale; 26.05.post1 loads them.
            VR_IMAGE="nvcr.io/nvidia/vllm:26.05.post1-py3"
            VR_MAXLEN=65536
            VR_MAXSEQS=4
            VR_REASONING_PARSER="qwen3"
            ;;
        gemma-4)
            VR_REPO="nvidia/Gemma-4-26B-A4B-NVFP4"
            VR_SERVED="gemma-4"
            VR_IMAGE="nvcr.io/nvidia/vllm:26.05.post1-py3"
            VR_MAXLEN=65536
            VR_MAXSEQS=4
            ;;
        *)
            return 1
            ;;
    esac
    return 0
}
