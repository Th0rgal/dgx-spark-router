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
    echo "nemotron-3-super qwen3.6 qwen3.6-27b gemma-4 gemma-heretic-smoke step3p7-flash-148b qwen3.6-aeon-dflash"
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
        qwen3.6-27b)
            VR_REPO="Qwen/Qwen3.6-27B"
            VR_SERVED="qwen3.6-27b"
            VR_IMAGE="ghcr.io/aeon-7/aeon-vllm-ultimate:latest"
            VR_MAXLEN=65536
            VR_KV_DTYPE="auto"
            VR_MAXSEQS=4
            VR_GPU_UTIL="0.85"
            VR_REASONING_PARSER="qwen3"
            VR_TOOL_PARSER="qwen3_coder"
            VR_ARGS+=(
                --max-num-batched-tokens 16384
                --enable-chunked-prefill
                --enable-prefix-caching
            )
            VR_ENV=(
                VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
                PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
            )
            ;;
        gemma-4)
            VR_REPO="nvidia/Gemma-4-26B-A4B-NVFP4"
            VR_SERVED="gemma-4"
            VR_IMAGE="nvcr.io/nvidia/vllm:26.05.post1-py3"
            VR_MAXLEN=65536
            VR_MAXSEQS=4
            # Gemma-4 is a multimodal NVFP4 MoE; it needs two GB10-specific fixes:
            #  1) vLLM forces --disable_chunked_mm_input for its bidirectional vision
            #     attention, which then requires max_num_batched_tokens >= 2496.
            #  2) The NVFP4 MoE oracle crashes (AVAILABLE_BACKENDS.remove on a backend
            #     that was never registered) whenever VLLM_USE_FLASHINFER_MOE_FP4 is
            #     set on this box. Drop the FlashInfer MoE vars and force the stable
            #     Marlin NVFP4 path; for a modelopt-NVFP4 model FORCE_FP8_MARLIN only
            #     selects the Marlin MoE kernel (GEMM already uses Marlin).
            VR_ARGS+=(--max-num-batched-tokens 8192)
            VR_ENV=(
                VLLM_NVFP4_GEMM_BACKEND=marlin
                VLLM_TEST_FORCE_FP8_MARLIN=1
                VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
                PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
            )
            ;;
        gemma-heretic-smoke)
            VR_REPO="local/gemma-4-31b-it-heretic-smoke-nvfp4"
            VR_LOCAL_DIR="$HOME/heretic-gemma-4-31b/20260608T081632Z/gemma-4-31b-it-heretic-smoke-nvfp4-export"
            VR_SERVED="gemma-heretic-smoke"
            VR_IMAGE="nvcr.io/nvidia/vllm:26.05.post1-py3"
            VR_MAXLEN=8192
            VR_MAXSEQS=1
            VR_ARGS+=(--max-num-batched-tokens 8192)
            VR_ENV=(
                VLLM_NVFP4_GEMM_BACKEND=marlin
                VLLM_TEST_FORCE_FP8_MARLIN=1
                VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
                PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
            )
            ;;
        step3p7-flash-148b)
            VR_REPO="0xSero/Step-3.7-Flash-148B"
            VR_SERVED="step3p7-flash-148b"
            VR_IMAGE="vllm/vllm-openai:stepfun37"
            VR_MAXLEN=8192
            VR_GPU_UTIL="0.92"
            VR_REASONING_PARSER="step3p5"
            VR_TOOL_PARSER="step3p5"
            VR_ARGS+=(--enable-expert-parallel --quantization modelopt --async-scheduling)
            ;;
        qwen3.6-aeon-dflash)
            VR_REPO="AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP-XS"
            VR_DRAFT_REPO="z-lab/Qwen3.6-27B-DFlash"
            VR_SPEC_TOKENS=12
            VR_SERVED="qwen3.6-aeon-dflash"
            VR_IMAGE="ghcr.io/aeon-7/aeon-vllm-ultimate:latest"
            VR_MAXLEN=65536
            # DFlash uses non-causal attention in the drafter. The AEON vLLM
            # image rejects fp8 KV cache for that attention path, so let vLLM
            # choose the compatible cache dtype for this model.
            VR_KV_DTYPE="auto"
            VR_MAXSEQS=16
            VR_GPU_UTIL="0.85"
            VR_REASONING_PARSER="qwen3"
            VR_TOOL_PARSER="qwen3_coder"
            VR_ARGS+=(
                --quantization modelopt
                --mamba-cache-dtype float16
                --mamba-block-size 256
                --limit-mm-per-prompt '{"image":4,"video":2}'
                --mm-encoder-tp-mode data
                --max-num-batched-tokens 16384
                --enable-chunked-prefill
                --enable-prefix-caching
            )
            # The AEON container has its own sm_121a/CUTLASS NVFP4 path. Do not
            # inherit the router default that forces Marlin on other NVFP4 models.
            VR_ENV=(
                VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
                PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
            )
            ;;
        *)
            return 1
            ;;
    esac
    return 0
}
