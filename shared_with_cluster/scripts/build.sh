#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p bin results/raw results/summary results/plots

echo "============================================="
echo "Team 7 CUDA SpMV Build System"
echo "============================================="

CXX="${CXX:-g++}"
if ! command -v "$CXX" &> /dev/null; then
    CXX=clang++
fi

echo "[1/2] Building Sequential CPU CSR SpMV baseline with $CXX -O3..."
$CXX -O3 -std=c++17 -Wall -Wextra \
    src/cpu_spmv.cpp \
    src/common/matrix_loader.cpp \
    src/common/correctness.cpp \
    src/common/benchmark.cpp \
    -o bin/cpu_spmv
echo "  -> bin/cpu_spmv built successfully."

if command -v nvcc &> /dev/null; then
    echo "[2/2] NVIDIA CUDA compiler (nvcc) detected."
    NVCC_VERSION=$(nvcc --version | grep "release" | sed -E 's/.*release ([0-9]+.[0-9]+).*/\1/')
    echo "  -> Detected CUDA version: $NVCC_VERSION"

    # Detect GPU architecture if nvidia-smi is available, otherwise compile for common Colab architectures (T4=75, V100=70, A100=80, L4=89)
    GENCODE=""
    if command -v nvidia-smi &> /dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
        echo "  -> Detected GPU: $GPU_NAME"
    fi

    NVCC_FLAGS="-O3 -std=c++17 -Xcompiler -Wall -Xcompiler -Wextra"
    # Auto-detect target architecture from nvidia-smi if available, otherwise target T4 (sm_75) and newer architectures
    if [ -n "$GPU_NAME" ]; then
        CCAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '. \r')
        if [ -n "$CCAP" ]; then
            echo "  -> Targeting detected GPU compute capability: sm_${CCAP}"
            GENCODE="-gencode arch=compute_${CCAP},code=sm_${CCAP}"
        fi
    fi
    if [ -z "$GENCODE" ]; then
        GENCODE="-gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_89,code=sm_89"
    fi

    COMMON_OBJS="src/common/matrix_loader.cpp src/common/correctness.cpp src/common/benchmark.cpp"

    echo "  -> Compiling Scalar CSR kernel (src/scalar_csr.cu)..."
    nvcc $NVCC_FLAGS $GENCODE $COMMON_OBJS src/scalar_csr.cu -o bin/scalar_csr

    echo "  -> Compiling Vector CSR kernel (src/vector_csr.cu)..."
    nvcc $NVCC_FLAGS $GENCODE $COMMON_OBJS src/vector_csr.cu -o bin/vector_csr

    echo "  -> Compiling ELLPACK kernel (src/ell.cu)..."
    nvcc $NVCC_FLAGS $GENCODE $COMMON_OBJS src/ell.cu -o bin/ell

    echo "  -> Compiling HYB kernel (src/hyb.cu)..."
    nvcc $NVCC_FLAGS $GENCODE $COMMON_OBJS src/hyb.cu -o bin/hyb

    echo "  -> Compiling cuSPARSE SpMV (src/cusparse_spmv.cu)..."
    nvcc $NVCC_FLAGS $GENCODE $COMMON_OBJS src/cusparse_spmv.cu -lcusparse -o bin/cusparse_spmv

    echo "All 5 CUDA SpMV implementations and CPU baseline built successfully!"
else
    echo "[2/2] nvcc not detected in PATH. (Local macOS environment)"
    echo "      CPU baseline is ready. For CUDA builds, run this script in Google Colab with GPU runtime."
fi

echo "Build process completed."
