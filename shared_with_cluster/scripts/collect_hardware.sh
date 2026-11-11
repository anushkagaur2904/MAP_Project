#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p results benchmarks/hardware
OUTPUT_FILE="results/hardware_info.txt"
JSON_FILE="results/hardware_info.json"

echo "Collecting hardware and environment information..."
{
    echo "================================================================================"
    echo "                 TEAM 7 HARDWARE & ENVIRONMENT SPECIFICATION                   "
    echo "================================================================================"
    echo "Date: $(date)"
    echo "Hostname: $(hostname)"
    echo "Kernel: $(uname -srm)"
    echo ""

    echo "--------------------------------------------------------------------------------"
    echo "1. GPU SPECIFICATIONS (via nvidia-smi)"
    echo "--------------------------------------------------------------------------------"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi
        echo ""
        echo "Detailed GPU Properties:"
        nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,power.limit,clocks.max.sm,clocks.max.memory,compute_cap --format=csv
    else
        echo "nvidia-smi: Not available in current environment (CPU only / macOS dev environment)."
    fi
    echo ""

    echo "--------------------------------------------------------------------------------"
    echo "2. CUDA & TOOLKIT SPECIFICATIONS"
    echo "--------------------------------------------------------------------------------"
    if command -v nvcc &> /dev/null; then
        nvcc --version
    else
        echo "nvcc: Not found in PATH."
    fi
    echo ""

    echo "--------------------------------------------------------------------------------"
    echo "3. CPU & SYSTEM MEMORY SPECIFICATIONS"
    echo "--------------------------------------------------------------------------------"
    if [ -f /proc/cpuinfo ]; then
        echo "CPU Model: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)"
        echo "CPU Cores: $(nproc)"
        echo "Architecture: $(uname -m)"
        echo ""
        echo "Memory Info:"
        free -h
    elif command -v sysctl &> /dev/null; then
        echo "CPU Model: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -m)"
        echo "Logical Cores: $(sysctl -n hw.ncpu 2>/dev/null || echo 'Unknown')"
        echo "Physical Cores: $(sysctl -n hw.physicalcpu 2>/dev/null || echo 'Unknown')"
        MEM_BYTES=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
        MEM_GB=$(( MEM_BYTES / 1024 / 1024 / 1024 ))
        echo "System RAM: ${MEM_GB} GB"
    fi
    echo ""

    echo "--------------------------------------------------------------------------------"
    echo "4. PROFILING & BENCHMARK TOOLS (Nsight Compute)"
    echo "--------------------------------------------------------------------------------"
    if command -v ncu &> /dev/null; then
        ncu --version
    else
        echo "Nsight Compute (ncu): Not detected. Hardware counter profiling unavailable."
    fi
    echo ""

    echo "--------------------------------------------------------------------------------"
    echo "5. COMPILER & BUILD FLAGS"
    echo "--------------------------------------------------------------------------------"
    if command -v g++ &> /dev/null; then
        echo "g++ version: $(g++ --version | head -n 1)"
    fi
    if command -v clang++ &> /dev/null; then
        echo "clang++ version: $(clang++ --version | head -n 1)"
    fi
    echo "Standard CPU CXXFLAGS: -O3 -std=c++17 -Wall -Wextra"
    echo "Standard CUDA NVCCFLAGS: -O3 -std=c++17 -gencode arch=compute_70..89"
    echo "================================================================================"
} > "$OUTPUT_FILE"

# Also generate JSON representation for roofline calculation
GPU_NAME="NVIDIA GPU"
PEAK_BANDWIDTH_GBS="320.0" # Default T4 spec
PEAK_FP64_GFLOPS="254.0"    # Default T4 spec

if command -v nvidia-smi &> /dev/null; then
    DETECTED_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | xargs)
    if [ -n "$DETECTED_NAME" ]; then
        GPU_NAME="$DETECTED_NAME"
        case "$GPU_NAME" in
            *"T4"*)
                PEAK_BANDWIDTH_GBS="320.0"
                PEAK_FP64_GFLOPS="254.0"
                ;;
            *"V100"*)
                PEAK_BANDWIDTH_GBS="900.0"
                PEAK_FP64_GFLOPS="7800.0"
                ;;
            *"A100"*)
                PEAK_BANDWIDTH_GBS="1555.0"
                PEAK_FP64_GFLOPS="9700.0"
                ;;
            *"L4"*)
                PEAK_BANDWIDTH_GBS="300.0"
                PEAK_FP64_GFLOPS="490.0"
                ;;
            *)
                PEAK_BANDWIDTH_GBS="320.0"
                PEAK_FP64_GFLOPS="500.0"
                ;;
        esac
    fi
fi

cat <<EOF > "$JSON_FILE"
{
  "gpu_name": "$GPU_NAME",
  "peak_bandwidth_gbs": $PEAK_BANDWIDTH_GBS,
  "peak_fp64_gflops": $PEAK_FP64_GFLOPS,
  "collected_date": "$(date)"
}
EOF

echo "Hardware info saved to $OUTPUT_FILE and $JSON_FILE."
