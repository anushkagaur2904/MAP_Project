#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p results/raw results/summary results/plots

RAW_CSV="results/raw/team7_raw.csv"
SUMMARY_CSV="results/summary/team7_summary.csv"
CONFIG_FILE="config/matrices.csv"

RUNS="${RUNS:-10}"
WARMUP="${WARMUP:-1}"

echo "==========================================================="
echo "Team 7 CUDA SpMV Benchmarking Suite"
echo "==========================================================="
echo "Runs per configuration: $RUNS (Warmup: $WARMUP)"
echo "Raw CSV: $RAW_CSV"
echo "Summary CSV: $SUMMARY_CSV"
echo "==========================================================="

# Check which binaries are available
HAVE_CPU=0
HAVE_GPU=0

if [ -f "bin/cpu_spmv" ]; then HAVE_CPU=1; fi
if [ -f "bin/scalar_csr" ] && [ -f "bin/vector_csr" ] && [ -f "bin/ell" ] && [ -f "bin/hyb" ] && [ -f "bin/cusparse_spmv" ]; then
    if command -v nvidia-smi &> /dev/null; then
        HAVE_GPU=1
    fi
fi

if [ "$HAVE_CPU" -eq 0 ] && [ "$HAVE_GPU" -eq 0 ]; then
    echo "ERROR: No binaries found. Run ./scripts/build.sh first."
    exit 1
fi

MATRICES=()

# Always include test_small first for rapid validation
if [ -f "datasets/original/test_small.mtx" ]; then
    MATRICES+=("test_small,datasets/original/test_small.mtx,datasets/vectors/test_small_x.txt,reference_outputs/test_small_y_reference.txt")
fi

if [ -f "$CONFIG_FILE" ]; then
    while IFS=, read -r mid rpath cat grp name; do
        # Skip header
        if [ "$mid" = "matrix_id" ]; then continue; fi
        # Trim carriage returns
        mid=$(echo "$mid" | tr -d '\r')
        rpath=$(echo "$rpath" | tr -d '\r')
        
        vec_path="datasets/vectors/${mid}_x.txt"
        ref_path="reference_outputs/${mid}_y_reference.txt"
        
        if [ -f "$rpath" ] && [ -f "$vec_path" ]; then
            MATRICES+=("${mid},${rpath},${vec_path},${ref_path}")
        fi
    done < "$CONFIG_FILE"
fi

echo "Total matrices to evaluate: ${#MATRICES[@]}"

for item in "${MATRICES[@]}"; do
    IFS=, read -r mid mpath vpath rpath <<< "$item"
    echo ""
    echo ">>> Processing Matrix: $mid ($mpath) <<<"

    REF_FLAG=""
    if [ -f "$rpath" ]; then
        REF_FLAG="--reference $rpath"
    fi

    # 1. CPU Baseline
    if [ "$HAVE_CPU" -eq 1 ]; then
        echo "  [CPU] Running sequential CSR baseline..."
        ./bin/cpu_spmv --matrix "$mpath" --vector "$vpath" $REF_FLAG \
            --dataset "$mid" --runs "$RUNS" --warmup "$WARMUP" \
            --raw "$RAW_CSV" --summary "$SUMMARY_CSV" --quiet
    fi

    # 2. GPU Implementations
    if [ "$HAVE_GPU" -eq 1 ]; then
        echo "  [GPU] Running Scalar CSR (threads per block: 128, 256, 512)..."
        for bs in 128 256 512; do
            ./bin/scalar_csr --matrix "$mpath" --vector "$vpath" $REF_FLAG \
                --dataset "$mid" --block-size "$bs" --runs "$RUNS" --warmup "$WARMUP" \
                --raw "$RAW_CSV" --summary "$SUMMARY_CSV" --quiet
        done

        echo "  [GPU] Running Vector CSR (block size: 64, 128, 256, 512; 2, 4, 8, 16 warps)..."
        for bs in 64 128 256 512; do
            ./bin/vector_csr --matrix "$mpath" --vector "$vpath" $REF_FLAG \
                --dataset "$mid" --block-size "$bs" --runs "$RUNS" --warmup "$WARMUP" \
                --raw "$RAW_CSV" --summary "$SUMMARY_CSV" --quiet
        done

        echo "  [GPU] Running ELLPACK (standard and tuned widths)..."
        ./bin/ell --matrix "$mpath" --vector "$vpath" $REF_FLAG \
            --dataset "$mid" --runs "$RUNS" --warmup "$WARMUP" \
            --raw "$RAW_CSV" --summary "$SUMMARY_CSV" --quiet

        echo "  [GPU] Running HYB (auto width and varying widths)..."
        ./bin/hyb --matrix "$mpath" --vector "$vpath" $REF_FLAG \
            --dataset "$mid" --runs "$RUNS" --warmup "$WARMUP" \
            --raw "$RAW_CSV" --summary "$SUMMARY_CSV" --quiet

        # Tuning experiment: run HYB with multiple widths (e.g., 4, 8, 16, 32, 64)
        for w in 8 16 32; do
            ./bin/hyb --matrix "$mpath" --vector "$vpath" $REF_FLAG \
                --dataset "$mid" --ell-width "$w" --runs "$RUNS" --warmup "$WARMUP" \
                --raw "$RAW_CSV" --summary "$SUMMARY_CSV" --quiet
        done

        echo "  [GPU] Running cuSPARSE SpMV reference..."
        ./bin/cusparse_spmv --matrix "$mpath" --vector "$vpath" $REF_FLAG \
            --dataset "$mid" --runs "$RUNS" --warmup "$WARMUP" \
            --raw "$RAW_CSV" --summary "$SUMMARY_CSV" --quiet
    fi
done

echo ""
echo "==========================================================="
echo "Benchmarking completed successfully!"
echo "Raw measurements saved to:     $RAW_CSV"
echo "Aggregated summary saved to:   $SUMMARY_CSV"
echo "==========================================================="
