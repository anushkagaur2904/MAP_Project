#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

echo "================================================================================"
echo "          TEAM 7: CUDA SPARSE MATRIX-VECTOR MULTIPLICATION (SpMV)              "
echo "                     END-TO-END AUTOMATION PIPELINE                            "
echo "================================================================================"

# Step 1: Inspect environment
echo ""
echo "[Step 1/10] Collecting hardware and environment specifications..."
chmod +x scripts/collect_hardware.sh
./scripts/collect_hardware.sh

# Step 2: Build CPU and CUDA implementations
echo ""
echo "[Step 2/10] Compiling CPU and CUDA implementations..."
chmod +x scripts/build.sh
./scripts/build.sh

# Step 3: Verify datasets
echo ""
echo "[Step 3/10] Verifying input matrix datasets and vectors..."
if [ ! -d "datasets/original" ] || [ ! -d "datasets/vectors" ]; then
    echo "ERROR: Datasets directory missing!"
    exit 1
fi
COUNT_MTX=$(find datasets/original -name "*.mtx" | wc -l | xargs)
COUNT_VEC=$(find datasets/vectors -name "*_x.txt" | wc -l | xargs)
echo "  -> Found $COUNT_MTX Matrix Market (.mtx) files."
echo "  -> Found $COUNT_VEC deterministic input vectors."

# Step 4: Run correctness validation on test_small
echo ""
echo "[Step 4/10] Executing strict correctness validation on test dataset..."
if [ -f "bin/cpu_spmv" ]; then
    ./bin/cpu_spmv --matrix datasets/original/test_small.mtx \
                   --vector datasets/vectors/test_small_x.txt \
                   --reference reference_outputs/test_small_y_reference.txt \
                   --dataset test_small --runs 2 --warmup 1 --quiet
    echo "  -> CPU sequential baseline PASSED reference verification."
fi

if [ -f "bin/scalar_csr" ] && command -v nvidia-smi &> /dev/null; then
    ./bin/scalar_csr --matrix datasets/original/test_small.mtx \
                     --vector datasets/vectors/test_small_x.txt \
                     --reference reference_outputs/test_small_y_reference.txt \
                     --dataset test_small --runs 2 --warmup 1 --quiet
    echo "  -> Scalar CSR CUDA kernel PASSED reference verification."

    ./bin/vector_csr --matrix datasets/original/test_small.mtx \
                     --vector datasets/vectors/test_small_x.txt \
                     --reference reference_outputs/test_small_y_reference.txt \
                     --dataset test_small --runs 2 --warmup 1 --quiet
    echo "  -> Vector CSR CUDA kernel PASSED reference verification."

    ./bin/ell --matrix datasets/original/test_small.mtx \
              --vector datasets/vectors/test_small_x.txt \
              --reference reference_outputs/test_small_y_reference.txt \
              --dataset test_small --runs 2 --warmup 1 --quiet
    echo "  -> ELLPACK CUDA kernel PASSED reference verification."

    ./bin/hyb --matrix datasets/original/test_small.mtx \
              --vector datasets/vectors/test_small_x.txt \
              --reference reference_outputs/test_small_y_reference.txt \
              --dataset test_small --runs 2 --warmup 1 --quiet
    echo "  -> HYB CUDA kernel PASSED reference verification."

    ./bin/cusparse_spmv --matrix datasets/original/test_small.mtx \
                        --vector datasets/vectors/test_small_x.txt \
                        --reference reference_outputs/test_small_y_reference.txt \
                        --dataset test_small --runs 2 --warmup 1 --quiet
    echo "  -> NVIDIA cuSPARSE SpMV PASSED reference verification."
else
    echo "  -> GPU execution skipped in this environment (No active NVIDIA GPU)."
fi

# Step 5: Execute benchmark experiments
echo ""
echo "[Step 5/10] Executing benchmark experiments across all matrices..."
chmod +x scripts/run_benchmarks.sh
./scripts/run_benchmarks.sh

# Step 6: Raw CSV files check
echo ""
echo "[Step 6/10] Confirming raw CSV benchmark output..."
if [ -f "results/raw/team7_raw.csv" ]; then
    RAW_LINES=$(wc -l < results/raw/team7_raw.csv | xargs)
    echo "  -> Raw CSV recorded $RAW_LINES entries in results/raw/team7_raw.csv."
else
    echo "  -> Warning: results/raw/team7_raw.csv was not generated."
fi

# Step 7: Execute profiling where supported
echo ""
echo "[Step 7/10] Executing profiling where supported..."
chmod +x scripts/profile.sh
./scripts/profile.sh

# Step 8: Generate summary tables
echo ""
echo "[Step 8/10] Generating comparison summary tables..."
if [ -f "results/summary/team7_summary.csv" ]; then
    cp results/summary/team7_summary.csv report/comparison_tables.csv
    echo "  -> Comparison table saved to report/comparison_tables.csv."
fi

# Step 9: Generate graphs
echo ""
echo "[Step 9/10] Generating publication-quality graphs..."
if command -v python3 &> /dev/null; then
    python3 scripts/generate_plots.py || echo "  -> Note: matplotlib required for plot generation (standard in Google Colab)."
fi

# Step 10: Generate / verify report
echo ""
echo "[Step 10/10] Finalizing performance analysis report..."
echo "  -> Report is available at report/performance_analysis.md."

echo ""
echo "================================================================================"
echo "          TEAM 7 PIPELINE COMPLETED SUCCESSFULLY!                              "
echo "================================================================================"
