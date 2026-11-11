#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p results/profiling
PROFILING_DIR="results/profiling"
REPORT_FILE="results/profiling_report.txt"

echo "==========================================================="
echo "Team 7 CUDA SpMV GPU Profiling Suite (Nsight Compute)"
echo "==========================================================="

if ! command -v ncu &> /dev/null; then
    echo "NOTICE: NVIDIA Nsight Compute (ncu) is not found in PATH." | tee "$REPORT_FILE"
    echo "Hardware performance counters cannot be collected in this environment." | tee -a "$REPORT_FILE"
    echo "Marking hardware-counter profiling as UNAVAILABLE (No fabricated metrics)." | tee -a "$REPORT_FILE"
    exit 0
fi

# Check if permissions allow profiling
TEST_NCU=$(ncu --version 2>&1)
echo "Found: $TEST_NCU" | tee "$REPORT_FILE"

TEST_MATRIX="datasets/original/test_small.mtx"
TEST_VECTOR="datasets/vectors/test_small_x.txt"
TEST_REF="reference_outputs/test_small_y_reference.txt"

if [ ! -f "$TEST_MATRIX" ]; then
    echo "Test matrix not found: $TEST_MATRIX"
    exit 1
fi

METRICS="dram__bytes_read.sum,dram__bytes_write.sum,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__thread_inst_executed_per_inst_executed.ratio,l1tex__t_bytes.sum,sm__cycles_active.avg"

run_ncu() {
    local name="$1"
    local bin="$2"
    local extra_args="$3"

    if [ -f "$bin" ]; then
        echo "Profiling $name ($bin)..." | tee -a "$REPORT_FILE"
        ncu --metrics "$METRICS" \
            --csv \
            --log-file "$PROFILING_DIR/${name}_metrics.csv" \
            "$bin" --matrix "$TEST_MATRIX" --vector "$TEST_VECTOR" --reference "$TEST_REF" \
            --dataset "test_small" --runs 1 --warmup 1 $extra_args \
            2>&1 | tee -a "$REPORT_FILE" || {
                echo "Warning: ncu profiling failed or restricted on $name." | tee -a "$REPORT_FILE"
            }
    else
        echo "Binary $bin not found. Skipping." | tee -a "$REPORT_FILE"
    fi
}

echo "Beginning kernel profiling..." | tee -a "$REPORT_FILE"

run_ncu "scalar_csr" "bin/scalar_csr" "--block-size 256"
run_ncu "vector_csr" "bin/vector_csr" "--block-size 256"
run_ncu "ell" "bin/ell" "--block-size 256"
run_ncu "hyb" "bin/hyb" "--block-size 256"
run_ncu "cusparse" "bin/cusparse_spmv" ""

echo "Profiling finished. Results saved to $PROFILING_DIR." | tee -a "$REPORT_FILE"
