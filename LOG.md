# Project Work Log — Team 7 CUDA SpMV

## Stage 1: Repository Inspection and Environment Setup
- Examined the inherited SuiteSparse dataset, configuration files, and deterministic input vectors in `shared_with_cluster/`.
- Verified 14 Matrix Market `.mtx` datasets, deterministic input vectors in `datasets/vectors/`, and reference output vectors in `reference_outputs/`.
- Established root-level symlinks so paths resolve identically across development and cluster environments.
- Confirmed zero data duplication and preserved all existing Team 5 CPU files.

## Stage 2: Matrix Loader, Correctness Validation, and CPU Baseline
- Created `src/common/matrix_loader.h` and `src/common/matrix_loader.cpp`:
  - Parses Matrix Market coordinate formats (real, integer, pattern, symmetric/hermitian expansion).
  - Implemented CSR conversion with sorted column indices per row.
  - Implemented column-major ELLPACK conversion (`idx = col_pos * num_rows + row`) with zero padding.
  - Implemented HYB conversion (ELL for regular rows, COO for overflow nonzeros).
  - Implemented deterministic dense vector loading and saving.
- Created `src/common/correctness.h` and `src/common/correctness.cpp`:
  - Implemented absolute and relative error validation accounting for floating-point reduction order differences.
  - Implemented NaN and infinity detection.
- Created `src/common/benchmark.h` and `src/common/benchmark.cpp`:
  - Implemented unified CLI argument parser.
  - Implemented exact CSV logging matching required schema.
- Implemented `src/cpu_spmv.cpp`:
  - High-performance sequential CSR SpMV compiled with `-O3`.
  - Validated against all 14 reference output vectors (passed with `max_abs_error = 0.0`).

## Stage 3: Scalar CSR CUDA Kernel
- Implemented `src/scalar_csr.cu`:
  - 1 thread per row.
  - Handles arbitrary row lengths and empty rows.
  - CUDA event timing for H2D transfer, kernel, and D2H transfer.
  - Validation against reference output vectors.

## Stage 4: Vector CSR CUDA Kernel
- Implemented `src/vector_csr.cu`:
  - 1 warp (32 threads) per row.
  - Strided access across row nonzeros achieving 100% coalesced 128-byte transactions.
  - Intra-warp reduction using `__shfl_down_sync`.
  - Configurable block sizes (64 to 512 threads, corresponding to 2 to 16 warps per block).

## Stage 5: ELLPACK CUDA Kernel
- Implemented `src/ell.cu`:
  - Column-major storage indexing: `idx = column_position * num_rows + row`.
  - Adjacent threads map to adjacent rows, guaranteeing fully coalesced global memory loads.
  - Zero-padding for unused entries, eliminating branch divergence.
  - Measures ELL padding overhead ratio.

## Stage 6: HYB CUDA Kernel
- Implemented `src/hyb.cu`:
  - Splits matrix into column-major ELL and COO overflow arrays.
  - Supports configurable and auto-tuned ELL widths.
  - Atomic accumulation (`atomicAdd`) for COO overflow avoiding race conditions.
  - Clean initialization and accumulation.

## Stage 7: NVIDIA cuSPARSE Implementation
- Implemented `src/cusparse_spmv.cu`:
  - Production reference using modern cuSPARSE generic API (`cusparseSpMV`).
  - Manages CSR and Dense vector descriptors (`cusparseCreateCsr`, `cusparseCreateDnVec`).
  - Strict CUDA and cuSPARSE error checking macros (`CUDA_CHECK`, `CUSPARSE_CHECK`).

## Stage 8: Benchmarking & Profiling Automation
- Created `scripts/build.sh` and `Makefile`:
  - Universal architecture flags (`sm_70`, `sm_75`, `sm_80`, `sm_89`).
  - Graceful fallback for non-GPU development environments.
- Created `scripts/run_benchmarks.sh`:
  - Iterates over all 14 matrices, logs raw and summary metrics.
- Created `scripts/collect_hardware.sh`:
  - Queries `nvidia-smi`, CPU topology, RAM, and compilers into `results/hardware_info.txt`.
- Created `scripts/profile.sh`:
  - Detects Nsight Compute (`ncu`) availability; records hardware counter status without fabricating data.
- Created `scripts/run_all.sh`:
  - 10-step end-to-end automation.

## Stage 9: Publication Plots & Google Colab Integration
- Created `scripts/generate_plots.py`:
  - Generates all 12 required publication-quality figures in `results/plots/`.
- Created `Team7_CUDA_SpMV.ipynb`:
  - Google Colab notebook supporting session restarts, GPU detection, automated builds, verification, and inline visualization.

## Stage 10: Performance Analysis Report
- Authored two-page technical report `report/performance_analysis.md` covering all 13 required HPC topics and architectural insights.
