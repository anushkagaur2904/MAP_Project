# CUDA Sparse Matrix-Vector Multiplication (SpMV) — Team 7

Research-grade CUDA C++ implementation, GPU profiling, and comparative benchmarking of Sparse Matrix-Vector Multiplication ($\mathbf{y} = A\mathbf{x}$) across diverse sparse matrix structures from the SuiteSparse collection.

---

## 1. Overview of SpMV Implementations

| Implementation | Source File | Thread / Warp Mapping | Primary Architectural Advantage |
|:---|:---|:---|:---|
| **Scalar CSR** | `src/scalar_csr.cu` | 1 thread per row | Simple row mapping; baseline CUDA kernel |
| **Vector CSR** | `src/vector_csr.cu` | 1 warp (32 threads) per row | 100% coalesced 128-byte DRAM transactions; intra-warp `__shfl_down_sync` reduction |
| **ELLPACK** | `src/ell.cu` | 1 thread per row, column-major | Column-major layout (`idx = col * rows + row`); perfect coalescing; 0 branch divergence |
| **HYB (ELL+COO)**| `src/hyb.cu` | Column-major ELL + COO overflow | Isolates irregular outlier rows; accumulates overflow via `atomicAdd` |
| **cuSPARSE** | `src/cusparse_spmv.cu`| Vendor-optimized generic API | High-performance industrial reference (`cusparseSpMV`) |
| **Sequential CPU**| `src/cpu_spmv.cpp` | Single-threaded CPU (-O3) | Reference baseline for speedup and error validation |

---

## 2. Directory Structure

```text
├── Makefile                   # Build automation for CPU baseline and CUDA targets
├── Team7_CUDA_SpMV.ipynb      # Google Colab end-to-end execution notebook
├── benchmarks/
│   ├── hardware/              # Hardware capture files
│   └── stream/                # STREAM benchmark records
├── config/
│   ├── experiment.json        # Benchmark parameters
│   ├── matrices.csv           # Registered SuiteSparse matrix dataset manifest
│   └── roofline.json          # Roofline configuration
├── datasets/
│   ├── metadata/              # Matrix statistics (irregularity, padding ratios, SHA256)
│   ├── original/              # Original Matrix Market (.mtx) files
│   └── vectors/               # Deterministic input vectors (x)
├── reference_outputs/         # Verified reference output vectors (y)
├── report/
│   ├── comparison_tables.csv  # Aggregated summary measurements
│   └── performance_analysis.md# Two-page comprehensive HPC performance report
├── results/
│   ├── hardware_info.txt      # Collected GPU/CPU/compiler specifications
│   ├── plots/                 # 12 publication-quality benchmark plots
│   ├── raw/                   # Raw benchmark measurements (team7_raw.csv)
│   └── summary/               # Aggregated statistics (mean, std, min, max)
├── scripts/
│   ├── build.sh               # Portable build script for CPU & CUDA
│   ├── collect_hardware.sh    # Queries GPU/CPU specifications
│   ├── generate_plots.py      # Generates 12 publication-quality plots
│   ├── profile.sh             # Nsight Compute profiling script
│   ├── run_all.sh             # One-command complete experiment pipeline
│   └── run_benchmarks.sh      # Automated benchmarking driver
└── src/
    ├── common/
    │   ├── benchmark.cpp / .h # Timing, arguments, and CSV logging
    │   ├── correctness.cpp / .h# Validation vs reference outputs
    │   ├── cuda_common.h      # CUDA error checking & GpuTimer
    │   └── matrix_loader.cpp / .h# MTX, CSR, ELLPACK, HYB loader
    ├── cpu_spmv.cpp           # Sequential CPU reference
    ├── cusparse_spmv.cu       # NVIDIA cuSPARSE SpMV
    ├── ell.cu                 # Column-major ELLPACK kernel
    ├── hyb.cu                 # Hybrid ELL+COO kernel
    ├── scalar_csr.cu          # Scalar CSR kernel
    └── vector_csr.cu          # Vector CSR kernel
```

---

## 3. Quick Start & Execution

### One-Command Full Automation:
```bash
./scripts/run_all.sh
```

### Manual Compilation:
```bash
# Build CPU baseline:
make cpu

# Build all CUDA implementations (on GPU host / Google Colab):
make cuda
```

### Running Benchmarks:
```bash
./scripts/run_benchmarks.sh
```

### Running in Google Colab:
Open `Team7_CUDA_SpMV.ipynb` in Google Colab, select an NVIDIA GPU runtime (T4, V100, A100, or L4), and execute all cells.
