# Team 5 – Parallel SpMV on a Multicore CPU (Windows)

This project implements the Team 5 Problem 5 scope from the supplied brief:
- Matrix Market coordinate input (real/integer/pattern; general/symmetric)
- COO, CSR and ELL sequential SpMV
- OpenMP COO (atomic row accumulation), CSR and ELL parallel SpMV
- static, dynamic and guided scheduling for row-based kernels
- repeated timing, warm-up exclusion, CSV output in the assignment schema
- matrix metadata, deterministic vector generation, correctness checks
- hardware/STREAM capture helpers, plotting and report skeleton
- serial vs parallel first-touch initialisation experiment
- OpenMP affinity/pinning experiment settings

## Important scope and scientific cautions
1. The brief requires 10–15 SuiteSparse matrices across regular/banded, power-law/irregular and very large classes. This package does not fabricate a dataset. Put downloaded `.mtx` files in `datasets/original/`, then register them in `config/matrices.csv`.
2. Run all teams against the exact same matrix files and the exact same saved vector files. Share matrix checksums, dimensions, NNZ and vectors with Teams 6 and 7.
3. COO parallel accumulation uses `omp atomic` because multiple nonzeros can update the same output row. This is correct but may be slower on some matrices.
4. ELL stores a fixed row width equal to the maximum row length. For highly irregular matrices this can use substantial padding memory. The program reports padding ratio; very large matrices may require excluding ELL only if it cannot fit, and the exclusion must be documented rather than silently omitted.
5. Windows NUMA support depends on the machine's NUMA topology and Windows APIs. OpenMP first-touch is a controlled initialisation comparison, but do not claim NUMA locality effects on a single-NUMA-node PC. Hardware capture records NUMA nodes where Windows exposes them.
6. GCC/MinGW OpenMP affinity support varies by build. `OMP_PROC_BIND` and `OMP_PLACES` are attempted and recorded; verify actual support. For strict Windows CPU affinity, use Windows Task Manager/Process Explorer or a Windows API implementation and document the method.
7. STREAM is a separate benchmark. Follow the included instructions and use a STREAM version/license appropriate to your institution. Do not substitute an invented bandwidth number.
8. The assignment requests peak GFLOP/s estimation for the common baseline. Record the method and CPU vendor specifications; STREAM gives measured memory bandwidth, not peak compute.
9. Raw CSV stores each measured repetition (including warm-up marker). Processed CSV excludes warm-up and contains mean and standard deviation.
10. Results are not pre-filled. Never report sample/example values as measured results.

## Requirements
- Windows 10/11
- MinGW-w64 GCC with OpenMP support (or another GCC build supporting `-fopenmp`)
- Python 3.10+
- Python packages: `numpy`, `pandas`, `matplotlib`
- Optional: `7z` for archives, `git`

## Setup
Open PowerShell in this folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
gcc --version
New-Item -ItemType Directory -Force bin | Out-Null
gcc -O3 -std=c11 -Wall -Wextra -fopenmp -march=native src\spmv.c -o bin\spmv.exe
```

If PowerShell blocks activation:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Input setup
1. Obtain 10–15 matrices from https://sparse.tamu.edu/ (SuiteSparse Matrix Collection). IMPORTANT: the collection site posted an outage notice on 16 Oct 2026 due to unusually high download traffic; it may return HTTP 403. If it is still unavailable, ask your instructor/cluster teams for the already-downloaded original `.mtx` files. Do not use unverified third-party copies or silently substitute generated matrices.
2. Extract each `.mtx` into `datasets/original/`.
3. Add one row per matrix to `config/matrices.csv`:
   `matrix_id,relative_path,category,group,name`
   Use the exact SuiteSparse group and matrix name shown on its collection page. Categories: `regular`, `power_law`, `large`, or a justified combination.
4. Download the registered matrices (after checking the collection entries):
```powershell
python scripts\download_matrices.py
```
5. Generate a fixed vector once per matrix and retain it:
```powershell
python scripts\prepare_inputs.py
```
This creates `datasets\vectors\<matrix_id>_x.txt`, metadata, SHA-256 checksums and a static row-partition imbalance table. The generated vector is deterministic from the matrix ID and saved for all cluster teams.
6. Copy the original `.mtx`, vector, metadata and checksum manifest to the shared cluster folder.

## Run a quick correctness test
```powershell
python scripts\make_test_matrix.py
New-Item -ItemType Directory -Force bin | Out-Null
gcc -O3 -std=c11 -Wall -Wextra -fopenmp -march=native src\spmv.c -o bin\spmv.exe
.\bin\spmv.exe --matrix datasets\original\test_small.mtx --vector datasets\vectors\test_small_x.txt --format csr --mode omp --schedule static --threads 4 --chunk 32 --runs 6 --raw results\raw\smoke.csv
```
The first run is warm-up (`warmup=1`); subsequent runs are measured.

## Run the full benchmark
```powershell
python scripts\run_benchmarks.py
```
Edit `config/experiment.json` first. Start with a small matrix subset and thread counts that exist on your PC. Full matrix/format/thread/schedule combinations can take a long time.

## Hardware and STREAM
```powershell
powershell -ExecutionPolicy Bypass -File scripts\collect_hardware.ps1
```
For the first-touch experiment, run `python scripts\run_first_touch.py`; it compiles and runs `src/first_touch.c`.
STREAM:
- Obtain the official STREAM source from https://www.cs.virginia.edu/stream/
- Compile using a supported compiler and OpenMP.
- Use an array size large enough that the arrays exceed last-level cache (follow STREAM's guidance).
- Record compiler command, array size, thread count, all four kernel bandwidths, run date and machine details in `benchmarks/stream/stream_results.csv`.
- Do not use a tiny array that fits in cache as the reported DRAM bandwidth.

## First-touch and pinning
```powershell
python scripts\run_first_touch.py
```
This measures serial vs parallel initialisation followed by the same memory-touch workload. It is a controlled experiment, not proof of multi-node NUMA locality. For OpenMP affinity:
```powershell
$env:OMP_PROC_BIND="true"
$env:OMP_PLACES="cores"
python scripts\run_benchmarks.py
```
Also run with:
```powershell
$env:OMP_PROC_BIND="false"
```
Record both settings. Confirm the GCC OpenMP runtime actually honours them.

## Analysis and graphs
```powershell
python scripts\analyze_results.py
python scripts\validate_results.py
```
Creates processed means/stddev and the five assignment graphs in `results/plots/`.

## Common CSV schema
The raw benchmark writes:
`cluster,problem,platform,implementation,dataset,size,threads_or_ranks,run_id,total_ms,kernel_ms,transfer_ms,throughput,error,power_w`
For Team 5, platform is `CPU`; transfer_ms and power_w are blank/not measured unless you actually measure them. `throughput` is GFLOP/s. `error` is maximum absolute difference from reference output. The raw CSV has additional `warmup`, `schedule`, `chunk`, `first_touch`, and `proc_bind` fields after the required columns.

## Handoff to Cluster B
After all experiments and analysis, run `python scripts\package_handoff.py`. Review and complete `shared_with_cluster/HANDOFF_TEMPLATE.md`, then share:
- original `.mtx` files (do not silently convert/change matrix)
- deterministic vector files
- metadata with rows, cols, NNZ, row-length mean/std/max and category
- SHA-256 checksums
- reference output vectors and correctness tolerances
- optimized sequential baseline source and executable build command
- raw CSV and processed CSV
- plots and hardware/STREAM details
- a short `HANDOFF.md` explaining compiler, flags, thread settings, timing method and limitations

Teams 6 and 7 must use the same matrix and vector for fair comparisons. Team 8 needs the exact raw/processed CSV and consistent identifiers.

## Deliverables checklist
See `report/TEAM5_CHECKLIST.md`. Fill every item with actual evidence before submission.


---

# Team 7 – CUDA Sparse Matrix-Vector Multiplication (SpMV) on NVIDIA GPUs

## 1. Overview and Architecture
This section documents the CUDA implementation and evaluation framework for Team 7 (Problem 7):
- **Scalar CSR CUDA Kernel** (`src/scalar_csr.cu`): 1 thread per matrix row.
- **Vector CSR CUDA Kernel** (`src/vector_csr.cu`): 1 warp (32 threads) per row with intra-warp `__shfl_down_sync` reduction.
- **ELLPACK CUDA Kernel** (`src/ell.cu`): Column-major layout (`index = col_pos * num_rows + row`) with zero padding and 100% coalesced memory access.
- **HYB CUDA Kernel** (`src/hyb.cu`): Column-major ELL for regular rows + COO overflow with `atomicAdd` accumulation.
- **NVIDIA cuSPARSE SpMV** (`src/cusparse_spmv.cu`): Industrial reference baseline using modern generic descriptors (`cusparseSpMV`).
- **Sequential CPU Baseline** (`src/cpu_spmv.cpp`): -O3 compiled sequential baseline for speedup calculation.

## 2. Compilation
```bash
# Build sequential CPU baseline
make cpu

# Build all CUDA kernels (requires nvcc and NVIDIA GPU)
make cuda

# Or run the universal build script
./scripts/build.sh
```

## 3. Automated Benchmarking
```bash
# Run complete benchmark suite across all matrices
./scripts/run_benchmarks.sh

# Run end-to-end pipeline (hardware capture, build, verification, benchmark, plots)
./scripts/run_all.sh
```

## 4. Google Colab Workflow
Open `Team7_CUDA_SpMV.ipynb` in Google Colab:
1. Select an NVIDIA GPU runtime (Runtime -> Change runtime type -> T4 / V100 / A100 / L4).
2. Execute the notebook cells sequentially to compile, test correctness, benchmark, profile, and view publication graphs inline.
