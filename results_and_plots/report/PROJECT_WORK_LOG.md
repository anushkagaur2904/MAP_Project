# Team 5 – Project Work Log
## Parallel Sparse Matrix–Vector Multiplication on a Multicore CPU

**Cluster:** B  
**Problem:** 5 – Parallel SpMV on a multicore CPU  
**Team:** 5

This document records the work completed during the project, from environment setup through implementation, experiments, analysis and handoff.

## 1. Project understanding and planning

The assignment required a comparison of COO, CSR and ELL sparse matrix storage formats, with sequential and OpenMP implementations.

The planned work included:
- Measuring CPU memory bandwidth using STREAM.
- Implementing and validating the three SpMV formats.
- Testing OpenMP static, dynamic and guided scheduling.
- Evaluating 10–15 SuiteSparse matrices.
- Studying row-length imbalance and ELL padding.
- Testing parallel initialization and thread binding.
- Producing performance graphs, a report and a reusable handoff package.

The project was scoped to CPU/OpenMP. MPI and CUDA were not part of Team 5's implementation.

## 2. Development environment

The experiments were conducted on:

| Component | Details |
|---|---|
| Operating system | Windows 11 Home, 64-bit |
| CPU | Intel Core i5-1235U |
| Physical cores | 10 |
| Logical processors | 12 |
| RAM | 16 GiB |
| Compiler | GCC 16.2.0 |
| Environment | MSYS2 UCRT64 |
| OpenMP version macro | `_OPENMP=202111` |
| SpMV build flags | `-O3 -fopenmp -Wall -Wextra` |

Python was used for input preparation, experiment orchestration, result processing, validation and plotting.

## 3. STREAM bandwidth measurement

STREAM was compiled with optimization and OpenMP:

`gcc -O3 -fopenmp -DSTREAM_ARRAY_SIZE=50000000 -DNTIMES=10 stream.c -o stream.exe`

The array size was 50 million elements, with 10 timing iterations. Validation passed.

The highest measured Triad bandwidth was **12.7907 GB/s at 6 threads**. The measurements for all tested thread counts are stored in `benchmarks/stream/STREAM_RESULTS.md`.

This value is measured bandwidth, not theoretical peak memory bandwidth.

## 4. SpMV implementation

The C implementation was developed in `src/spmv.c`.

The implementation supports:
- COO, CSR and ELL storage formats.
- Sequential and OpenMP execution.
- Static, dynamic and guided scheduling.
- Configurable thread counts and chunk size.
- Repeated timing and CSV output.
- Reference-result checking.

The implementation was compiled using:

`gcc -O3 -fopenmp -Wall -Wextra src/spmv.c -o src/spmv.exe -lm`

All tested formats were checked against a reference SpMV computation. The correctness tolerance was applied by the benchmark program.

COO parallel accumulation uses atomic updates because multiple nonzeros can update the same output row. CSR and ELL use row-oriented computation.

## 5. SuiteSparse dataset preparation

Thirteen SuiteSparse matrices were selected and extracted:

1. appu
2. uni_chimera_i4
3. soc-sign-Slashdot090221
4. wy2010
5. mixtank_new
6. Goodwin_071
7. bbmat
8. ns3Da
9. sparsine
10. bcsstk35
11. av41092
12. TSOPF_FS_b39_c19
13. qpband

The matrices were converted/prepared for the benchmark workflow. Deterministic input vectors were generated and retained. Matrix metadata includes dimensions, effective nonzeros, average and maximum row lengths, and irregularity information.

The prepared metadata is in `datasets/metadata/`.

## 6. Benchmark configuration and execution

The main experiment tested:
- COO, CSR and ELL.
- Sequential and OpenMP execution.
- 1, 4 and 8 threads.
- Static, dynamic and guided OpenMP schedules.
- Chunk size 16.
- One warm-up and five measured runs per configuration.
- 100 SpMV repetitions in each timed run.

The main raw file is:

`results/raw/team5_suitesparse_raw.csv`

It contains 2,100 rows: 350 configurations, each with six observations including the warm-up.

ELL was excluded for four matrices where the padding ratio exceeded 20. This exclusion is controlled by the experiment configuration and is documented in the results.

## 7. Additional experiments

### First-touch initialization

A separate C program, `src/first_touch.c`, was used to compare serial and parallel initialization for arrays of 10 million and 50 million doubles.

For the 50-million-element case, parallel initialization reduced measured initialization time at several thread counts. For example:
- 4 threads: 98 ms serial versus 61 ms parallel.
- 8 threads: 108 ms serial versus 63 ms parallel.

These results show initialization-time differences. They do not prove NUMA locality because NUMA topology and memory placement were not verified.

Raw results are in `results/raw/first_touch.csv`.

### Thread binding

Thread binding was attempted on sparsine and TSOPF_FS_b39_c19 with 1, 4 and 8 threads.

When `OMP_PROC_BIND=true` was used, the runtime reported:

`libgomp: Affinity not supported on this configuration`

Therefore successful processor pinning was not established. The experiment is retained as an attempted test, not as evidence of a pinning benefit.

Raw results are in `results/raw/thread_pinning.csv`.

## 8. Result analysis

The analysis script is:

`scripts/analyze_results.py`

It produces:
- `results/processed/team5_summary.csv`
- Per-matrix GFLOP/s graphs.
- Per-matrix speedup graphs.
- Scheduling comparison.
- Load imbalance versus matrix irregularity.
- Bandwidth-only roofline plot.

The aggregate mean GFLOP/s values were:

| Implementation | Mean GFLOP/s |
|---|---:|
| COO sequential | 1.2882 |
| COO OpenMP | 0.3072 |
| CSR sequential | 1.7972 |
| CSR OpenMP | 1.8731 |
| ELL sequential | 0.9937 |
| ELL OpenMP | 0.9192 |

The aggregate schedule means were:

| Schedule | Mean GFLOP/s |
|---|---:|
| Dynamic | 0.9255 |
| Guided | 1.1865 |
| Static | 1.0265 |

These are descriptive averages across different matrices and configurations. Per-matrix results should be consulted for controlled comparisons.

The roofline uses the measured STREAM bandwidth and approximate bytes-per-nonzero models. A compute ceiling was not established, so the plot is bandwidth-only.

## 9. Report and handoff preparation

The final report is:

`report/TEAM5_REPORT.md`

The report discusses the objective, setup, methodology, measured results, graphs, interpretation and limitations.

The handoff package is:

`shared_with_cluster/`

It contains selected datasets, vectors, reference outputs, source code, scripts, configuration, raw and processed results, graphs, hardware information, STREAM measurements and documentation.

A SHA-256 manifest was generated at:

`shared_with_cluster/SHA256SUMS.csv`

The final package verification checked 112 manifest entries and reported zero checksum failures.

## 10. Important limitations recorded

- NUMA locality was not verified.
- Thread affinity was unsupported by the tested GCC runtime configuration.
- The roofline has a measured bandwidth ceiling but no established FP64 compute ceiling.
- Power consumption was not measured.
- ELL was excluded for four matrices because of excessive padding.
- Experiments were conducted on Windows/MSYS2 UCRT64.

These limitations are documented rather than presented as completed measurements.
