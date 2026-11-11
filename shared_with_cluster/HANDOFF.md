# Team 5 – Cluster B SpMV Handoff

This package contains the CPU/OpenMP SpMV implementation and experiment results.

## Contents
- `datasets/original/`: selected SuiteSparse Matrix Market inputs
- `datasets/vectors/`: deterministic input vectors
- `datasets/metadata/`: matrix metadata and partition statistics
- `reference_outputs/`: reference output vectors
- `src/`: SpMV and first-touch C source
- `scripts/`, `config/`: experiment scripts and configuration
- `results/raw/`: benchmark, first-touch and thread-binding CSVs
- `results/processed/`: processed summaries
- `results/plots/`: generated figures
- `benchmarks/`: hardware and STREAM records
- `report/`: report draft, template and checklist

## Reproduction
Use MSYS2 UCRT64 with GCC and OpenMP. Build the SpMV program with:

`gcc -O3 -fopenmp -Wall -Wextra src/spmv.c -o src/spmv.exe -lm`

See `BUILD_AND_METHOD.md` and the files in `config/` and `scripts/` for experiment details.

## Important limitations
- STREAM Triad measured 12.7907 GB/s at 6 threads.
- NUMA locality was not verified.
- Thread affinity was attempted, but this libgomp configuration reported affinity unsupported.
- The roofline figure uses the measured bandwidth ceiling only; a compute ceiling was not established.
- Four matrices were excluded from ELL timing because their padding ratio exceeded the configured limit.
- Power measurements were not collected.

Raw benchmark CSV includes warm-up rows; use the `warmup` field when processing measurements.
