# Cluster B – Team 5 CPU handoff

- Team: 5, Problem 5, CPU/OpenMP SpMV
- Compiler and version:
- Compiler flags:
- CPU model:
- Physical cores/logical processors:
- Cache sizes:
- RAM:
- Windows version:
- OpenMP runtime/version:
- STREAM array size/thread count:
- STREAM measured bandwidth:
- Matrix collection date:
- Vector generation method/seed:
- Correctness tolerance:
- Timing method:
- Warm-up policy:
- Repetition count:
- Thread counts tested:
- OMP_PROC_BIND / OMP_PLACES:
- NUMA topology and limitations:
- Files included:
- Known limitations:

All teams should use the exact same matrix and vector files. Verify SHA-256 checksums before benchmarking.


## Correctness and reproducibility
- Reference implementation: CSR sequential in `src/spmv.c`
- Numerical check: max absolute error; current default tolerance is `1e-10 * (1 + max_abs(reference))`
- Saved vector generation: deterministic NumPy generator seeded from SHA-256 of matrix ID; vector file is shared
- SHA-256 manifest: `SHA256SUMS.csv`
- Matrix input format: Matrix Market coordinate; real/integer/pattern; general/symmetric/hermitian supported
- Index storage: 32-bit C `int`; matrices requiring indices/NNZ beyond 32-bit limits need a 64-bit implementation before use
