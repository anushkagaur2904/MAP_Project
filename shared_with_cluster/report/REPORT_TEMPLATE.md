# Parallel Sparse Matrix–Vector Multiplication on a Multicore CPU

## 1. Objective
State the goal: compare COO, CSR and ELL SpMV on a multicore CPU using sequential and OpenMP implementations, and evaluate scheduling, matrix irregularity, NUMA first-touch and thread binding.

## 2. Experimental setup
- CPU model / cores / logical processors:
- Cache sizes:
- RAM:
- OS:
- Compiler and flags:
- OpenMP runtime:
- STREAM array size, thread count and measured bandwidth:
- SuiteSparse matrix collection date and selected matrix IDs:
- Number of runs and warm-up policy:
- Thread counts and chunk sizes:
- OMP_PROC_BIND / OMP_PLACES settings:

## 3. Methodology
Describe Matrix Market input handling, double precision, fixed saved vector per matrix, COO/CSR/ELL layouts, reference verification, OpenMP schedules, timing method and metric formulas. Explain COO atomic updates and ELL padding.

## 4. Results
Insert the five required graphs. Report means and standard deviations, not only best runs.

- GFLOP/s per format per matrix:
- Speedup vs threads and ideal line:
- Static/dynamic/guided comparison:
- Load imbalance vs matrix irregularity:
- Roofline using measured STREAM bandwidth and documented peak compute estimate:

## 5. Discussion
Explain how matrix row-length distribution affects scheduling and ELL padding. Discuss when COO atomics, CSR row scheduling and ELL regular storage help or hurt. Discuss thread binding and first-touch results only to the extent supported by the hardware topology and measurements.

## 6. Conclusion and limitations
State measured findings only. Document matrices/configurations excluded for memory/time reasons, Windows profiling limitations, NUMA topology, and any missing power measurements.
