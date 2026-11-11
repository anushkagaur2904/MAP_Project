# Team 5 submission checklist

## Common baseline
- [ ] Hardware CPU model, cores/threads, cache sizes, memory, compiler/flags, MPI/CUDA versions/availability
- [ ] STREAM bandwidth measured with array larger than LLC; command and configuration saved
- [ ] Optimised sequential baseline compiled with -O3
- [ ] At least 3 problem sizes
- [ ] Each configuration >=5 repetitions; first warm-up ignored
- [ ] Mean and standard deviation
- [ ] Error bars on every applicable graph

## Problem 5
- [ ] 10–15 SuiteSparse matrices: regular/banded, power-law/irregular, very large
- [ ] Double precision; fixed saved random vector x per matrix
- [ ] COO sequential + OpenMP
- [ ] CSR sequential + OpenMP
- [ ] ELL sequential + OpenMP
- [ ] Reference correctness verification
- [ ] Static/dynamic/guided scheduling
- [ ] NUMA first-touch initialisation experiment
- [ ] Thread pinning experiment
- [ ] Roofline using bytes moved per nonzero

## Required outputs
- [ ] Code for all versions
- [ ] Common raw CSV schema
- [ ] GFLOP/s per format per matrix
- [ ] Speedup vs threads (ideal line)
- [ ] Scheduling comparison
- [ ] Load imbalance vs matrix irregularity
- [ ] Roofline plot
- [ ] Two-page report section

## Cluster B handoff
- [ ] Same original matrix files
- [ ] Same fixed vector files
- [ ] Matrix dimensions/NNZ/row statistics/category
- [ ] SHA-256 checksums
- [ ] Reference output and tolerance
- [ ] Optimised sequential CPU baseline
- [ ] Final raw and processed CSV
- [ ] Hardware/STREAM metadata and limitations
- [ ] Handoff note for Teams 6, 7 and 8
