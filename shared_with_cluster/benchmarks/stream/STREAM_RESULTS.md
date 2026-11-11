# STREAM Bandwidth Results

Compiler: GCC 16.2.0 (MSYS2 UCRT64)
Compilation flags: -O3 -fopenmp
Array size: 50,000,000 elements per array
Number of timing iterations: 10
Total array footprint: approximately 1.144 GiB
Validation: STREAM validation passed

Measured bandwidth (MB/s):

| Threads | Copy | Scale | Add | Triad |
|---:|---:|---:|---:|---:|
| 1 | 16220.6 | 10526.5 | 12148.6 | 12054.4 |
| 2 | 15899.5 | 11360.3 | 12840.7 | 12672.3 |
| 4 | 14745.2 | 11940.1 | 13230.9 | 12317.8 |
| 6 | 15528.6 | 11987.0 | 12966.1 | 12790.7 |
| 8 | 14943.8 | 11860.5 | 12593.3 | 12685.3 |
| 10 | 15631.4 | 12357.3 | 12789.6 | 12779.7 |
| 12 | 15536.4 | 11817.4 | 12542.7 | 12706.7 |

Highest measured Triad bandwidth: 12,790.7 MB/s (12.7907 GB/s) at 6 threads.

This is measured STREAM bandwidth, not theoretical peak memory bandwidth.
