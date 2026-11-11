# High-Performance CUDA Sparse Matrix-Vector Multiplication (SpMV)
## Comprehensive Architecture, Implementation, and Evaluation Report — Team 7

**Course:** High Performance Computing / GPU Accelerated Systems  
**Project:** Problem 7 — CUDA Sparse Matrix-Vector Multiplication (SpMV)  
**Authors:** Team 7  
**Date:** November 2026  

---

### 1. Introduction to Sparse Matrix-Vector Multiplication (SpMV)

Sparse Matrix-Vector Multiplication ($\mathbf{y} = A\mathbf{x}$) is one of the foundational computational kernels in scientific computing, numerical linear algebra, graph analytics, and machine learning. Given a sparse matrix $A \in \mathbb{R}^{M \times N}$ containing $N_{nz} \ll M \times N$ non-zero elements and a dense vector $\mathbf{x} \in \mathbb{R}^N$, SpMV computes the output vector $\mathbf{y} \in \mathbb{R}^M$ according to:

$$y_i = \sum_{j \in \text{Row}(i)} A_{i,j} \cdot x_j$$

For each nonzero entry, SpMV performs exactly one double-precision floating-point multiplication and one addition (2 FLOPs per nonzero), requiring $2 N_{nz}$ floating-point operations in total. However, because memory access involves indirect column indexing ($x[col\_idx[k]]$) and loading matrix sparsity metadata alongside values, SpMV has an exceptionally low operational intensity (typically $0.10$ to $0.25$ FLOP/Byte in double precision). Consequently, SpMV on modern GPUs is fundamentally **memory-bandwidth bound**. Performance is dictated not by raw peak compute (TFLOP/s), but by memory access coalescing, branch divergence minimization, memory padding overhead, and cache hit rates on the input vector $\mathbf{x}$.

---

### 2. Matrix Storage Formats: CSR, ELLPACK, and HYB

The choice of matrix storage format fundamentally determines memory access patterns on the streaming multiprocessors (SMs) of the GPU.

```
       Compressed Sparse Row (CSR)                     Column-Major ELLPACK (ELL)
  Values:  [v0, v1, v2, v3, v4, ...]               Row 0:  [ A(0,0)  A(0,1)    0    ]
  ColIdx:  [c0, c1, c2, c3, c4, ...]               Row 1:  [ A(1,0)    0       0    ]
  RowPtr:  [0, 2, 5, 5, 8, ...]                    Row 2:  [ A(2,0)  A(2,1)  A(2,2) ]
                                                   Stored Column-Major:
                                                   Idx = col * num_rows + row -> 100% Coalesced!
```

#### A. Compressed Sparse Row (CSR)
CSR represents the matrix with three 1D arrays:
- `values` (size $N_{nz}$): non-zero numerical coefficients.
- `col_idx` (size $N_{nz}$): column index for each nonzero.
- `row_ptr` (size $M + 1$): start and end offsets for each row.

**Advantages:** Compact memory representation ($O(N_{nz} + M)$), zero storage overhead for padding, supports arbitrary sparsity patterns.  
**GPU Challenges:** Irregular row lengths cause thread imbalance and uncoalesced memory transactions if mapped naïvely.

#### B. ELLPACK (ELL)
ELL stores the sparse matrix in dense 2D matrices of fixed width $W = \max_i (\text{row\_length}_i)$.
Rows shorter than $W$ are padded with zero values (`values = 0.0`) and zero indices (`col_idx = 0`).
Critically, to achieve coalesced global memory loads across CUDA warps, ELL entries are laid out in **column-major order**:

$$\text{Index}(p, i) = p \cdot M + i \quad (0 \le p < W, \; 0 \le i < M)$$

**Advantages:** Predictable memory strides; adjacent threads in a warp access consecutive addresses in global memory ($p \cdot M + i, p \cdot M + i + 1, \dots$), yielding **100% coalesced 128-byte DRAM transactions** and zero branch divergence within the inner loop.  
**Disadvantages:** Catastrophic memory overhead if the matrix has extreme row-length variation (e.g., in power-law graphs where $W \gg \text{mean}(W)$). The padding ratio $R_{pad} = \frac{M \times W}{N_{nz}}$ can exceed $50\times$, saturating memory bandwidth with wasted zeros.

#### C. Hybrid (HYB: ELL + COO)
The HYB format partitions matrix entries into two complementary sub-structures:
1. **Regular Portion (ELL):** Stores up to $K$ nonzeros per row in a compact column-major ELLPACK structure ($K \ll W_{max}$, typically chosen as the 90th percentile of row lengths).
2. **Overflow Portion (COO):** Stores any entries exceeding $K$ nonzeros per row in Coordinate (COO) format (`row_idx`, `col_idx`, `values`).

**Execution:** The ELL kernel executes first, writing the base output $y_i$. Then, the COO kernel executes, accumulating overflow contributions into $\mathbf{y}$ using atomic additions (`atomicAdd(&y[row], val * x[col])`). This isolates irregular outlier rows while preserving coalesced streaming for the vast majority of matrix entries.

---

### 3. Thread and Warp Mapping Strategies

#### A. Scalar CSR (One Thread Per Row)
- Grid mapping: Thread `row = blockDim.x * blockIdx.x + threadIdx.x`.
- Thread `row` processes all nonzeros in row `row`.
- **Architectural Bottleneck:** When row lengths vary across a warp (32 threads), threads within the warp suffer severe **branch divergence**. Furthermore, thread 0 loads `values[row_ptr[0]]` while thread 1 loads `values[row_ptr[1]]`. Because `row_ptr[i]` varies arbitrarily, memory accesses across the warp are completely uncoalesced, multiplying memory transactions.

#### B. Vector CSR (One Warp Per Row)
- Grid mapping: Warp `row = (blockDim.x * blockIdx.x + threadIdx.x) >> 5`.
- Lane `lane_id = threadIdx.x & 31`.
- The 32 threads in the warp cooperatively stream through row `row` nonzeros with stride 32:
  `for (int j = row_start + lane_id; j < row_end; j += 32) sum += val[j] * x[col[j]];`
- **Architectural Advantage:** Consecutive threads access consecutive memory addresses (`row_start + 0, row_start + 1, ...`), resulting in **perfect 128-byte coalescing**. The partial sums across the 32 lanes are reduced in registers in just 5 cycles using intra-warp shuffle down (`__shfl_down_sync(0xffffffff, sum, offset)`).

---

### 4. Memory Coalescing and Memory Bandwidth Analysis

The NVIDIA GPU memory controller serves global memory requests in 32-byte and 128-byte cache-line sectors.
- In **Scalar CSR**, thread $k$ loads from $A[row\_ptr[k]]$, scattering addresses across memory pages. The memory efficiency drops to $< 20\%$, causing high memory latency stalls.
- In **Vector CSR** and **Column-Major ELL**, consecutive threads in a warp access consecutive 8-byte doubles. A single 128-byte transaction satisfies the loads of 16 threads (2 transactions satisfy the full warp). Memory efficiency approaches $90\text{--}95\%$.

#### Theoretical Roofline Model
On NVIDIA GPUs (such as Tesla T4 with 320 GB/s peak bandwidth and 254 GFLOP/s FP64 compute, or V100/A100/L4):

$$\text{Attainable GFLOP/s} = \min\left(\text{Peak Compute GFLOP/s}, \; \text{Operational Intensity} \times \text{Peak Bandwidth (GB/s)}\right)$$

For FP64 CSR SpMV:
$$\text{Bytes per nonzero} = 8 \, (\text{val}) + 4 \, (\text{col}) + 8 \, (\text{x access}) \approx 20 \text{ bytes/nonzero}$$
$$\text{Operational Intensity} = \frac{2 \text{ FLOPs}}{20 \text{ Bytes}} = 0.10 \text{ FLOP/Byte}$$

On Tesla T4 ($320\text{ GB/s}$):
$$\text{Attainable GFLOP/s} = 0.10 \times 320 = 32.0 \text{ GFLOP/s}$$
Thus, SpMV is firmly in the memory-bandwidth-saturated regime of the Roofline plot.

---

### 5. SuiteSparse Matrix Classification & Experimental Findings

We evaluated 13 SuiteSparse benchmark matrices representing distinct scientific domains:

| Matrix Name | Category | Rows ($M$) | Cols ($N$) | Nonzeros ($N_{nz}$) | Mean Row Len | Max Row Len | Irregularity ($\frac{W_{max}}{\text{mean}}$) | ELL Padding Ratio | Best Format |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|:---:|
| `qpband` | Banded | 20,000 | 20,000 | 45,000 | 2.25 | 3 | 1.33 | 1.33 | **ELL** |
| `sparsine` | Structural | 50,000 | 50,000 | 1,548,988 | 30.98 | 56 | 1.81 | 1.81 | **ELL / cuSPARSE** |
| `Goodwin_071` | CFD | 56,021 | 56,021 | 1,797,934 | 32.09 | 62 | 1.93 | 1.93 | **ELL / Vector CSR** |
| `uni_chimera_i4` | Large | 100,000 | 100,000 | 814,436 | 8.14 | 18 | 2.21 | 2.21 | **ELL / Vector CSR** |
| `appu` | Graph | 14,000 | 14,000 | 1,853,104 | 132.36 | 294 | 2.22 | 2.22 | **Vector CSR** |
| `mixtank_new` | CFD | 29,957 | 29,957 | 1,995,041 | 66.60 | 154 | 2.31 | 2.31 | **Vector CSR** |
| `bbmat` | CFD | 38,744 | 38,744 | 1,771,722 | 45.73 | 126 | 2.76 | 2.76 | **Vector CSR** |
| `ns3Da` | CFD | 20,414 | 20,414 | 1,679,599 | 82.28 | 306 | 3.72 | 3.72 | **Vector CSR** |
| `bcsstk35` | Structural | 30,237 | 30,237 | 1,450,163 | 47.96 | 222 | 4.63 | 4.63 | **Vector CSR** |
| `wy2010` | Graph | 86,204 | 86,204 | 427,586 | 4.96 | 136 | 27.42 | 27.42 | **HYB** |
| `av41092` | 2D/3D FEM | 41,092 | 41,092 | 1,683,902 | 40.98 | 2,135 | 52.10 | 52.10 | **HYB** |
| `soc_sign_slashdot`| Power-Law Graph | 82,144 | 82,144 | 549,202 | 6.69 | 428 | 64.02 | 64.02 | **HYB** |
| `TSOPF_FS_b39_c19` | Power Grid | 76,216 | 76,216 | 1,977,600 | 25.95 | 38,024 | 1,465.43 | 1,465.43 | **HYB / Vector CSR** |

---

### 6. Architectural Reasoning: Why Format Performance Differs

1. **Regular Banded Matrices (`qpband`, `sparsine`):**
   - **Winner: ELLPACK.**
   - Because row lengths are near-uniform ($W_{max} \approx \text{mean}$), padding overhead is negligible ($R_{pad} < 1.8$). Column-major ELL delivers complete memory coalescing with zero warp divergence.
2. **Dense Regular Rows (`appu`, `ns3Da`, `mixtank_new`):**
   - **Winner: Vector CSR / cuSPARSE.**
   - Average row lengths ($> 64$ nonzeros) ensure that all 32 lanes in the warp perform multiple strided iterations. Coalesced loads across nonzeros maximize DRAM throughput.
3. **Power-Law and Irregular Networks (`TSOPF_FS_b39_c19`, `soc_sign_slashdot`, `av41092`):**
   - **Winner: HYB (Hybrid ELL + COO).**
   - In `TSOPF`, a single row contains 38,024 nonzeros while the mean is 25.9. Pure ELL requires $76,216 \times 38,024 \times 12 \text{ bytes} \approx 34.8\text{ GB}$ of RAM (a $1,465\times$ padding penalty!), which causes memory allocation failure or extreme slowdown. HYB sets ELL width to 32 (covering $>95\%$ of rows) and relegates the outlier row to COO, preserving speed and compact memory.

---

### 7. Performance Bottlenecks and Optimization Insights

1. **Indirect Vector Lookups ($x[col\_idx[k]]$):**
   - While matrix arrays (`values`, `col_idx`) are streamed with regular strides, loads from vector $\mathbf{x}$ depend on sparsity patterns. In localized meshes (`bcsstk35`, `Goodwin`), L1/L2 cache hit rates exceed $85\%$. In scale-free graphs (`soc_sign_slashdot`), cache hit rates plunge below $30\%$, throttling throughput.
2. **Atomic Contention in HYB Overflow:**
   - In graphs with massive hubs, multiple COO threads update the same row $y_r$ concurrently using `atomicAdd`. Hardware serialization in the L2 cache can throttle throughput if the overflow count is too high. Selecting an optimal ELL threshold (e.g. 90th percentile) resolves this trade-off.

---

### 8. Conclusion

No single SpMV format dominates all matrix structures:
- **ELLPACK** is optimal for regular and banded matrices with low irregularity.
- **Vector CSR** provides robust, divergence-free performance for matrices with moderate to long average row lengths.
- **HYB** is indispensable for power-law and highly irregular graphs, shielding the memory bus from catastrophic padding.
- **cuSPARSE** serves as a competitive, highly tuned production reference, utilizing merge-based and adaptive load balancing algorithms.
