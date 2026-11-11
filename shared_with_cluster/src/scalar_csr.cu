#include <iostream>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include "common/cuda_common.h"
#include "common/matrix_loader.h"
#include "common/correctness.h"
#include "common/benchmark.h"

// Scalar CSR Kernel: 1 thread per row
__global__ void spmv_scalar_csr_kernel(int num_rows,
                                       const int * __restrict__ row_ptr,
                                       const int * __restrict__ col_idx,
                                       const double * __restrict__ values,
                                       const double * __restrict__ x,
                                       double * __restrict__ y) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < num_rows) {
        double sum = 0.0;
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];
        #pragma unroll 4
        for (int j = row_start; j < row_end; ++j) {
            sum += values[j] * x[col_idx[j]];
        }
        y[row] = sum;
    }
}

int main(int argc, char **argv) {
    BenchmarkConfig cfg = parse_args(argc, argv);
    cfg.platform = "NVIDIA_GPU";
    cfg.implementation = "scalar_csr";

    if (cfg.matrix_path.empty() || cfg.vector_path.empty()) {
        std::cerr << "Usage: " << argv[0] << " --matrix <file.mtx> --vector <vec.txt> [--reference <ref.txt>] [--block-size <N>] ...\n";
        return 1;
    }

    if (cfg.dataset_name.empty()) {
        size_t slash = cfg.matrix_path.find_last_of("/\\");
        size_t dot = cfg.matrix_path.find_last_of('.');
        if (slash == std::string::npos) slash = 0; else slash += 1;
        cfg.dataset_name = cfg.matrix_path.substr(slash, dot - slash);
    }

    try {
        if (cfg.verbose) std::cout << "Loading matrix: " << cfg.matrix_path << "..." << std::endl;
        COO coo = load_matrix_market(cfg.matrix_path);
        CSR csr = coo_to_csr(coo, true);
        std::vector<double> h_x = load_vector(cfg.vector_path, csr.cols);
        std::vector<double> h_y(csr.rows, 0.0);

        int *d_row_ptr = nullptr;
        int *d_col_idx = nullptr;
        double *d_values = nullptr;
        double *d_x = nullptr;
        double *d_y = nullptr;

        CUDA_CHECK(cudaMalloc(&d_row_ptr, (csr.rows + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_col_idx, csr.nnz * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_values, csr.nnz * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_x, csr.cols * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_y, csr.rows * sizeof(double)));

        GpuTimer timer_transfer;
        GpuTimer timer_kernel;

        // Measure H2D transfer time
        timer_transfer.start();
        CUDA_CHECK(cudaMemcpy(d_row_ptr, csr.row_ptr.data(), (csr.rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_col_idx, csr.col_idx.data(), csr.nnz * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_values, csr.values.data(), csr.nnz * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), csr.cols * sizeof(double), cudaMemcpyHostToDevice));
        timer_transfer.stop();
        float h2d_ms = timer_transfer.elapsed_ms();

        int block_size = cfg.block_size > 0 ? cfg.block_size : 256;
        int grid_size = (csr.rows + block_size - 1) / block_size;

        // Warmup runs
        for (int w = 0; w < cfg.warmup_runs; ++w) {
            spmv_scalar_csr_kernel<<<grid_size, block_size>>>(csr.rows, d_row_ptr, d_col_idx, d_values, d_x, d_y);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Validate correctness
        CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, csr.rows * sizeof(double), cudaMemcpyDeviceToHost));
        double max_error = 0.0;
        if (!cfg.reference_path.empty()) {
            std::vector<double> h_ref = load_vector(cfg.reference_path, csr.rows);
            ValidationResult v = validate_result(h_y, h_ref);
            max_error = v.max_abs_error;
            if (!v.passed) {
                std::cerr << "ERROR: Correctness validation failed for " << cfg.dataset_name << ": " << v.message << std::endl;
                return 3;
            } else if (cfg.verbose) {
                std::cout << "Validation PASSED (max_abs_error = " << max_error << ")" << std::endl;
            }
        }

        // Measured runs
        std::vector<BenchmarkRecord> records;
        std::string size_str = std::to_string(csr.rows) + "x" + std::to_string(csr.cols);
        std::string threads_str = "block" + std::to_string(block_size);

        for (int r = 0; r < cfg.measured_runs; ++r) {
            timer_kernel.start();
            spmv_scalar_csr_kernel<<<grid_size, block_size>>>(csr.rows, d_row_ptr, d_col_idx, d_values, d_x, d_y);
            timer_kernel.stop();
            CUDA_CHECK(cudaGetLastError());
            float k_ms = timer_kernel.elapsed_ms();

            timer_transfer.start();
            CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, csr.rows * sizeof(double), cudaMemcpyDeviceToHost));
            timer_transfer.stop();
            float d2h_ms = timer_transfer.elapsed_ms();

            double total_transfer_ms = (double)(h2d_ms + d2h_ms);
            double total_ms = (double)k_ms + total_transfer_ms;
            double gflops = (2.0 * (double)csr.nnz) / ((double)k_ms * 1e6);

            BenchmarkRecord rec;
            rec.cluster = cfg.cluster;
            rec.problem = cfg.problem;
            rec.platform = cfg.platform;
            rec.implementation = cfg.implementation;
            rec.dataset = cfg.dataset_name;
            rec.size = size_str;
            rec.threads_or_ranks = threads_str;
            rec.run_id = r;
            rec.total_ms = total_ms;
            rec.kernel_ms = (double)k_ms;
            rec.transfer_ms = total_transfer_ms;
            rec.throughput = gflops;
            rec.error = max_error;
            rec.power_w = "";

            records.push_back(rec);
            append_raw_record(cfg.raw_csv_path, rec);
        }

        size_t bytes_transferred = (size_t)(csr.rows + 1) * sizeof(int) + 
                                  (size_t)csr.nnz * (sizeof(int) + sizeof(double)) + 
                                  (size_t)(csr.cols + csr.rows) * sizeof(double);

        BenchmarkSummary summary = compute_summary(records, csr.nnz, bytes_transferred);
        append_summary_record(cfg.summary_csv_path, summary);

        if (cfg.verbose) {
            std::cout << "Scalar CSR Summary for " << cfg.dataset_name << " (" << threads_str << "):\n"
                      << "  Mean Kernel: " << summary.mean_kernel_ms << " ms (+/- " << summary.std_kernel_ms << ")\n"
                      << "  Throughput:  " << summary.mean_gflops << " GFLOP/s\n"
                      << "  Bandwidth:   " << summary.bandwidth_gbs << " GB/s\n";
        }

        if (!cfg.output_vector_path.empty()) {
            save_vector(cfg.output_vector_path, h_y);
        }

        CUDA_CHECK(cudaFree(d_row_ptr));
        CUDA_CHECK(cudaFree(d_col_idx));
        CUDA_CHECK(cudaFree(d_values));
        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFree(d_y));

    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 2;
    }

    return 0;
}
