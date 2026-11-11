#include <iostream>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include "common/cuda_common.h"
#include "common/matrix_loader.h"
#include "common/correctness.h"
#include "common/benchmark.h"

// ELLPACK Column-Major Kernel:
// Index = column_position * num_rows + row
// Adjacent threads access adjacent rows at column_position -> perfectly coalesced!
__global__ void spmv_ell_kernel(int num_rows,
                                int ell_width,
                                const int * __restrict__ ell_col_idx,
                                const double * __restrict__ ell_values,
                                const double * __restrict__ x,
                                double * __restrict__ y) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < num_rows) {
        double sum = 0.0;
        #pragma unroll 4
        for (int p = 0; p < ell_width; ++p) {
            int idx = p * num_rows + row;
            sum += ell_values[idx] * x[ell_col_idx[idx]];
        }
        y[row] = sum;
    }
}

int main(int argc, char **argv) {
    BenchmarkConfig cfg = parse_args(argc, argv);
    cfg.platform = "NVIDIA_GPU";
    cfg.implementation = "ell";

    if (cfg.matrix_path.empty() || cfg.vector_path.empty()) {
        std::cerr << "Usage: " << argv[0] << " --matrix <file.mtx> --vector <vec.txt> [--reference <ref.txt>] [--ell-width <W>] ...\n";
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
        
        // Convert to ELLPACK
        ELL ell = csr_to_ell(csr, cfg.ell_width);

        if (cfg.verbose) {
            std::cout << "ELLPACK format: rows=" << ell.rows << ", cols=" << ell.cols 
                      << ", ell_width=" << ell.width 
                      << ", padding_ratio=" << ell.padding_ratio << std::endl;
        }

        std::vector<double> h_x = load_vector(cfg.vector_path, ell.cols);
        std::vector<double> h_y(ell.rows, 0.0);

        size_t ell_total = (size_t)ell.rows * (size_t)ell.width;

        int *d_ell_col = nullptr;
        double *d_ell_val = nullptr;
        double *d_x = nullptr;
        double *d_y = nullptr;

        CUDA_CHECK(cudaMalloc(&d_ell_col, ell_total * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_ell_val, ell_total * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_x, ell.cols * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_y, ell.rows * sizeof(double)));

        GpuTimer timer_transfer;
        GpuTimer timer_kernel;

        // Measure H2D transfer time
        timer_transfer.start();
        CUDA_CHECK(cudaMemcpy(d_ell_col, ell.col_idx.data(), ell_total * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ell_val, ell.values.data(), ell_total * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), ell.cols * sizeof(double), cudaMemcpyHostToDevice));
        timer_transfer.stop();
        float h2d_ms = timer_transfer.elapsed_ms();

        int block_size = cfg.block_size > 0 ? cfg.block_size : 256;
        int grid_size = (ell.rows + block_size - 1) / block_size;

        // Warmup runs
        for (int w = 0; w < cfg.warmup_runs; ++w) {
            spmv_ell_kernel<<<grid_size, block_size>>>(ell.rows, ell.width, d_ell_col, d_ell_val, d_x, d_y);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Validate correctness
        CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, ell.rows * sizeof(double), cudaMemcpyDeviceToHost));
        double max_error = 0.0;
        if (!cfg.reference_path.empty()) {
            std::vector<double> h_ref = load_vector(cfg.reference_path, ell.rows);
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
        std::string size_str = std::to_string(ell.rows) + "x" + std::to_string(ell.cols);
        std::string threads_str = "ell_width" + std::to_string(ell.width) + "_block" + std::to_string(block_size);

        for (int r = 0; r < cfg.measured_runs; ++r) {
            timer_kernel.start();
            spmv_ell_kernel<<<grid_size, block_size>>>(ell.rows, ell.width, d_ell_col, d_ell_val, d_x, d_y);
            timer_kernel.stop();
            CUDA_CHECK(cudaGetLastError());
            float k_ms = timer_kernel.elapsed_ms();

            timer_transfer.start();
            CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, ell.rows * sizeof(double), cudaMemcpyDeviceToHost));
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

        size_t bytes_transferred = ell_total * (sizeof(int) + sizeof(double)) + 
                                  (size_t)(ell.cols + ell.rows) * sizeof(double);

        BenchmarkSummary summary = compute_summary(records, csr.nnz, bytes_transferred);
        append_summary_record(cfg.summary_csv_path, summary);

        if (cfg.verbose) {
            std::cout << "ELLPACK Summary for " << cfg.dataset_name << " (" << threads_str << "):\n"
                      << "  Mean Kernel: " << summary.mean_kernel_ms << " ms (+/- " << summary.std_kernel_ms << ")\n"
                      << "  Throughput:  " << summary.mean_gflops << " GFLOP/s\n"
                      << "  Bandwidth:   " << summary.bandwidth_gbs << " GB/s\n"
                      << "  Padding Ratio: " << ell.padding_ratio << "\n";
        }

        if (!cfg.output_vector_path.empty()) {
            save_vector(cfg.output_vector_path, h_y);
        }

        CUDA_CHECK(cudaFree(d_ell_col));
        CUDA_CHECK(cudaFree(d_ell_val));
        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFree(d_y));

    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 2;
    }

    return 0;
}
