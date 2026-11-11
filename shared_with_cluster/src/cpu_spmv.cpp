#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include "common/matrix_loader.h"
#include "common/correctness.h"
#include "common/benchmark.h"

// Sequential CPU CSR SpMV: y[i] = sum(A[i, j] * x[j])
void cpu_spmv_csr(int rows, const int *row_ptr, const int *col_idx, const double *values,
                  const double *x, double *y) {
    for (int i = 0; i < rows; ++i) {
        double sum = 0.0;
        int start = row_ptr[i];
        int end = row_ptr[i + 1];
        for (int k = start; k < end; ++k) {
            sum += values[k] * x[col_idx[k]];
        }
        y[i] = sum;
    }
}

int main(int argc, char **argv) {
    BenchmarkConfig cfg = parse_args(argc, argv);
    cfg.platform = "CPU";
    cfg.implementation = "cpu_sequential";

    if (cfg.matrix_path.empty() || cfg.vector_path.empty()) {
        std::cerr << "Usage: " << argv[0] << " --matrix <file.mtx> --vector <vec.txt> [--reference <ref.txt>] [--dataset <name>] ...\n";
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
        if (cfg.verbose) {
            std::cout << "Matrix: " << cfg.dataset_name 
                      << " (" << csr.rows << " x " << csr.cols << ", " 
                      << csr.nnz << " nnz)" << std::endl;
        }

        std::vector<double> x = load_vector(cfg.vector_path, csr.cols);
        std::vector<double> y(csr.rows, 0.0);

        // Validation against reference output if available
        double max_error = 0.0;
        if (!cfg.reference_path.empty()) {
            std::vector<double> y_ref = load_vector(cfg.reference_path, csr.rows);
            cpu_spmv_csr(csr.rows, csr.row_ptr.data(), csr.col_idx.data(), csr.values.data(), x.data(), y.data());
            ValidationResult v = validate_result(y, y_ref);
            max_error = v.max_abs_error;
            if (!v.passed) {
                std::cerr << "WARNING: Correctness validation against reference failed: " << v.message << std::endl;
            } else if (cfg.verbose) {
                std::cout << "Validation PASSED (max_abs_error = " << max_error << ")" << std::endl;
            }
        }

        // Warmup runs
        for (int w = 0; w < cfg.warmup_runs; ++w) {
            cpu_spmv_csr(csr.rows, csr.row_ptr.data(), csr.col_idx.data(), csr.values.data(), x.data(), y.data());
        }

        // Measured runs
        CPUTimer timer;
        std::vector<BenchmarkRecord> records;
        std::string size_str = std::to_string(csr.rows) + "x" + std::to_string(csr.cols);
        std::string threads_str = "1_thread";

        for (int r = 0; r < cfg.measured_runs; ++r) {
            timer.start();
            cpu_spmv_csr(csr.rows, csr.row_ptr.data(), csr.col_idx.data(), csr.values.data(), x.data(), y.data());
            timer.stop();

            double ms = timer.elapsed_ms();
            double gflops = (2.0 * (double)csr.nnz) / (ms * 1e6);

            BenchmarkRecord rec;
            rec.cluster = cfg.cluster;
            rec.problem = cfg.problem;
            rec.platform = cfg.platform;
            rec.implementation = cfg.implementation;
            rec.dataset = cfg.dataset_name;
            rec.size = size_str;
            rec.threads_or_ranks = threads_str;
            rec.run_id = r;
            rec.total_ms = ms;
            rec.kernel_ms = ms;
            rec.transfer_ms = 0.0;
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
            std::cout << "CPU Baseline Summary for " << cfg.dataset_name << ":\n"
                      << "  Mean Kernel: " << summary.mean_kernel_ms << " ms (+/- " << summary.std_kernel_ms << ")\n"
                      << "  Throughput:  " << summary.mean_gflops << " GFLOP/s\n"
                      << "  Bandwidth:   " << summary.bandwidth_gbs << " GB/s\n";
        }

        if (!cfg.output_vector_path.empty()) {
            save_vector(cfg.output_vector_path, y);
        }

    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 2;
    }

    return 0;
}
