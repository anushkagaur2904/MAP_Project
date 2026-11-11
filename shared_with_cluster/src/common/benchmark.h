#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <string>
#include <vector>
#include <chrono>

struct BenchmarkConfig {
    std::string matrix_path;
    std::string vector_path;
    std::string reference_path;
    std::string dataset_name;
    std::string implementation;
    std::string platform = "NVIDIA_GPU";
    std::string cluster = "Team7";
    std::string problem = "P7";
    std::string raw_csv_path = "results/raw/team7_raw.csv";
    std::string summary_csv_path = "results/summary/team7_summary.csv";
    std::string output_vector_path;
    
    int block_size = 256;         // threads per block
    int warps_per_block = 8;      // for vector CSR
    int ell_width = -1;           // -1 for auto
    int warmup_runs = 1;
    int measured_runs = 10;
    
    bool verbose = true;
};

struct BenchmarkRecord {
    std::string cluster = "Team7";
    std::string problem = "P7";
    std::string platform = "NVIDIA_GPU";
    std::string implementation;
    std::string dataset;
    std::string size;             // e.g. "14000x14000"
    std::string threads_or_ranks; // e.g. "block256"
    int run_id = 0;
    double total_ms = 0.0;
    double kernel_ms = 0.0;
    double transfer_ms = 0.0;
    double throughput = 0.0;     // GFLOP/s
    double error = 0.0;          // max abs error
    std::string power_w;         // power in watts if available, else empty
};

struct BenchmarkSummary {
    std::string implementation;
    std::string dataset;
    std::string size;
    std::string config;
    long long nnz = 0;
    
    int num_runs = 0;
    double mean_kernel_ms = 0.0;
    double std_kernel_ms = 0.0;
    double min_kernel_ms = 0.0;
    double max_kernel_ms = 0.0;
    
    double mean_total_ms = 0.0;
    double mean_transfer_ms = 0.0;
    double mean_gflops = 0.0;
    double bandwidth_gbs = 0.0;
    double max_error = 0.0;
    
    double speedup_vs_cpu = 1.0;
    double cusparse_percentage = 0.0;
};

// Utilities
BenchmarkConfig parse_args(int argc, char **argv);
void write_csv_header_if_needed(const std::string &csv_path);
void append_raw_record(const std::string &csv_path, const BenchmarkRecord &rec);
BenchmarkSummary compute_summary(const std::vector<BenchmarkRecord> &records, long long nnz, size_t bytes_transferred);
void append_summary_record(const std::string &csv_path, const BenchmarkSummary &summary);

// CPU High resolution timer
class CPUTimer {
public:
    void start() {
        t_start = std::chrono::high_resolution_clock::now();
    }
    void stop() {
        t_end = std::chrono::high_resolution_clock::now();
    }
    double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(t_end - t_start).count();
    }
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> t_start, t_end;
};

#endif // BENCHMARK_H
