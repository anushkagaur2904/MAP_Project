#include "benchmark.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <sys/stat.h>

static bool file_exists_and_nonempty(const std::string &path) {
    std::ifstream f(path);
    if (!f.is_open()) return false;
    f.seekg(0, std::ios::end);
    return f.tellg() > 0;
}

static void ensure_parent_dir(const std::string &path) {
    size_t pos = path.find_last_of("/\\");
    if (pos != std::string::npos) {
        std::string dir = path.substr(0, pos);
#ifdef _WIN32
        _mkdir(dir.c_str());
#else
        mkdir(dir.c_str(), 0777);
#endif
    }
}

BenchmarkConfig parse_args(int argc, char **argv) {
    BenchmarkConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--matrix" || arg == "-m") && i + 1 < argc) cfg.matrix_path = argv[++i];
        else if ((arg == "--vector" || arg == "-v") && i + 1 < argc) cfg.vector_path = argv[++i];
        else if ((arg == "--reference" || arg == "-r") && i + 1 < argc) cfg.reference_path = argv[++i];
        else if ((arg == "--dataset" || arg == "-d") && i + 1 < argc) cfg.dataset_name = argv[++i];
        else if ((arg == "--implementation" || arg == "--impl" || arg == "-i") && i + 1 < argc) cfg.implementation = argv[++i];
        else if (arg == "--platform" && i + 1 < argc) cfg.platform = argv[++i];
        else if (arg == "--cluster" && i + 1 < argc) cfg.cluster = argv[++i];
        else if (arg == "--problem" && i + 1 < argc) cfg.problem = argv[++i];
        else if (arg == "--raw" && i + 1 < argc) cfg.raw_csv_path = argv[++i];
        else if (arg == "--summary" && i + 1 < argc) cfg.summary_csv_path = argv[++i];
        else if (arg == "--output" && i + 1 < argc) cfg.output_vector_path = argv[++i];
        else if (arg == "--block-size" && i + 1 < argc) cfg.block_size = std::stoi(argv[++i]);
        else if (arg == "--warps" && i + 1 < argc) cfg.warps_per_block = std::stoi(argv[++i]);
        else if (arg == "--ell-width" && i + 1 < argc) cfg.ell_width = std::stoi(argv[++i]);
        else if (arg == "--runs" && i + 1 < argc) cfg.measured_runs = std::stoi(argv[++i]);
        else if (arg == "--warmup" && i + 1 < argc) cfg.warmup_runs = std::stoi(argv[++i]);
        else if (arg == "--quiet") cfg.verbose = false;
    }
    return cfg;
}

void write_csv_header_if_needed(const std::string &csv_path) {
    if (file_exists_and_nonempty(csv_path)) return;
    ensure_parent_dir(csv_path);
    std::ofstream f(csv_path, std::ios::app);
    if (f.is_open()) {
        f << "cluster,problem,platform,implementation,dataset,size,threads_or_ranks,run_id,total_ms,kernel_ms,transfer_ms,throughput,error,power_w\n";
    }
}

void append_raw_record(const std::string &csv_path, const BenchmarkRecord &rec) {
    write_csv_header_if_needed(csv_path);
    ensure_parent_dir(csv_path);
    std::ofstream f(csv_path, std::ios::app);
    if (f.is_open()) {
        f << rec.cluster << ","
          << rec.problem << ","
          << rec.platform << ","
          << rec.implementation << ","
          << rec.dataset << ","
          << rec.size << ","
          << rec.threads_or_ranks << ","
          << rec.run_id << ","
          << std::fixed << std::setprecision(6) << rec.total_ms << ","
          << rec.kernel_ms << ","
          << rec.transfer_ms << ","
          << rec.throughput << ","
          << std::scientific << std::setprecision(8) << rec.error << ","
          << rec.power_w << "\n";
    }
}

BenchmarkSummary compute_summary(const std::vector<BenchmarkRecord> &records, long long nnz, size_t bytes_transferred) {
    BenchmarkSummary sum;
    if (records.empty()) return sum;

    sum.implementation = records[0].implementation;
    sum.dataset = records[0].dataset;
    sum.size = records[0].size;
    sum.config = records[0].threads_or_ranks;
    sum.nnz = nnz;
    sum.num_runs = (int)records.size();

    double sum_kernel = 0.0, sum_total = 0.0, sum_transfer = 0.0, sum_gflops = 0.0;
    sum.min_kernel_ms = records[0].kernel_ms;
    sum.max_kernel_ms = records[0].kernel_ms;
    sum.max_error = 0.0;

    for (const auto &r : records) {
        sum_kernel += r.kernel_ms;
        sum_total += r.total_ms;
        sum_transfer += r.transfer_ms;
        sum_gflops += r.throughput;
        if (r.kernel_ms < sum.min_kernel_ms) sum.min_kernel_ms = r.kernel_ms;
        if (r.kernel_ms > sum.max_kernel_ms) sum.max_kernel_ms = r.kernel_ms;
        if (r.error > sum.max_error) sum.max_error = r.error;
    }

    sum.mean_kernel_ms = sum_kernel / sum.num_runs;
    sum.mean_total_ms = sum_total / sum.num_runs;
    sum.mean_transfer_ms = sum_transfer / sum.num_runs;
    sum.mean_gflops = sum_gflops / sum.num_runs;

    double var = 0.0;
    for (const auto &r : records) {
        double d = r.kernel_ms - sum.mean_kernel_ms;
        var += d * d;
    }
    sum.std_kernel_ms = std::sqrt(var / sum.num_runs);

    // Effective bandwidth in GB/s = bytes / (mean_kernel_sec * 1e9)
    if (sum.mean_kernel_ms > 0.0 && bytes_transferred > 0) {
        sum.bandwidth_gbs = (double)bytes_transferred / (sum.mean_kernel_ms * 1e6);
    }

    return sum;
}

void append_summary_record(const std::string &csv_path, const BenchmarkSummary &summary) {
    bool write_hdr = !file_exists_and_nonempty(csv_path);
    ensure_parent_dir(csv_path);
    std::ofstream f(csv_path, std::ios::app);
    if (!f.is_open()) return;

    if (write_hdr) {
        f << "implementation,dataset,size,config,nnz,runs,mean_kernel_ms,std_kernel_ms,min_kernel_ms,max_kernel_ms,mean_total_ms,mean_transfer_ms,mean_gflops,bandwidth_gbs,max_error,speedup_vs_cpu,cusparse_percentage\n";
    }

    f << summary.implementation << ","
      << summary.dataset << ","
      << summary.size << ","
      << summary.config << ","
      << summary.nnz << ","
      << summary.num_runs << ","
      << std::fixed << std::setprecision(6)
      << summary.mean_kernel_ms << ","
      << summary.std_kernel_ms << ","
      << summary.min_kernel_ms << ","
      << summary.max_kernel_ms << ","
      << summary.mean_total_ms << ","
      << summary.mean_transfer_ms << ","
      << summary.mean_gflops << ","
      << summary.bandwidth_gbs << ","
      << std::scientific << std::setprecision(8) << summary.max_error << ","
      << std::fixed << std::setprecision(4) << summary.speedup_vs_cpu << ","
      << std::fixed << std::setprecision(2) << summary.cusparse_percentage << "\n";
}
