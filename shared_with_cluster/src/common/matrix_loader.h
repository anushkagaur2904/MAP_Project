#ifndef MATRIX_LOADER_H
#define MATRIX_LOADER_H

#include <vector>
#include <string>
#include <cstddef>

struct COO {
    int rows = 0;
    int cols = 0;
    long long nnz = 0;
    std::vector<int> row_idx;
    std::vector<int> col_idx;
    std::vector<double> values;
};

struct CSR {
    int rows = 0;
    int cols = 0;
    long long nnz = 0;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<double> values;
};

// Column-major ELLPACK storage:
// index = column_position * num_rows + row
struct ELL {
    int rows = 0;
    int cols = 0;
    int width = 0;
    long long nnz = 0;
    std::vector<int> col_idx;      // size = rows * width
    std::vector<double> values;    // size = rows * width
    double padding_ratio = 1.0;
};

// Hybrid ELL + COO storage:
// First ell_width entries of each row stored in column-major ELL,
// remaining overflow nonzeros stored in COO.
struct HYB {
    int rows = 0;
    int cols = 0;
    int ell_width = 0;
    long long total_nnz = 0;
    long long ell_nnz = 0;
    long long coo_nnz = 0;
    ELL ell;
    COO coo;
};

struct MatrixStats {
    int rows = 0;
    int cols = 0;
    long long nnz = 0;
    double mean_nnz_per_row = 0.0;
    double std_nnz_per_row = 0.0;
    int max_nnz_per_row = 0;
    int min_nnz_per_row = 0;
    double irregularity = 1.0;     // max / mean
};

// I/O and conversion functions
COO load_matrix_market(const std::string &path);
CSR coo_to_csr(const COO &coo, bool sort_indices = true);
ELL csr_to_ell(const CSR &csr, int max_width = -1);
HYB csr_to_hyb(const CSR &csr, int ell_width = -1);
MatrixStats compute_stats(const CSR &csr);

std::vector<double> load_vector(const std::string &path, int size);
void save_vector(const std::string &path, const double *vec, int size);
void save_vector(const std::string &path, const std::vector<double> &vec);

#endif // MATRIX_LOADER_H
