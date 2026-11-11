#include "matrix_loader.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <numeric>

COO load_matrix_market(const std::string &path) {
    FILE *f = std::fopen(path.c_str(), "r");
    if (!f) {
        throw std::runtime_error("Cannot open Matrix Market file: " + path);
    }

    char line[4096];
    char object[64] = {0}, format[64] = {0}, field[64] = {0}, symmetry[64] = {0};

    if (!std::fgets(line, sizeof(line), f)) {
        std::fclose(f);
        throw std::runtime_error("Empty matrix file: " + path);
    }

    if (std::sscanf(line, "%%%%MatrixMarket %63s %63s %63s %63s", object, format, field, symmetry) != 4) {
        std::fclose(f);
        throw std::runtime_error("Invalid Matrix Market header: " + path);
    }

    for (int i = 0; object[i]; ++i) object[i] = (char)std::tolower(object[i]);
    for (int i = 0; format[i]; ++i) format[i] = (char)std::tolower(format[i]);
    for (int i = 0; field[i]; ++i) field[i] = (char)std::tolower(field[i]);
    for (int i = 0; symmetry[i]; ++i) symmetry[i] = (char)std::tolower(symmetry[i]);

    if (std::strcmp(object, "matrix") != 0 || std::strcmp(format, "coordinate") != 0) {
        std::fclose(f);
        throw std::runtime_error("Only Matrix Market coordinate matrices are supported: " + path);
    }

    bool symmetric = (std::strcmp(symmetry, "symmetric") == 0 || std::strcmp(symmetry, "hermitian") == 0);
    bool pattern = (std::strcmp(field, "pattern") == 0);
    bool complex_field = (std::strcmp(field, "complex") == 0);

    if (complex_field) {
        std::fclose(f);
        throw std::runtime_error("Complex Matrix Market values are not supported: " + path);
    }

    do {
        if (!std::fgets(line, sizeof(line), f)) {
            std::fclose(f);
            throw std::runtime_error("Missing Matrix Market dimensions: " + path);
        }
    } while (line[0] == '%' || line[0] == '\n' || line[0] == '\r');

    int rows = 0, cols = 0;
    long long nz = 0;
    if (std::sscanf(line, "%d %d %lld", &rows, &cols, &nz) != 3 || rows <= 0 || cols <= 0 || nz < 0) {
        std::fclose(f);
        throw std::runtime_error("Invalid Matrix Market dimensions in: " + path);
    }

    COO coo;
    coo.rows = rows;
    coo.cols = cols;
    size_t reserve_cap = symmetric ? (size_t)(2 * nz) : (size_t)nz;
    coo.row_idx.reserve(reserve_cap);
    coo.col_idx.reserve(reserve_cap);
    coo.values.reserve(reserve_cap);

    long long k = 0;
    while (k < nz && std::fgets(line, sizeof(line), f)) {
        if (line[0] == '%' || line[0] == '\n' || line[0] == '\r') continue;

        int r = 0, c = 0;
        double v = 1.0;
        int n = pattern ? std::sscanf(line, "%d %d", &r, &c) : std::sscanf(line, "%d %d %lf", &r, &c, &v);
        if (n < (pattern ? 2 : 3)) continue;

        r--; // 1-based to 0-based
        c--;
        if (r < 0 || r >= rows || c < 0 || c >= cols) {
            std::fclose(f);
            throw std::runtime_error("Matrix coordinate out of range in: " + path);
        }

        coo.row_idx.push_back(r);
        coo.col_idx.push_back(c);
        coo.values.push_back(v);

        if (symmetric && r != c) {
            coo.row_idx.push_back(c);
            coo.col_idx.push_back(r);
            coo.values.push_back(v);
        }
        k++;
    }

    std::fclose(f);
    if (k != nz) {
        throw std::runtime_error("Matrix Market file ended prematurely: " + path);
    }
    coo.nnz = (long long)coo.values.size();
    return coo;
}

CSR coo_to_csr(const COO &coo, bool sort_indices) {
    CSR csr;
    csr.rows = coo.rows;
    csr.cols = coo.cols;
    csr.nnz = coo.nnz;
    csr.row_ptr.assign(csr.rows + 1, 0);
    csr.col_idx.resize(csr.nnz);
    csr.values.resize(csr.nnz);

    for (long long k = 0; k < coo.nnz; ++k) {
        csr.row_ptr[coo.row_idx[k] + 1]++;
    }
    for (int i = 0; i < csr.rows; ++i) {
        csr.row_ptr[i + 1] += csr.row_ptr[i];
    }

    std::vector<int> next = csr.row_ptr;
    for (long long k = 0; k < coo.nnz; ++k) {
        int r = coo.row_idx[k];
        int p = next[r]++;
        csr.col_idx[p] = coo.col_idx[k];
        csr.values[p] = coo.values[k];
    }

    if (sort_indices) {
        // Sort nonzeros within each row ascending by column index
        for (int i = 0; i < csr.rows; ++i) {
            int start = csr.row_ptr[i];
            int end = csr.row_ptr[i + 1];
            int count = end - start;
            if (count > 1) {
                std::vector<std::pair<int, double>> row_entries(count);
                for (int j = 0; j < count; ++j) {
                    row_entries[j] = {csr.col_idx[start + j], csr.values[start + j]};
                }
                std::sort(row_entries.begin(), row_entries.end(),
                          [](const std::pair<int, double> &a, const std::pair<int, double> &b) {
                              return a.first < b.first;
                          });
                for (int j = 0; j < count; ++j) {
                    csr.col_idx[start + j] = row_entries[j].first;
                    csr.values[start + j] = row_entries[j].second;
                }
            }
        }
    }

    return csr;
}

ELL csr_to_ell(const CSR &csr, int max_width) {
    ELL ell;
    ell.rows = csr.rows;
    ell.cols = csr.cols;
    ell.nnz = csr.nnz;

    int computed_max_width = 0;
    for (int i = 0; i < csr.rows; ++i) {
        int row_len = csr.row_ptr[i + 1] - csr.row_ptr[i];
        if (row_len > computed_max_width) {
            computed_max_width = row_len;
        }
    }
    if (computed_max_width == 0) computed_max_width = 1;

    ell.width = (max_width > 0) ? max_width : computed_max_width;
    size_t total_elements = (size_t)ell.rows * (size_t)ell.width;

    // Column-major storage: index = col_pos * num_rows + row
    // Unused padding entries are zeroed (col = 0, val = 0.0)
    ell.col_idx.assign(total_elements, 0);
    ell.values.assign(total_elements, 0.0);

    for (int i = 0; i < csr.rows; ++i) {
        int row_len = csr.row_ptr[i + 1] - csr.row_ptr[i];
        int n = std::min(row_len, ell.width);
        for (int k = 0; k < n; ++k) {
            size_t idx = (size_t)k * (size_t)ell.rows + (size_t)i;
            ell.col_idx[idx] = csr.col_idx[csr.row_ptr[i] + k];
            ell.values[idx] = csr.values[csr.row_ptr[i] + k];
        }
    }

    ell.padding_ratio = (double)total_elements / (double)(csr.nnz > 0 ? csr.nnz : 1);
    return ell;
}

HYB csr_to_hyb(const CSR &csr, int ell_width) {
    HYB hyb;
    hyb.rows = csr.rows;
    hyb.cols = csr.cols;
    hyb.total_nnz = csr.nnz;

    if (ell_width <= 0) {
        // Automatic determination: use mean row length rounded up, capped at max row length
        std::vector<int> row_lengths(csr.rows);
        for (int i = 0; i < csr.rows; ++i) {
            row_lengths[i] = csr.row_ptr[i + 1] - csr.row_ptr[i];
        }
        std::sort(row_lengths.begin(), row_lengths.end());
        int p90 = row_lengths.empty() ? 1 : row_lengths[(size_t)(row_lengths.size() * 0.90)];
        // Use 90th percentile to balance ELL compactness and minimal COO overflow
        ell_width = std::max(1, p90);
        if (ell_width > row_lengths.back()) ell_width = row_lengths.back();
    }

    hyb.ell_width = ell_width;
    hyb.ell = csr_to_ell(csr, ell_width);

    // Count and extract overflow entries
    long long overflow_count = 0;
    for (int i = 0; i < csr.rows; ++i) {
        int row_len = csr.row_ptr[i + 1] - csr.row_ptr[i];
        if (row_len > ell_width) {
            overflow_count += (row_len - ell_width);
        }
    }

    hyb.coo_nnz = overflow_count;
    hyb.ell_nnz = csr.nnz - overflow_count;

    hyb.coo.rows = csr.rows;
    hyb.coo.cols = csr.cols;
    hyb.coo.nnz = overflow_count;
    hyb.coo.row_idx.reserve(overflow_count);
    hyb.coo.col_idx.reserve(overflow_count);
    hyb.coo.values.reserve(overflow_count);

    for (int i = 0; i < csr.rows; ++i) {
        int row_len = csr.row_ptr[i + 1] - csr.row_ptr[i];
        if (row_len > ell_width) {
            for (int k = ell_width; k < row_len; ++k) {
                int pos = csr.row_ptr[i] + k;
                hyb.coo.row_idx.push_back(i);
                hyb.coo.col_idx.push_back(csr.col_idx[pos]);
                hyb.coo.values.push_back(csr.values[pos]);
            }
        }
    }

    return hyb;
}

MatrixStats compute_stats(const CSR &csr) {
    MatrixStats stats;
    stats.rows = csr.rows;
    stats.cols = csr.cols;
    stats.nnz = csr.nnz;

    if (csr.rows == 0) return stats;

    double sum = 0.0;
    stats.max_nnz_per_row = 0;
    stats.min_nnz_per_row = csr.row_ptr[1] - csr.row_ptr[0];

    for (int i = 0; i < csr.rows; ++i) {
        int len = csr.row_ptr[i + 1] - csr.row_ptr[i];
        sum += len;
        if (len > stats.max_nnz_per_row) stats.max_nnz_per_row = len;
        if (len < stats.min_nnz_per_row) stats.min_nnz_per_row = len;
    }

    stats.mean_nnz_per_row = sum / csr.rows;

    double var_sum = 0.0;
    for (int i = 0; i < csr.rows; ++i) {
        int len = csr.row_ptr[i + 1] - csr.row_ptr[i];
        double diff = len - stats.mean_nnz_per_row;
        var_sum += diff * diff;
    }
    stats.std_nnz_per_row = std::sqrt(var_sum / csr.rows);
    stats.irregularity = (stats.mean_nnz_per_row > 0) ? (stats.max_nnz_per_row / stats.mean_nnz_per_row) : 0.0;

    return stats;
}

std::vector<double> load_vector(const std::string &path, int size) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open vector file: " + path);
    }
    std::vector<double> vec;
    vec.reserve(size);
    double val = 0.0;
    while (file >> val) {
        vec.push_back(val);
    }
    if ((int)vec.size() < size) {
        throw std::runtime_error("Vector file " + path + " has fewer elements (" + 
                                 std::to_string(vec.size()) + ") than expected (" + 
                                 std::to_string(size) + ")");
    }
    if ((int)vec.size() > size) {
        vec.resize(size);
    }
    return vec;
}

void save_vector(const std::string &path, const double *vec, int size) {
    FILE *f = std::fopen(path.c_str(), "w");
    if (!f) {
        throw std::runtime_error("Cannot write output vector: " + path);
    }
    for (int i = 0; i < size; ++i) {
        std::fprintf(f, "%.17g\n", vec[i]);
    }
    std::fclose(f);
}

void save_vector(const std::string &path, const std::vector<double> &vec) {
    save_vector(path, vec.data(), (int)vec.size());
}
