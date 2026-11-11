#ifndef CORRECTNESS_H
#define CORRECTNESS_H

#include <vector>
#include <string>

struct ValidationResult {
    bool passed = false;
    double max_abs_error = 0.0;
    double relative_error = 0.0;
    bool has_nan = false;
    bool has_inf = false;
    int error_index = -1;
    double expected_val = 0.0;
    double actual_val = 0.0;
    std::string message;
};

// Documented tolerance for FP64 reduction order differences:
// Default absolute / relative tolerance: 1e-6 (or 1e-8 for FP64)
ValidationResult validate_result(const double *computed, const double *reference, int size, 
                                double abs_tol = 1e-8, double rel_tol = 1e-6);

ValidationResult validate_result(const std::vector<double> &computed, 
                                const std::vector<double> &reference, 
                                double abs_tol = 1e-8, double rel_tol = 1e-6);

#endif // CORRECTNESS_H
