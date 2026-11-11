#include "correctness.h"
#include <cmath>
#include <algorithm>
#include <sstream>

ValidationResult validate_result(const double *computed, const double *reference, int size, 
                                double abs_tol, double rel_tol) {
    ValidationResult res;
    res.passed = true;
    res.max_abs_error = 0.0;
    res.relative_error = 0.0;
    res.has_nan = false;
    res.has_inf = false;

    if (!computed || !reference || size <= 0) {
        res.passed = false;
        res.message = "Null or empty input to validation";
        return res;
    }

    double max_ref_abs = 0.0;
    for (int i = 0; i < size; ++i) {
        double r = reference[i];
        if (std::isnan(r) || std::isinf(r)) continue;
        double ar = std::fabs(r);
        if (ar > max_ref_abs) max_ref_abs = ar;
    }

    for (int i = 0; i < size; ++i) {
        double c = computed[i];
        double r = reference[i];

        if (std::isnan(c)) {
            res.has_nan = true;
            res.passed = false;
            res.error_index = i;
            res.expected_val = r;
            res.actual_val = c;
            res.message = "NaN detected at index " + std::to_string(i);
            return res;
        }

        if (std::isinf(c)) {
            res.has_inf = true;
            res.passed = false;
            res.error_index = i;
            res.expected_val = r;
            res.actual_val = c;
            res.message = "Infinity detected at index " + std::to_string(i);
            return res;
        }

        double diff = std::fabs(c - r);
        if (diff > res.max_abs_error) {
            res.max_abs_error = diff;
            res.error_index = i;
            res.expected_val = r;
            res.actual_val = c;
        }
    }

    if (max_ref_abs > 0.0) {
        res.relative_error = res.max_abs_error / max_ref_abs;
    } else {
        res.relative_error = res.max_abs_error;
    }

    // Allowable tolerance accounting for reduction-order differences in floating-point summation:
    // tol = abs_tol + rel_tol * max_ref_abs
    double allowed = abs_tol + rel_tol * max_ref_abs;
    if (res.max_abs_error > allowed) {
        res.passed = false;
        std::ostringstream oss;
        oss << "Tolerance exceeded: max_abs_error=" << res.max_abs_error 
            << " > allowed=" << allowed << " (at idx " << res.error_index 
            << ", expected " << res.expected_val << ", got " << res.actual_val << ")";
        res.message = oss.str();
    } else {
        res.message = "Passed";
    }

    return res;
}

ValidationResult validate_result(const std::vector<double> &computed, 
                                const std::vector<double> &reference, 
                                double abs_tol, double rel_tol) {
    if (computed.size() != reference.size()) {
        ValidationResult res;
        res.passed = false;
        res.message = "Dimension mismatch: computed=" + std::to_string(computed.size()) + 
                      ", reference=" + std::to_string(reference.size());
        return res;
    }
    return validate_result(computed.data(), reference.data(), (int)computed.size(), abs_tol, rel_tol);
}
