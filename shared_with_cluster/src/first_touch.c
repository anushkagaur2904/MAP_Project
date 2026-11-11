#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(void) {
    long long sizes[] = {10000000LL, 50000000LL};
    int thread_counts[] = {1, 2, 4, 6, 8, 12};

    printf("size,threads,mode,init_ms,parallel_read_ms,checksum\n");

    for (int s = 0; s < 2; s++) {
        long long n = sizes[s];

        for (int t = 0; t < 6; t++) {
            int threads = thread_counts[t];

            for (int mode = 0; mode < 2; mode++) {
                double *a = malloc((size_t)n * sizeof(double));

                if (!a) {
                    fprintf(stderr, "Allocation failed\n");
                    return 2;
                }

                double start = omp_get_wtime();

                if (mode == 0) {
                    for (long long i = 0; i < n; i++)
                        a[i] = 1.0;
                } else {
                    #pragma omp parallel for num_threads(threads) schedule(static)
                    for (long long i = 0; i < n; i++)
                        a[i] = 1.0;
                }

                double init_ms = (omp_get_wtime() - start) * 1000.0;

                double sum = 0.0;
                start = omp_get_wtime();

                #pragma omp parallel for reduction(+:sum) num_threads(threads) schedule(static)
                for (long long i = 0; i < n; i++)
                    sum += a[i];

                double read_ms = (omp_get_wtime() - start) * 1000.0;

                printf("%lld,%d,%s,%.6f,%.6f,%.0f\n",
                    n, threads,
                    mode == 0 ? "serial" : "parallel",
                    init_ms, read_ms, sum);

                free(a);
            }
        }
    }

    return 0;
}