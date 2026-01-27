/*
 * Parallel SOR (Red-Black) - OpenMP
 * Compile: gcc -O3 -fopenmp -o sor_omp sor_omp.c -lm
 * Run:     ./sor_omp 100 1e-6 10000 [num_threads] [omega]
 */

#include "laplace_common.h"
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

double sor_redblack_step(double *u, int n, double omega)
{
    int size = n + 2;
    double max_diff = 0.0;

#pragma omp parallel
    {
        double thread_max = 0.0;

// Red sweep: (i+j) % 2 == 0
#pragma omp for nowait
        for (int i = 1; i <= n; i++)
        {
            int row_off = i * size;
            int js = (i % 2 == 0) ? 2 : 1;
            for (int j = js; j <= n; j += 2)
            {
                int idx = row_off + j;
                double old = u[idx];
                double avg = 0.25 * (u[idx - size] + u[idx + size] + u[idx - 1] + u[idx + 1]);
                u[idx] = old + omega * (avg - old);
                double diff = fabs(u[idx] - old);
                if (diff > thread_max)
                    thread_max = diff;
            }
        }

#pragma omp barrier

// Black sweep: (i+j) % 2 == 1
#pragma omp for
        for (int i = 1; i <= n; i++)
        {
            int row_off = i * size;
            int js = (i % 2 == 0) ? 1 : 2;
            for (int j = js; j <= n; j += 2)
            {
                int idx = row_off + j;
                double old = u[idx];
                double avg = 0.25 * (u[idx - size] + u[idx + size] + u[idx - 1] + u[idx + 1]);
                u[idx] = old + omega * (avg - old);
                double diff = fabs(u[idx] - old);
                if (diff > thread_max)
                    thread_max = diff;
            }
        }

#pragma omp critical
        {
            if (thread_max > max_diff)
                max_diff = thread_max;
        }
    }

    return max_diff;
}

int main(int argc, char *argv[])
{
    int n = (argc > 1) ? atoi(argv[1]) : DEFAULT_N;
    double tol = (argc > 2) ? atof(argv[2]) : DEFAULT_TOL;
    int maxiter = (argc > 3) ? atoi(argv[3]) : DEFAULT_MAX_ITER;
    int nthreads = (argc > 4) ? atoi(argv[4]) : omp_get_max_threads();
    double omega = (argc > 5) ? atof(argv[5]) : 2.0 - 2.0 * M_PI / n;

    omp_set_num_threads(nthreads);

    printf("SOR OpenMP (Red-Black): %dx%d grid, %d threads, omega=%.4f\n", n, n, nthreads, omega);

    double *u = alloc_grid(n);
    init_grid(u, n);

    double t0 = omp_get_wtime();
    int iter;
    double delta;
    for (iter = 1; iter <= maxiter; iter++)
    {
        delta = sor_redblack_step(u, n, omega);
        if (delta < tol)
        {
            break;
        }
        if (iter % 100 == 0)
        {
            printf("iter %d: delta=%.2e\n", iter, delta);
        }
    }

    double t1 = omp_get_wtime();
    print_result("SOR OpenMP", n, iter, delta, t1 - t0);
    print_grid(u, n, "Final Solution");
    free_grid(u);
    return 0;
}