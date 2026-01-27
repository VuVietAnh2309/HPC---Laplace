/*
 * Parallel Jacobi Iteration - OpenMP
 * Compile: gcc -O3 -fopenmp -o jacobi_omp jacobi_omp.c -lm
 * Run:     ./jacobi_omp 100 1e-6 10000 [num_threads]
 */

#include "laplace_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define DEFAULT_N 100
#define DEFAULT_MAX_ITER 100000
#define DEFAULT_TOL 1e-6

double jacobi_iterate(double *u, double *new_u, int n)
{
    int size = n + 2;
    double max_diff = 0.0;
#pragma omp parallel for reduction(max : max_diff) schedule(static)
    for (int row = 1; row <= n; row++)
    {
        for (int col = 1; col <= n; col++)
        {
            int curr = row * size;
            int above = (row - 1) * size;
            int below = (row + 1) * size;
            double val = 0.25 * (u[above + col] + u[below + col] +
                                 u[curr + col - 1] + u[curr + col + 1]);
            new_u[curr + col] = val;
            double diff = fabs(val - u[curr + col]);
            if (diff > max_diff)
            {
                max_diff = diff;
            }
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

    omp_set_num_threads(nthreads);

    printf("Jacobi OpenMP: %dx%d grid, %d threads, tol=%.1e\n", n, n, nthreads, tol);

    double *u = alloc_grid(n);
    double *new_u = alloc_grid(n);
    init_grid(u, n);
    init_grid(new_u, n);

    double t0 = omp_get_wtime();
    int iter;
    double delta;
    for (iter = 1; iter <= maxiter; iter++)
    {
        delta = jacobi_iterate(u, new_u, n);
        if (delta < tol)
        {
            break;
        }
        double **tmp = u;
        u = new_u;
        new_u = tmp;
        if (iter % 1000 == 0)
        {
            printf("iter %d: delta=%.2e\n", iter, delta);
        }
    }

    double t1 = omp_get_wtime();
    print_result("OMP Jacobi", n, iter, delta, t1 - t0);
    print_grid(new_u, n, "Final Solution");
    free_grid(u);
    free_grid(new_u);
    return 0;
}
