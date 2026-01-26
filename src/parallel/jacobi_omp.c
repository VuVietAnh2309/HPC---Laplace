/*
 * Parallel Jacobi Iteration - OpenMP
 * Compile: gcc -O3 -fopenmp -o jacobi_omp jacobi_omp.c -lm
 * Run:     ./jacobi_omp 100 1e-6 10000 [num_threads]
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define DEFAULT_N 100
#define DEFAULT_MAX_ITER 100000
#define DEFAULT_TOL 1e-6

double **alloc_grid(int n)
{
    double **g = malloc((n + 2) * sizeof(double *));
    for (int i = 0; i < n + 2; i++)
        g[i] = malloc((n + 2) * sizeof(double));
    return g;
}

void free_grid(double **g, int n)
{
    for (int i = 0; i < n + 2; i++)
        free(g[i]);
    free(g);
}

void init_grid(double **u, int n)
{
    // boundaries: top=0, bottom=100, left=0, right=100
    for (int i = 0; i < n + 2; i++)
        for (int j = 0; j < n + 2; j++)
            u[i][j] = 50.0;

    for (int j = 0; j < n + 2; j++)
    {
        u[0][j] = 0.0;
        u[n + 1][j] = 100.0;
    }
    for (int i = 0; i < n + 2; i++)
    {
        u[i][0] = 0.0;
        u[i][n + 1] = 100.0;
    }
}

double jacobi_iterate(double **u, double **unew, int n)
{
    double maxdiff = 0.0;

#pragma omp parallel for reduction(max : maxdiff) schedule(static)
    for (int i = 1; i <= n; i++)
    {
        for (int j = 1; j <= n; j++)
        {
            double val = 0.25 * (u[i - 1][j] + u[i + 1][j] + u[i][j - 1] + u[i][j + 1]);
            unew[i][j] = val;
            double diff = fabs(val - u[i][j]);
            if (diff > maxdiff)
                maxdiff = diff;
        }
    }
    return maxdiff;
}

void print_matrix(double **u, int n)
{
    int size = n + 2;
    printf("\nFinal Solution (%d x %d):\n", size, size);

    if (n <= 20)
    {
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
                printf("%7.2f ", u[i][j]);
            printf("\n");
        }
    }
    else
    {
        printf("(showing corners)\nTop-left 5x5:\n");
        for (int i = 0; i < 5; i++)
        {
            for (int j = 0; j < 5; j++)
                printf("%7.2f ", u[i][j]);
            printf("\n");
        }
        printf("\nBottom-right 5x5:\n");
        for (int i = n - 3; i < size; i++)
        {
            for (int j = n - 3; j < size; j++)
                printf("%7.2f ", u[i][j]);
            printf("\n");
        }
    }
}

int main(int argc, char *argv[])
{
    int n = (argc > 1) ? atoi(argv[1]) : DEFAULT_N;
    double tol = (argc > 2) ? atof(argv[2]) : DEFAULT_TOL;
    int maxiter = (argc > 3) ? atoi(argv[3]) : DEFAULT_MAX_ITER;
    int nthreads = (argc > 4) ? atoi(argv[4]) : omp_get_max_threads();

    omp_set_num_threads(nthreads);

    printf("Jacobi OpenMP: %dx%d grid, %d threads, tol=%.1e\n", n, n, nthreads, tol);

    double **u = alloc_grid(n);
    double **unew = alloc_grid(n);
    init_grid(u, n);
    init_grid(unew, n);

    double t0 = omp_get_wtime();
    int iter;
    double delta;

    for (iter = 1; iter <= maxiter; iter++)
    {
        delta = jacobi_iterate(u, unew, n);
        if (delta < tol)
            break;

        double **tmp = u;
        u = unew;
        unew = tmp;

        if (iter % 1000 == 0)
            printf("iter %d: delta=%.2e\n", iter, delta);
    }

    double t1 = omp_get_wtime();

    printf("\n=== Results ===\n");
    printf("Iterations: %d\n", iter);
    printf("Final delta: %.2e\n", delta);
    printf("Time: %.4f sec\n", t1 - t0);

    print_matrix(unew, n);

    free_grid(u, n);
    free_grid(unew, n);
    return 0;
}
