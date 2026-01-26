/*
 * Parallel SOR (Red-Black) - OpenMP
 * Compile: gcc -O3 -fopenmp -o sor_omp sor_omp.c -lm
 * Run:     ./sor_omp 100 1e-6 10000 [num_threads] [omega]
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define DEFAULT_N 100
#define DEFAULT_MAX_ITER 100000
#define DEFAULT_TOL 1e-6

double **alloc_grid(int n) {
    double **g = malloc((n + 2) * sizeof(double*));
    for (int i = 0; i < n + 2; i++)
        g[i] = malloc((n + 2) * sizeof(double));
    return g;
}

void free_grid(double **g, int n) {
    for (int i = 0; i < n + 2; i++) free(g[i]);
    free(g);
}

void init_grid(double **u, int n) {
    for (int i = 0; i < n + 2; i++)
        for (int j = 0; j < n + 2; j++)
            u[i][j] = 50.0;

    for (int j = 0; j < n + 2; j++) {
        u[0][j] = 0.0;
        u[n+1][j] = 100.0;
    }
    for (int i = 0; i < n + 2; i++) {
        u[i][0] = 0.0;
        u[i][n+1] = 100.0;
    }
}

double sor_redblack_step(double **u, int n, double omega) {
    double maxdiff = 0.0;

    // red sweep: (i+j) % 2 == 0
    #pragma omp parallel for reduction(max:maxdiff) schedule(static)
    for (int i = 1; i <= n; i++) {
        int js = (i % 2 == 0) ? 2 : 1;
        for (int j = js; j <= n; j += 2) {
            double old = u[i][j];
            double avg = 0.25 * (u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1]);
            u[i][j] = old + omega * (avg - old);
            double diff = fabs(u[i][j] - old);
            if (diff > maxdiff) maxdiff = diff;
        }
    }

    // black sweep: (i+j) % 2 == 1
    #pragma omp parallel for reduction(max:maxdiff) schedule(static)
    for (int i = 1; i <= n; i++) {
        int js = (i % 2 == 0) ? 1 : 2;
        for (int j = js; j <= n; j += 2) {
            double old = u[i][j];
            double avg = 0.25 * (u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1]);
            u[i][j] = old + omega * (avg - old);
            double diff = fabs(u[i][j] - old);
            if (diff > maxdiff) maxdiff = diff;
        }
    }

    return maxdiff;
}

void print_matrix(double **u, int n) {
    int size = n + 2;
    printf("\nFinal Solution (%d x %d):\n", size, size);

    if (n <= 20) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++)
                printf("%7.2f ", u[i][j]);
            printf("\n");
        }
    } else {
        printf("(showing corners)\nTop-left 5x5:\n");
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++)
                printf("%7.2f ", u[i][j]);
            printf("\n");
        }
        printf("\nBottom-right 5x5:\n");
        for (int i = n - 3; i < size; i++) {
            for (int j = n - 3; j < size; j++)
                printf("%7.2f ", u[i][j]);
            printf("\n");
        }
    }
}

int main(int argc, char *argv[]) {
    int n = (argc > 1) ? atoi(argv[1]) : DEFAULT_N;
    double tol = (argc > 2) ? atof(argv[2]) : DEFAULT_TOL;
    int maxiter = (argc > 3) ? atoi(argv[3]) : DEFAULT_MAX_ITER;
    int nthreads = (argc > 4) ? atoi(argv[4]) : omp_get_max_threads();
    double omega = (argc > 5) ? atof(argv[5]) : 2.0 - 2.0 * M_PI / n;

    omp_set_num_threads(nthreads);

    printf("SOR OpenMP (Red-Black): %dx%d grid, %d threads\n", n, n, nthreads);
    printf("omega=%.4f, tol=%.1e\n", omega, tol);

    double **u = alloc_grid(n);
    init_grid(u, n);

    double t0 = omp_get_wtime();
    int iter;
    double delta;

    for (iter = 1; iter <= maxiter; iter++) {
        delta = sor_redblack_step(u, n, omega);
        if (delta < tol) break;

        if (iter % 100 == 0)
            printf("iter %d: delta=%.2e\n", iter, delta);
    }

    double t1 = omp_get_wtime();

    printf("\n=== Results ===\n");
    printf("Iterations: %d\n", iter);
    printf("Final delta: %.2e\n", delta);
    printf("Time: %.4f sec\n", t1 - t0);

    print_matrix(u, n);

    free_grid(u, n);
    return 0;
}
