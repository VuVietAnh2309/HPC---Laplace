/*
 * Gauss-Seidel Iteration for Laplace Equation
 * Compile: gcc -O3 -o gauss_seidel gauss_seidel.c -lm
 * Run:     ./gauss_seidel 100 1e-6 10000
 */

#include "laplace_common.h"

double gs_step(double **u, int n) {
    double maxdiff = 0.0;
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++) {
            double old = u[i][j];
            u[i][j] = 0.25 * (u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1]);
            double diff = fabs(u[i][j] - old);
            if (diff > maxdiff) maxdiff = diff;
        }
    }
    return maxdiff;
}

int main(int argc, char *argv[]) {
    int n = (argc > 1) ? atoi(argv[1]) : DEFAULT_N;
    double tol = (argc > 2) ? atof(argv[2]) : DEFAULT_TOL;
    int maxiter = (argc > 3) ? atoi(argv[3]) : DEFAULT_MAX_ITER;

    printf("Gauss-Seidel: %dx%d grid, tol=%.1e\n", n, n, tol);

    double **u = alloc_grid(n);
    init_grid(u, n);

    double t0 = get_time();
    int iter;
    double delta;

    for (iter = 1; iter <= maxiter; iter++) {
        delta = gs_step(u, n);
        if (delta < tol) break;

        if (iter % 1000 == 0)
            printf("iter %d: delta=%.2e\n", iter, delta);
    }

    double t1 = get_time();
    print_result("Gauss-Seidel", n, iter, delta, t1 - t0);
    print_matrix(u, n, "Final Solution");

    free_grid(u, n);
    return 0;
}
