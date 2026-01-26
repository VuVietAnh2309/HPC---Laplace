/*
 * SOR (Successive Over-Relaxation) for Laplace Equation
 * Compile: gcc -O3 -o sor sor.c -lm
 * Run:     ./sor 100 1e-6 10000 [redblack=1] [omega]
 */

#include "laplace_common.h"

double sor_step_rowwise(double **u, int n, double omega) {
    double maxdiff = 0.0;
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++) {
            double old = u[i][j];
            double avg = 0.25 * (u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1]);
            u[i][j] = old + omega * (avg - old);
            double diff = fabs(u[i][j] - old);
            if (diff > maxdiff) maxdiff = diff;
        }
    }
    return maxdiff;
}

double sor_step_redblack(double **u, int n, double omega) {
    double maxdiff = 0.0;

    // red points: (i+j) % 2 == 0
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

    // black points: (i+j) % 2 == 1
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

int main(int argc, char *argv[]) {
    int n = (argc > 1) ? atoi(argv[1]) : DEFAULT_N;
    double tol = (argc > 2) ? atof(argv[2]) : DEFAULT_TOL;
    int maxiter = (argc > 3) ? atoi(argv[3]) : DEFAULT_MAX_ITER;
    int redblack = (argc > 4) ? atoi(argv[4]) : 1;
    double omega = (argc > 5) ? atof(argv[5]) : 2.0 - 2.0 * M_PI / n;

    printf("SOR: %dx%d grid, omega=%.4f, %s\n", n, n, omega,
           redblack ? "red-black" : "row-wise");

    double **u = alloc_grid(n);
    init_grid(u, n);

    double t0 = get_time();
    int iter;
    double delta;

    for (iter = 1; iter <= maxiter; iter++) {
        delta = redblack ? sor_step_redblack(u, n, omega)
                         : sor_step_rowwise(u, n, omega);
        if (delta < tol) break;

        if (iter % 100 == 0)
            printf("iter %d: delta=%.2e\n", iter, delta);
    }

    double t1 = get_time();
    print_result("SOR", n, iter, delta, t1 - t0);
    print_matrix(u, n, "Final Solution");

    free_grid(u, n);
    return 0;
}
