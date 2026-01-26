/**
 * Successive Over-Relaxation (SOR) Method for Laplace Equation
 * High Performance Computing - Master's Course
 *
 * Algorithm:
 *   aver = 0.25 * (u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1])
 *   res = aver - u[i][j]
 *   u[i][j] = u[i][j] + omega * res
 *
 * Or equivalently:
 *   u[i][j] = (1 - omega) * u[i][j] + omega * aver
 *
 * Parameters:
 *   - omega = 1.0: Gauss-Seidel
 *   - 0 < omega < 1: Under-relaxation
 *   - 1 < omega < 2: Over-relaxation (faster convergence)
 *   - omega >= 2: Diverges
 *
 * Optimal omega (Young, 1954):
 *   omega_opt = 2 - 2*pi/n  (for large n)
 *
 * This implementation uses Red-Black (Parity) ordering for parallelizability.
 */

#include "laplace_common.h"

/**
 * Compute optimal relaxation factor
 * Based on Young's theory for square grids
 */
double compute_optimal_omega(int n) {
    return 2.0 - 2.0 * M_PI / n;
}

/**
 * Perform one SOR iteration with row-wise ordering
 * Returns the maximum change (delta) in this iteration
 */
double sor_iteration_rowwise(double** u, int n, double omega) {
    double delta = 0.0;

    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++) {
            double old_value = u[i][j];

            // Compute average of neighbors
            double aver = 0.25 * (u[i-1][j] + u[i+1][j] +
                                  u[i][j-1] + u[i][j+1]);

            // Compute residual and apply relaxation
            double res = aver - old_value;
            u[i][j] = old_value + omega * res;

            // Track maximum change
            double diff = fabs(u[i][j] - old_value);
            if (diff > delta) {
                delta = diff;
            }
        }
    }

    return delta;
}

/**
 * Perform one SOR iteration with Red-Black (Parity) ordering
 * This ordering allows parallelization!
 *
 * Parity of (i,j) = (i + j) % 2
 *   - Even (0): "Red" points
 *   - Odd (1):  "Black" points
 *
 * Red points only depend on Black points and vice versa.
 *
 * Returns the maximum change (delta) in this iteration
 */
double sor_iteration_redblack(double** u, int n, double omega) {
    double delta = 0.0;

    // Phase 1: Update all RED points (parity 0)
    for (int i = 1; i <= n; i++) {
        // Start j so that (i + j) % 2 == 0
        int j_start = (i % 2 == 0) ? 2 : 1;
        for (int j = j_start; j <= n; j += 2) {
            double old_value = u[i][j];

            double aver = 0.25 * (u[i-1][j] + u[i+1][j] +
                                  u[i][j-1] + u[i][j+1]);
            double res = aver - old_value;
            u[i][j] = old_value + omega * res;

            double diff = fabs(u[i][j] - old_value);
            if (diff > delta) delta = diff;
        }
    }

    // Phase 2: Update all BLACK points (parity 1)
    for (int i = 1; i <= n; i++) {
        // Start j so that (i + j) % 2 == 1
        int j_start = (i % 2 == 0) ? 1 : 2;
        for (int j = j_start; j <= n; j += 2) {
            double old_value = u[i][j];

            double aver = 0.25 * (u[i-1][j] + u[i+1][j] +
                                  u[i][j-1] + u[i][j+1]);
            double res = aver - old_value;
            u[i][j] = old_value + omega * res;

            double diff = fabs(u[i][j] - old_value);
            if (diff > delta) delta = diff;
        }
    }

    return delta;
}

/**
 * SOR solver main function
 * use_redblack: 0 = row-wise, 1 = red-black ordering
 */
int sor_solve(double** u, int n, double omega, double tolerance,
              int max_iter, int use_redblack) {
    int iter;
    double delta;

    printf("Using omega = %.6f (optimal = %.6f)\n", omega, compute_optimal_omega(n));
    printf("Ordering: %s\n\n", use_redblack ? "Red-Black" : "Row-wise");

    for (iter = 1; iter <= max_iter; iter++) {
        // Perform one iteration
        if (use_redblack) {
            delta = sor_iteration_redblack(u, n, omega);
        } else {
            delta = sor_iteration_rowwise(u, n, omega);
        }

        // Check convergence
        if (delta < tolerance) {
            break;
        }

        // Progress indicator every 100 iterations (SOR converges much faster)
        if (iter % 100 == 0) {
            printf("Iteration %d: delta = %.2e\n", iter, delta);
        }
    }

    return iter;
}

/**
 * Main function
 */
int main(int argc, char* argv[]) {
    // Parse command line arguments
    int n = (argc > 1) ? atoi(argv[1]) : DEFAULT_N;
    double tolerance = (argc > 2) ? atof(argv[2]) : DEFAULT_TOLERANCE;
    int max_iter = (argc > 3) ? atoi(argv[3]) : DEFAULT_MAX_ITER;
    int use_redblack = (argc > 4) ? atoi(argv[4]) : 1;  // Default: red-black
    double omega = (argc > 5) ? atof(argv[5]) : compute_optimal_omega(n);

    printf("SOR (Successive Over-Relaxation) for Laplace Equation\n");
    printf("Grid: %d x %d, Tolerance: %.2e, Max iterations: %d\n",
           n, n, tolerance, max_iter);

    // Allocate and initialize grid
    double** u = allocate_grid(n);
    initialize_grid(u, n, BC_TOP, BC_BOTTOM, BC_LEFT, BC_RIGHT, INITIAL_GUESS);

    // Solve
    double start_time = get_time();
    int iterations = sor_solve(u, n, omega, tolerance, max_iter, use_redblack);
    double end_time = get_time();

    // Compute final delta for reporting
    double final_delta = use_redblack ?
        sor_iteration_redblack(u, n, omega) :
        sor_iteration_rowwise(u, n, omega);

    // Print results
    char method_name[64];
    snprintf(method_name, sizeof(method_name), "SOR (omega=%.3f, %s)",
             omega, use_redblack ? "Red-Black" : "Row-wise");
    print_statistics(method_name, n, iterations, final_delta, end_time - start_time);

    // Save solution to file
    save_grid_to_file(u, n, "sor_solution.dat");

    // Cleanup
    free_grid(u, n);

    return 0;
}
