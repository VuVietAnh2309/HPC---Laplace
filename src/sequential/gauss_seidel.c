/**
 * Gauss-Seidel Iteration Method for Laplace Equation
 * High Performance Computing - Master's Course
 *
 * Algorithm:
 *   u[i][j] = 0.25 * (u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1])
 *
 * Difference from Jacobi:
 *   - Uses NEW values as soon as they are computed
 *   - u[i-1][j] and u[i][j-1] are already updated in this iteration
 *   - Requires only ONE array (in-place update)
 *   - Converges ~2x faster than Jacobi
 *   - Harder to parallelize (inherently sequential in row-wise order)
 */

#include "laplace_common.h"

/**
 * Perform one Gauss-Seidel iteration (row-wise ordering)
 * Returns the maximum change (delta) in this iteration
 *
 * Note: This updates in-place, so:
 *   - u[i-1][j] is from the NEW iteration (already updated)
 *   - u[i+1][j] is from the OLD iteration (not yet updated)
 *   - u[i][j-1] is from the NEW iteration (already updated)
 *   - u[i][j+1] is from the OLD iteration (not yet updated)
 */
double gauss_seidel_iteration(double** u, int n) {
    double delta = 0.0;

    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++) {
            double old_value = u[i][j];

            // Five-point stencil average (using latest values)
            u[i][j] = 0.25 * (u[i-1][j] + u[i+1][j] +
                              u[i][j-1] + u[i][j+1]);

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
 * Gauss-Seidel solver main function
 */
int gauss_seidel_solve(double** u, int n, double tolerance, int max_iter) {
    int iter;
    double delta;

    for (iter = 1; iter <= max_iter; iter++) {
        // Perform one iteration (in-place)
        delta = gauss_seidel_iteration(u, n);

        // Check convergence
        if (delta < tolerance) {
            break;
        }

        // Progress indicator every 1000 iterations
        if (iter % 1000 == 0) {
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

    printf("Gauss-Seidel Iteration for Laplace Equation\n");
    printf("Grid: %d x %d, Tolerance: %.2e, Max iterations: %d\n\n",
           n, n, tolerance, max_iter);

    // Allocate and initialize grid
    double** u = allocate_grid(n);
    initialize_grid(u, n, BC_TOP, BC_BOTTOM, BC_LEFT, BC_RIGHT, INITIAL_GUESS);

    // Solve
    double start_time = get_time();
    int iterations = gauss_seidel_solve(u, n, tolerance, max_iter);
    double end_time = get_time();

    // Compute final delta for reporting
    double final_delta = gauss_seidel_iteration(u, n);

    // Print results
    print_statistics("Gauss-Seidel", n, iterations, final_delta, end_time - start_time);

    // Save solution to file
    save_grid_to_file(u, n, "gauss_seidel_solution.dat");

    // Cleanup
    free_grid(u, n);

    return 0;
}
