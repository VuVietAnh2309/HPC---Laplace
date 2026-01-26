/**
 * Jacobi Iteration Method for Laplace Equation
 * High Performance Computing - Master's Course
 *
 * Algorithm:
 *   u_new[i][j] = 0.25 * (u_old[i-1][j] + u_old[i+1][j] +
 *                         u_old[i][j-1] + u_old[i][j+1])
 *
 * Characteristics:
 *   - Uses old values for all neighbors
 *   - Requires two arrays (old and new)
 *   - Easily parallelizable
 *   - Slowest convergence among iterative methods
 */

#include "laplace_common.h"

/**
 * Perform one Jacobi iteration
 * Returns the maximum change (delta) in this iteration
 */
double jacobi_iteration(double **u_old, double **u_new, int n)
{
    double delta = 0.0;

    for (int i = 1; i <= n; i++)
    {
        for (int j = 1; j <= n; j++)
        {
            // Five-point stencil average
            u_new[i][j] = 0.25 * (u_old[i - 1][j] + u_old[i + 1][j] +
                                  u_old[i][j - 1] + u_old[i][j + 1]);

            // Track maximum change
            double diff = fabs(u_new[i][j] - u_old[i][j]);
            if (diff > delta)
            {
                delta = diff;
            }
        }
    }

    return delta;
}

/**
 * Jacobi solver main function
 */
int jacobi_solve(double **u, int n, double tolerance, int max_iter)
{
    // Allocate temporary grid for new values
    double **u_new = allocate_grid(n);

    // Copy initial values including boundaries
    copy_grid(u_new, u, n);

    int iter;
    double delta;

    for (iter = 1; iter <= max_iter; iter++)
    {
        // Perform one iteration
        delta = jacobi_iteration(u, u_new, n);

        // Swap grids (swap pointers for interior, boundaries stay the same)
        double **temp = u;
        for (int i = 1; i <= n; i++)
        {
            for (int j = 1; j <= n; j++)
            {
                u[i][j] = u_new[i][j];
            }
        }

        // Check convergence
        if (delta < tolerance)
        {
            break;
        }

        // Progress indicator every 1000 iterations
        if (iter % 1000 == 0)
        {
            printf("Iteration %d: delta = %.2e\n", iter, delta);
        }
    }

    free_grid(u_new, n);
    return iter;
}

/**
 * Main function
 */
int main(int argc, char *argv[])
{
    // Parse command line arguments
    int n = (argc > 1) ? atoi(argv[1]) : DEFAULT_N;
    double tolerance = (argc > 2) ? atof(argv[2]) : DEFAULT_TOLERANCE;
    int max_iter = (argc > 3) ? atoi(argv[3]) : DEFAULT_MAX_ITER;

    printf("Jacobi Iteration for Laplace Equation\n");
    printf("Grid: %d x %d, Tolerance: %.2e, Max iterations: %d\n\n",
           n, n, tolerance, max_iter);

    // Allocate and initialize grid
    double **u = allocate_grid(n);
    initialize_grid(u, n, BC_TOP, BC_BOTTOM, BC_LEFT, BC_RIGHT, INITIAL_GUESS);

    // Solve
    double start_time = get_time();
    int iterations = jacobi_solve(u, n, tolerance, max_iter);
    double end_time = get_time();

    // Compute final delta for reporting
    double **u_check = allocate_grid(n);
    copy_grid(u_check, u, n);
    double final_delta = jacobi_iteration(u, u_check, n);
    free_grid(u_check, n);

    // Print results
    print_statistics("Jacobi", n, iterations, final_delta, end_time - start_time);

    // Save solution to file
    save_grid_to_file(u, n, "jacobi_solution.dat");

    // Cleanup
    free_grid(u, n);

    return 0;
}
