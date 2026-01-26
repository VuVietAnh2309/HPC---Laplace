/**
 * Parallel SOR with Red-Black Ordering using MPI
 * High Performance Computing - Master's Course
 *
 * Key Concepts:
 *   - SOR (Successive Over-Relaxation) accelerates convergence
 *   - Red-Black ordering enables parallelization of Gauss-Seidel/SOR
 *   - All RED points can be updated simultaneously (depend only on BLACK)
 *   - All BLACK points can be updated simultaneously (depend only on RED)
 *
 * Algorithm per iteration:
 *   1. Exchange ghost rows (BLACK values needed)
 *   2. Update all RED points in parallel
 *   3. Exchange ghost rows (RED values needed)
 *   4. Update all BLACK points in parallel
 *   5. Check convergence (global reduction)
 *
 * Compile: mpicc -O3 -o sor_mpi sor_mpi.c -lm
 * Run:     mpirun -np 4 ./sor_mpi 100 1e-6 10000
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Default parameters
#define DEFAULT_N 100
#define DEFAULT_MAX_ITER 100000
#define DEFAULT_TOLERANCE 1e-6

// Boundary conditions
#define BC_TOP    0.0
#define BC_BOTTOM 100.0
#define BC_RIGHT  100.0
#define BC_LEFT   0.0
#define INITIAL_GUESS 50.0

// Parity constants
#define RED   0
#define BLACK 1

/**
 * Allocate 2D array as contiguous memory
 */
double* allocate_array(int rows, int cols) {
    double* arr = (double*)calloc(rows * cols, sizeof(double));
    if (!arr) {
        fprintf(stderr, "Memory allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    return arr;
}

#define IDX(i, j, cols) ((i) * (cols) + (j))

/**
 * Compute parity of grid point
 * global_i, global_j are 1-indexed global coordinates
 */
static inline int get_parity(int global_i, int global_j) {
    return (global_i + global_j) % 2;
}

/**
 * Compute optimal omega for SOR
 */
double compute_optimal_omega(int n) {
    return 2.0 - 2.0 * M_PI / n;
}

/**
 * Initialize local grid
 */
void initialize_local_grid(double* u, int local_rows, int n,
                           int global_start_row, int rank, int size) {
    int cols = n + 2;
    int rows = local_rows + 2;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            u[IDX(i, j, cols)] = INITIAL_GUESS;
        }
    }

    // Left boundary
    for (int i = 0; i < rows; i++) {
        u[IDX(i, 0, cols)] = BC_LEFT;
    }

    // Right boundary
    for (int i = 0; i < rows; i++) {
        u[IDX(i, n + 1, cols)] = BC_RIGHT;
    }

    // Top boundary (rank 0 only)
    if (rank == 0) {
        for (int j = 0; j < cols; j++) {
            u[IDX(0, j, cols)] = BC_TOP;
        }
    }

    // Bottom boundary (last rank only)
    if (rank == size - 1) {
        for (int j = 0; j < cols; j++) {
            u[IDX(local_rows + 1, j, cols)] = BC_BOTTOM;
        }
    }
}

/**
 * Exchange ghost rows with neighbors
 */
void exchange_boundaries(double* u, int local_rows, int n, int rank, int size) {
    int cols = n + 2;
    MPI_Status status;

    // Exchange with upper neighbor
    if (rank > 0) {
        MPI_Sendrecv(&u[IDX(1, 0, cols)], cols, MPI_DOUBLE, rank - 1, 0,
                     &u[IDX(0, 0, cols)], cols, MPI_DOUBLE, rank - 1, 1,
                     MPI_COMM_WORLD, &status);
    }

    // Exchange with lower neighbor
    if (rank < size - 1) {
        MPI_Sendrecv(&u[IDX(local_rows, 0, cols)], cols, MPI_DOUBLE, rank + 1, 1,
                     &u[IDX(local_rows + 1, 0, cols)], cols, MPI_DOUBLE, rank + 1, 0,
                     MPI_COMM_WORLD, &status);
    }
}

/**
 * Update points of given parity using SOR
 * Returns local maximum change
 */
double sor_update_parity(double* u, int local_rows, int n,
                         int global_start_row, double omega, int parity) {
    int cols = n + 2;
    double local_delta = 0.0;

    for (int i = 1; i <= local_rows; i++) {
        int global_i = global_start_row + i - 1;  // Convert to global row index

        for (int j = 1; j <= n; j++) {
            // Check if this point has the required parity
            if (get_parity(global_i, j) != parity) {
                continue;
            }

            int idx = IDX(i, j, cols);
            double old_value = u[idx];

            // Compute average of neighbors
            double aver = 0.25 * (u[IDX(i-1, j, cols)] +
                                  u[IDX(i+1, j, cols)] +
                                  u[IDX(i, j-1, cols)] +
                                  u[IDX(i, j+1, cols)]);

            // SOR update
            double res = aver - old_value;
            u[idx] = old_value + omega * res;

            // Track change
            double diff = fabs(u[idx] - old_value);
            if (diff > local_delta) {
                local_delta = diff;
            }
        }
    }

    return local_delta;
}

/**
 * Gather full solution to rank 0
 */
void gather_solution(double* local_u, double* global_u, int local_rows, int n,
                     int* all_local_rows, int* displs, int rank, int size) {
    int cols = n + 2;
    int send_count = local_rows * cols;

    int* recv_counts = NULL;
    int* recv_displs = NULL;

    if (rank == 0) {
        recv_counts = (int*)malloc(size * sizeof(int));
        recv_displs = (int*)malloc(size * sizeof(int));

        int offset = cols;
        for (int p = 0; p < size; p++) {
            recv_counts[p] = all_local_rows[p] * cols;
            recv_displs[p] = offset;
            offset += all_local_rows[p] * cols;
        }

        for (int j = 0; j < cols; j++) {
            global_u[IDX(0, j, cols)] = BC_TOP;
            global_u[IDX(n + 1, j, cols)] = BC_BOTTOM;
        }
        for (int i = 0; i < n + 2; i++) {
            global_u[IDX(i, 0, cols)] = BC_LEFT;
            global_u[IDX(i, n + 1, cols)] = BC_RIGHT;
        }
    }

    MPI_Gatherv(&local_u[IDX(1, 0, cols)], send_count, MPI_DOUBLE,
                global_u, recv_counts, recv_displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        free(recv_counts);
        free(recv_displs);
    }
}

int main(int argc, char* argv[]) {
    int rank, size;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Parse arguments
    int n = (argc > 1) ? atoi(argv[1]) : DEFAULT_N;
    double tolerance = (argc > 2) ? atof(argv[2]) : DEFAULT_TOLERANCE;
    int max_iter = (argc > 3) ? atoi(argv[3]) : DEFAULT_MAX_ITER;
    double omega = (argc > 4) ? atof(argv[4]) : compute_optimal_omega(n);

    if (rank == 0) {
        printf("Parallel SOR with Red-Black Ordering (MPI)\n");
        printf("Grid: %d x %d, Processors: %d\n", n, n, size);
        printf("Omega: %.6f (optimal: %.6f)\n", omega, compute_optimal_omega(n));
        printf("Tolerance: %.2e, Max iterations: %d\n\n", tolerance, max_iter);
    }

    // Domain decomposition
    int base_rows = n / size;
    int extra = n % size;

    int* all_local_rows = (int*)malloc(size * sizeof(int));
    int* displs = (int*)malloc(size * sizeof(int));

    int offset = 0;
    for (int p = 0; p < size; p++) {
        all_local_rows[p] = base_rows + (p < extra ? 1 : 0);
        displs[p] = offset;
        offset += all_local_rows[p];
    }

    int local_rows = all_local_rows[rank];
    int global_start_row = displs[rank] + 1;

    if (rank == 0) {
        printf("Domain decomposition:\n");
        for (int p = 0; p < size; p++) {
            printf("  Processor %d: %d rows (global rows %d-%d)\n",
                   p, all_local_rows[p],
                   displs[p] + 1, displs[p] + all_local_rows[p]);
        }
        printf("\n");
    }

    // Allocate local array
    int cols = n + 2;
    int local_total_rows = local_rows + 2;
    double* u = allocate_array(local_total_rows, cols);

    // Initialize
    initialize_local_grid(u, local_rows, n, global_start_row, rank, size);

    // Main iteration loop
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    int iter;
    double global_delta;

    for (iter = 1; iter <= max_iter; iter++) {
        double local_delta_red, local_delta_black;
        double local_delta, temp_delta;

        // Phase 1: Update RED points
        exchange_boundaries(u, local_rows, n, rank, size);
        local_delta_red = sor_update_parity(u, local_rows, n, global_start_row, omega, RED);

        // Phase 2: Update BLACK points
        exchange_boundaries(u, local_rows, n, rank, size);
        local_delta_black = sor_update_parity(u, local_rows, n, global_start_row, omega, BLACK);

        // Local delta is max of red and black
        local_delta = (local_delta_red > local_delta_black) ? local_delta_red : local_delta_black;

        // Global reduction
        MPI_Allreduce(&local_delta, &global_delta, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        // Check convergence
        if (global_delta < tolerance) {
            break;
        }

        // Progress (only rank 0, every 100 iterations since SOR converges fast)
        if (rank == 0 && iter % 100 == 0) {
            printf("Iteration %d: delta = %.2e\n", iter, global_delta);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();

    // Print results
    if (rank == 0) {
        printf("\n=== Parallel SOR Red-Black (MPI) Results ===\n");
        printf("Grid size:        %d x %d\n", n, n);
        printf("Processors:       %d\n", size);
        printf("Omega:            %.6f\n", omega);
        printf("Iterations:       %d\n", iter);
        printf("Final delta:      %.2e\n", global_delta);
        printf("Elapsed time:     %.4f seconds\n", end_time - start_time);
        printf("Time/iteration:   %.6f ms\n", ((end_time - start_time) / iter) * 1000);
        printf("=============================================\n");
    }

    // Gather and save solution
    double* global_u = NULL;
    if (rank == 0) {
        global_u = allocate_array(n + 2, cols);
    }

    gather_solution(u, global_u, local_rows, n, all_local_rows, displs, rank, size);

    if (rank == 0) {
        FILE* fp = fopen("sor_mpi_solution.dat", "w");
        if (fp) {
            fprintf(fp, "# Parallel SOR Red-Black MPI Solution\n");
            fprintf(fp, "# Grid: %d x %d, Processors: %d, Omega: %.6f\n", n, n, size, omega);
            for (int i = 0; i < n + 2; i++) {
                for (int j = 0; j < cols; j++) {
                    fprintf(fp, "%d %d %.10f\n", i, j, global_u[IDX(i, j, cols)]);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
            printf("Solution saved to sor_mpi_solution.dat\n");
        }
        free(global_u);
    }

    // Cleanup
    free(u);
    free(all_local_rows);
    free(displs);

    MPI_Finalize();
    return 0;
}
