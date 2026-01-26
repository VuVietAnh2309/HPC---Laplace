/**
 * Parallel Jacobi Iteration for Laplace Equation using MPI
 * High Performance Computing - Master's Course
 *
 * Domain Decomposition:
 *   - 1D decomposition (row-wise strips)
 *   - Each processor owns a horizontal strip of the grid
 *   - Ghost rows are exchanged between neighbors
 *
 * Communication Pattern:
 *   - Each processor exchanges one row with top neighbor
 *   - Each processor exchanges one row with bottom neighbor
 *   - Uses MPI_Sendrecv to avoid deadlock
 *
 * Compile: mpicc -O3 -o jacobi_mpi jacobi_mpi.c -lm
 * Run:     mpirun -np 4 ./jacobi_mpi 100 1e-6 10000
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>

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

/**
 * Allocate 2D array as contiguous memory (for MPI communication)
 */
double* allocate_array(int rows, int cols) {
    double* arr = (double*)calloc(rows * cols, sizeof(double));
    if (!arr) {
        fprintf(stderr, "Memory allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    return arr;
}

/**
 * Access 2D array element: arr[i * cols + j]
 */
#define IDX(i, j, cols) ((i) * (cols) + (j))

/**
 * Initialize local grid portion
 * local_rows: number of interior rows owned by this processor (excluding ghosts)
 * n: total number of interior columns
 * global_start_row: global index of first interior row owned by this processor
 */
void initialize_local_grid(double* u, int local_rows, int n,
                           int global_start_row, int rank, int size) {
    int cols = n + 2;  // Include left and right boundaries
    int rows = local_rows + 2;  // Include ghost rows

    // Initialize everything to initial guess
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            u[IDX(i, j, cols)] = INITIAL_GUESS;
        }
    }

    // Left boundary (column 0)
    for (int i = 0; i < rows; i++) {
        u[IDX(i, 0, cols)] = BC_LEFT;
    }

    // Right boundary (column n+1)
    for (int i = 0; i < rows; i++) {
        u[IDX(i, n + 1, cols)] = BC_RIGHT;
    }

    // Top boundary (only for rank 0)
    if (rank == 0) {
        for (int j = 0; j < cols; j++) {
            u[IDX(0, j, cols)] = BC_TOP;
        }
    }

    // Bottom boundary (only for last rank)
    if (rank == size - 1) {
        for (int j = 0; j < cols; j++) {
            u[IDX(local_rows + 1, j, cols)] = BC_BOTTOM;
        }
    }
}

/**
 * Exchange ghost rows with neighboring processors
 */
void exchange_boundaries(double* u, int local_rows, int n, int rank, int size) {
    int cols = n + 2;
    MPI_Status status;

    // Send my first interior row UP, receive into bottom ghost row FROM BELOW
    if (rank > 0) {
        MPI_Sendrecv(&u[IDX(1, 0, cols)], cols, MPI_DOUBLE, rank - 1, 0,
                     &u[IDX(0, 0, cols)], cols, MPI_DOUBLE, rank - 1, 1,
                     MPI_COMM_WORLD, &status);
    }

    // Send my last interior row DOWN, receive into top ghost row FROM ABOVE
    if (rank < size - 1) {
        MPI_Sendrecv(&u[IDX(local_rows, 0, cols)], cols, MPI_DOUBLE, rank + 1, 1,
                     &u[IDX(local_rows + 1, 0, cols)], cols, MPI_DOUBLE, rank + 1, 0,
                     MPI_COMM_WORLD, &status);
    }
}

/**
 * Perform Jacobi iteration on local grid
 * Returns local maximum change (delta)
 */
double jacobi_iteration_local(double* u_old, double* u_new, int local_rows, int n) {
    int cols = n + 2;
    double local_delta = 0.0;

    // Update only interior points (rows 1..local_rows, cols 1..n)
    for (int i = 1; i <= local_rows; i++) {
        for (int j = 1; j <= n; j++) {
            int idx = IDX(i, j, cols);
            double new_val = 0.25 * (u_old[IDX(i-1, j, cols)] +
                                      u_old[IDX(i+1, j, cols)] +
                                      u_old[IDX(i, j-1, cols)] +
                                      u_old[IDX(i, j+1, cols)]);
            u_new[idx] = new_val;

            double diff = fabs(new_val - u_old[idx]);
            if (diff > local_delta) {
                local_delta = diff;
            }
        }
    }

    return local_delta;
}

/**
 * Copy interior points from u_new to u_old (boundaries stay the same)
 */
void copy_interior(double* dst, double* src, int local_rows, int n) {
    int cols = n + 2;
    for (int i = 1; i <= local_rows; i++) {
        for (int j = 1; j <= n; j++) {
            dst[IDX(i, j, cols)] = src[IDX(i, j, cols)];
        }
    }
}

/**
 * Gather full solution to rank 0 for output
 */
void gather_solution(double* local_u, double* global_u, int local_rows, int n,
                     int* all_local_rows, int* displs, int rank, int size) {
    int cols = n + 2;

    // Each process sends its interior rows
    int send_count = local_rows * cols;

    // Gather all interior data
    int* recv_counts = NULL;
    int* recv_displs = NULL;

    if (rank == 0) {
        recv_counts = (int*)malloc(size * sizeof(int));
        recv_displs = (int*)malloc(size * sizeof(int));

        int offset = cols;  // Skip top boundary row in global
        for (int p = 0; p < size; p++) {
            recv_counts[p] = all_local_rows[p] * cols;
            recv_displs[p] = offset;
            offset += all_local_rows[p] * cols;
        }

        // Initialize global boundaries
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

    // Parse arguments (all processes)
    int n = (argc > 1) ? atoi(argv[1]) : DEFAULT_N;
    double tolerance = (argc > 2) ? atof(argv[2]) : DEFAULT_TOLERANCE;
    int max_iter = (argc > 3) ? atoi(argv[3]) : DEFAULT_MAX_ITER;

    if (rank == 0) {
        printf("Parallel Jacobi Iteration (MPI) for Laplace Equation\n");
        printf("Grid: %d x %d, Processors: %d\n", n, n, size);
        printf("Tolerance: %.2e, Max iterations: %d\n\n", tolerance, max_iter);
    }

    // Compute domain decomposition (1D row-wise)
    int base_rows = n / size;
    int extra = n % size;

    // Array to store how many rows each process owns
    int* all_local_rows = (int*)malloc(size * sizeof(int));
    int* displs = (int*)malloc(size * sizeof(int));

    int offset = 0;
    for (int p = 0; p < size; p++) {
        all_local_rows[p] = base_rows + (p < extra ? 1 : 0);
        displs[p] = offset;
        offset += all_local_rows[p];
    }

    int local_rows = all_local_rows[rank];
    int global_start_row = displs[rank] + 1;  // 1-indexed in global grid

    if (rank == 0) {
        printf("Domain decomposition:\n");
        for (int p = 0; p < size; p++) {
            printf("  Processor %d: %d rows (global rows %d-%d)\n",
                   p, all_local_rows[p],
                   displs[p] + 1, displs[p] + all_local_rows[p]);
        }
        printf("\n");
    }

    // Allocate local arrays (with 2 ghost rows)
    int cols = n + 2;
    int local_total_rows = local_rows + 2;  // +2 for ghost rows
    double* u_old = allocate_array(local_total_rows, cols);
    double* u_new = allocate_array(local_total_rows, cols);

    // Initialize
    initialize_local_grid(u_old, local_rows, n, global_start_row, rank, size);
    initialize_local_grid(u_new, local_rows, n, global_start_row, rank, size);

    // Main iteration loop
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    int iter;
    double global_delta;

    for (iter = 1; iter <= max_iter; iter++) {
        // Exchange ghost rows
        exchange_boundaries(u_old, local_rows, n, rank, size);

        // Jacobi iteration
        double local_delta = jacobi_iteration_local(u_old, u_new, local_rows, n);

        // Global reduction for convergence check
        MPI_Allreduce(&local_delta, &global_delta, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        // Check convergence
        if (global_delta < tolerance) {
            break;
        }

        // Copy new values to old
        copy_interior(u_old, u_new, local_rows, n);

        // Progress (only rank 0)
        if (rank == 0 && iter % 1000 == 0) {
            printf("Iteration %d: delta = %.2e\n", iter, global_delta);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();

    // Print results
    if (rank == 0) {
        printf("\n=== Parallel Jacobi (MPI) Results ===\n");
        printf("Grid size:        %d x %d\n", n, n);
        printf("Processors:       %d\n", size);
        printf("Iterations:       %d\n", iter);
        printf("Final delta:      %.2e\n", global_delta);
        printf("Elapsed time:     %.4f seconds\n", end_time - start_time);
        printf("Time/iteration:   %.6f ms\n", ((end_time - start_time) / iter) * 1000);
        printf("=====================================\n");
    }

    // Gather and save solution
    double* global_u = NULL;
    if (rank == 0) {
        global_u = allocate_array(n + 2, cols);
    }

    // Copy final values to u_old for gathering
    copy_interior(u_old, u_new, local_rows, n);
    gather_solution(u_old, global_u, local_rows, n, all_local_rows, displs, rank, size);

    if (rank == 0) {
        FILE* fp = fopen("jacobi_mpi_solution.dat", "w");
        if (fp) {
            fprintf(fp, "# Parallel Jacobi MPI Solution\n");
            fprintf(fp, "# Grid: %d x %d, Processors: %d\n", n, n, size);
            for (int i = 0; i < n + 2; i++) {
                for (int j = 0; j < cols; j++) {
                    fprintf(fp, "%d %d %.10f\n", i, j, global_u[IDX(i, j, cols)]);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
            printf("Solution saved to jacobi_mpi_solution.dat\n");
        }
        free(global_u);
    }

    // Cleanup
    free(u_old);
    free(u_new);
    free(all_local_rows);
    free(displs);

    MPI_Finalize();
    return 0;
}
