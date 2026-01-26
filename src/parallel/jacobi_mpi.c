/*
 * Parallel Jacobi Iteration - MPI
 * Solves 2D Laplace equation with 1D domain decomposition
 *
 * Compile: mpicc -O3 -o jacobi_mpi jacobi_mpi.c -lm
 * Run:     mpirun -np 4 ./jacobi_mpi 100 1e-6 10000
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>

#define DEFAULT_N 100
#define DEFAULT_MAX_ITER 100000
#define DEFAULT_TOL 1e-6
#define IDX(i, j, cols) ((i) * (cols) + (j))

double *alloc_grid(int rows, int cols) {
    double *arr = calloc(rows * cols, sizeof(double));
    if (!arr) {
        fprintf(stderr, "malloc failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return arr;
}

void init_grid(double *u, int nrows, int n, int rank, int nprocs) {
    int cols = n + 2;

    for (int i = 0; i < nrows + 2; i++)
        for (int j = 0; j < cols; j++)
            u[IDX(i, j, cols)] = 50.0;

    // boundaries: left=0, right=100, top=0, bottom=100
    for (int i = 0; i < nrows + 2; i++) {
        u[IDX(i, 0, cols)] = 0.0;
        u[IDX(i, n + 1, cols)] = 100.0;
    }
    if (rank == 0)
        for (int j = 0; j < cols; j++)
            u[IDX(0, j, cols)] = 0.0;
    if (rank == nprocs - 1)
        for (int j = 0; j < cols; j++)
            u[IDX(nrows + 1, j, cols)] = 100.0;
}

void exchange_ghost(double *u, int nrows, int n, int rank, int nprocs) {
    int cols = n + 2;
    MPI_Status st;

    if (rank > 0)
        MPI_Sendrecv(&u[IDX(1, 0, cols)], cols, MPI_DOUBLE, rank - 1, 0,
                     &u[IDX(0, 0, cols)], cols, MPI_DOUBLE, rank - 1, 1,
                     MPI_COMM_WORLD, &st);
    if (rank < nprocs - 1)
        MPI_Sendrecv(&u[IDX(nrows, 0, cols)], cols, MPI_DOUBLE, rank + 1, 1,
                     &u[IDX(nrows + 1, 0, cols)], cols, MPI_DOUBLE, rank + 1, 0,
                     MPI_COMM_WORLD, &st);
}

double jacobi_step(double *u_old, double *u_new, int nrows, int n) {
    int cols = n + 2;
    double maxdiff = 0.0;

    for (int i = 1; i <= nrows; i++) {
        for (int j = 1; j <= n; j++) {
            int idx = IDX(i, j, cols);
            double val = 0.25 * (u_old[IDX(i-1, j, cols)] + u_old[IDX(i+1, j, cols)] +
                                 u_old[IDX(i, j-1, cols)] + u_old[IDX(i, j+1, cols)]);
            u_new[idx] = val;
            double diff = fabs(val - u_old[idx]);
            if (diff > maxdiff) maxdiff = diff;
        }
    }
    return maxdiff;
}

void print_matrix(double *u, int rows, int cols, const char *label) {
    printf("\n%s (%d x %d):\n", label, rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            printf("%7.2f ", u[IDX(i, j, cols)]);
        printf("\n");
    }
}

int main(int argc, char *argv[]) {
    int rank, nprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int n = (argc > 1) ? atoi(argv[1]) : DEFAULT_N;
    double tol = (argc > 2) ? atof(argv[2]) : DEFAULT_TOL;
    int maxiter = (argc > 3) ? atoi(argv[3]) : DEFAULT_MAX_ITER;

    if (rank == 0)
        printf("Jacobi MPI: %dx%d grid, %d procs, tol=%.1e\n\n", n, n, nprocs, tol);

    // domain decomposition
    int base = n / nprocs;
    int extra = n % nprocs;
    int *rowcnt = malloc(nprocs * sizeof(int));
    int *offset = malloc(nprocs * sizeof(int));

    int off = 0;
    for (int p = 0; p < nprocs; p++) {
        rowcnt[p] = base + (p < extra ? 1 : 0);
        offset[p] = off;
        off += rowcnt[p];
    }
    int nrows = rowcnt[rank];

    if (rank == 0) {
        for (int p = 0; p < nprocs; p++)
            printf("  proc %d: %d rows [%d-%d]\n", p, rowcnt[p], offset[p]+1, offset[p]+rowcnt[p]);
        printf("\n");
    }

    int cols = n + 2;
    double *u_old = alloc_grid(nrows + 2, cols);
    double *u_new = alloc_grid(nrows + 2, cols);

    init_grid(u_old, nrows, n, rank, nprocs);
    init_grid(u_new, nrows, n, rank, nprocs);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    int iter;
    double delta;
    for (iter = 1; iter <= maxiter; iter++) {
        exchange_ghost(u_old, nrows, n, rank, nprocs);
        double local_delta = jacobi_step(u_old, u_new, nrows, n);
        MPI_Allreduce(&local_delta, &delta, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if (delta < tol) break;

        // swap pointers
        double *tmp = u_old; u_old = u_new; u_new = tmp;

        if (rank == 0 && iter % 1000 == 0)
            printf("iter %d: delta=%.2e\n", iter, delta);
    }

    double t1 = MPI_Wtime();

    if (rank == 0) {
        printf("\n=== Results ===\n");
        printf("Iterations: %d\n", iter);
        printf("Final delta: %.2e\n", delta);
        printf("Time: %.4f sec\n", t1 - t0);
    }

    // gather solution to rank 0
    double *global = NULL;
    if (rank == 0)
        global = alloc_grid(n + 2, cols);

    int *rcnt = NULL, *rdisp = NULL;
    if (rank == 0) {
        rcnt = malloc(nprocs * sizeof(int));
        rdisp = malloc(nprocs * sizeof(int));
        int d = cols;
        for (int p = 0; p < nprocs; p++) {
            rcnt[p] = rowcnt[p] * cols;
            rdisp[p] = d;
            d += rcnt[p];
        }
        // set boundaries
        for (int j = 0; j < cols; j++) {
            global[IDX(0, j, cols)] = 0.0;
            global[IDX(n+1, j, cols)] = 100.0;
        }
        for (int i = 0; i < n + 2; i++) {
            global[IDX(i, 0, cols)] = 0.0;
            global[IDX(i, n+1, cols)] = 100.0;
        }
    }

    MPI_Gatherv(&u_new[IDX(1, 0, cols)], nrows * cols, MPI_DOUBLE,
                global, rcnt, rdisp, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // print final matrix
    if (rank == 0) {
        if (n <= 20) {
            print_matrix(global, n + 2, cols, "Final Solution");
        } else {
            printf("\n(Matrix too large to display, showing corners)\n");
            printf("Top-left 5x5:\n");
            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 5; j++)
                    printf("%7.2f ", global[IDX(i, j, cols)]);
                printf("\n");
            }
            printf("\nBottom-right 5x5:\n");
            for (int i = n - 3; i < n + 2; i++) {
                for (int j = n - 3; j < n + 2; j++)
                    printf("%7.2f ", global[IDX(i, j, cols)]);
                printf("\n");
            }
        }
        free(global);
        free(rcnt);
        free(rdisp);
    }

    free(u_old);
    free(u_new);
    free(rowcnt);
    free(offset);
    MPI_Finalize();
    return 0;
}
