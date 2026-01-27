/*
 * Parallel Jacobi Iteration - MPI
 * Solves 2D Laplace equation with 1D domain decomposition
 *
 * Compile: mpicc -O3 -o jacobi_mpi jacobi_mpi.c -lm
 * Run:     mpirun -np 4 ./jacobi_mpi 100 1e-6 10000
 */

#include "laplace_common_mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>

#define DEFAULT_N 100
#define DEFAULT_MAX_ITER 100000
#define DEFAULT_TOL 1e-6
#define BC_TOP 0.0
#define BC_BOTTOM 100.0
#define BC_LEFT 0.0
#define BC_RIGHT 100.0
#define INIT_VAL 50.0

int main(int argc, char *argv[])
{
    int rank, nprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int n = (argc > 1) ? atoi(argv[1]) : DEFAULT_N;
    double tol = (argc > 2) ? atof(argv[2]) : DEFAULT_TOL;
    int max_iter = (argc > 3) ? atoi(argv[3]) : DEFAULT_MAX_ITER;

    int base = n / nprocs;
    int extra = n % nprocs;
    int nrows = base + (rank < extra ? 1 : 0);
    int size = n + 2;

    double *u = alloc_grid(nrows + 2, size);
    double *new_u = alloc_grid(nrows + 2, size);
    init_grid(u, nrows, n, rank, nprocs);
    init_grid(new_u, nrows, n, rank, nprocs);

    if (rank == 0)
    {
        printf("Jacobi MPI: %dx%d grid, %d procs, tol=%.1e\n\n", n, n, nprocs, tol);
    }

    double t0 = MPI_Wtime();
    double delta = 0.0;
    int iter;
    for (iter = 1; iter <= max_iter; iter++)
    {
        exchange_rows(u, nrows, n, rank, nprocs);
        double local_max_diff = 0.0;
        for (int i = 1; i <= nrows; i++)
        {
            int curr = i * size;
            int above = (i - 1) * size;
            int below = (i + 1) * size;
            for (int j = 1; j <= n; j++)
            {
                double val = 0.25 * (u[above + j] + u[below + j] +
                                     u[curr + j - 1] + u[curr + j + 1]);
                new_u[curr + j] = val;
                double diff = fabs(val - u[curr + j]);
                if (diff > local_max_diff)
                    local_max_diff = diff;
            }
        }
        MPI_Allreduce(&local_max_diff, &delta, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if (delta < tol)
        {
            break;
        }
        double *tmp = u;
        u = new_u;
        new_u = tmp;
        if (rank == 0 && iter % 1000 == 0)
        {
            printf("iter %d: delta=%.2e\n", iter, delta);
        }
    }

    double t1 = MPI_Wtime();
    if (rank == 0)
    {
        print_result("MPI Jacobi", n, iter, delta, t1 - t0);
    }

    double *global_grid = NULL;
    int *recv_counts = NULL;
    int *displs = NULL;
    if (rank == 0)
    {
        global_grid = alloc_grid(n + 2, size);
        recv_counts = malloc(nprocs * sizeof(int));
        displs = malloc(nprocs * sizeof(int));
        int current_displ = size;
        for (int p = 0; p < nprocs; p++)
        {
            int p_nrows = base + (p < extra ? 1 : 0);
            recv_counts[p] = p_nrows * size;
            displs[p] = current_displ;
            current_displ += recv_counts[p];
        }
        for (int col = 0; col < size; col++)
        {
            global_grid[0 * size + col] = BC_TOP;
            global_grid[(n + 1) * size + col] = BC_BOTTOM;
        }
        for (int row = 0; row < n + 2; row++)
        {
            global_grid[row * size + 0] = BC_LEFT;
            global_grid[row * size + (n + 1)] = BC_RIGHT;
        }
    }
    MPI_Gatherv(&u[1 * size], nrows * size, MPI_DOUBLE,
                global_grid, recv_counts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        print_grid(global_grid, n, "Final Solution");
        free(global_grid);
        free(recv_counts);
        free(displs);
    }
    free_grid(u);
    free_grid(new_u);
    MPI_Finalize();
    return 0;
}