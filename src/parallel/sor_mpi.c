/*
 * Parallel SOR (Red-Black) - MPI
 * Solves 2D Laplace equation with 1D domain decomposition
 *
 * Compile: mpicc -O3 -o sor_mpi sor_mpi.c -lm
 * Run:     mpirun -np 4 ./sor_mpi 100 1e-6 10000
 */

#include "laplace_common_mpi.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// update points where (global_i + j) % 2 == parity
static inline double sor_sweep(double *u, int nrows, int n, int gstart, double omega, int parity)
{
    int size = n + 2;
    double maxdiff = 0.0;

    for (int i = 1; i <= nrows; i++)
    {
        int gi = gstart + i - 1; // Global row index
        int row_off = i * size;
        for (int j = 1; j <= n; j++)
        {
            if ((gi + j) % 2 != parity)
                continue;

            int idx = row_off + j;
            double old_val = u[idx];
            double avg = 0.25 * (u[idx - size] + u[idx + size] + u[idx - 1] + u[idx + 1]);
            u[idx] = old_val + omega * (avg - old_val);

            double diff = fabs(u[idx] - old_val);
            if (diff > maxdiff)
                maxdiff = diff;
        }
    }
    return maxdiff;
}

int main(int argc, char *argv[])
{
    int rank, nprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int n = (argc > 1) ? atoi(argv[1]) : DEFAULT_N;
    double tol = (argc > 2) ? atof(argv[2]) : DEFAULT_TOL;
    int max_iter = (argc > 3) ? atoi(argv[3]) : DEFAULT_MAX_ITER;
    double omega = (argc > 4) ? atof(argv[4]) : 2.0 - 2.0 * M_PI / n;

    if (rank == 0)
    {
        printf("SOR MPI (Red-Black): %dx%d grid, %d procs, omega=%.4f\n", n, n, nprocs, omega);
    }

    int base = n / nprocs;
    int extra = n % nprocs;
    int nrows = base + (rank < extra ? 1 : 0);
    int gstart = 1;
    for (int p = 0; p < rank; p++)
    {
        gstart += (base + (p < extra ? 1 : 0));
    }
    int size = n + 2;
    double *u = alloc_grid(nrows + 2, size);
    init_grid(u, nrows, n, rank, nprocs);

    double t0 = MPI_Wtime();
    double delta;
    int iter;
    for (iter = 1; iter <= max_iter; iter++)
    {
        exchange_rows(u, nrows, n, rank, nprocs);
        double d1 = sor_sweep(u, nrows, n, gstart, omega, 0);
        exchange_rows(u, nrows, n, rank, nprocs);
        double d2 = sor_sweep(u, nrows, n, gstart, omega, 1);
        double local_delta = (d1 > d2) ? d1 : d2;
        MPI_Allreduce(&local_delta, &delta, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (delta < tol)
        {
            break;
        }
        if (rank == 0 && iter % 100 == 0)
        {
            {
                printf("iter %d: delta=%.2e\n", iter, delta);
            }
        }
    }

    double t1 = MPI_Wtime();
    if (rank == 0)
    {
        print_result("MPI SOR", n, iter, delta, t1 - t0);
    }

    double *global_grid = NULL;
    int *rcnts = NULL, *displs = NULL;
    if (rank == 0)
    {
        global_grid = alloc_grid(n + 2, size);
        rcnts = malloc(nprocs * sizeof(int));
        displs = malloc(nprocs * sizeof(int));
        int curr_displ = size;
        for (int p = 0; p < nprocs; p++)
        {
            int p_nrows = base + (p < extra ? 1 : 0);
            rcnts[p] = p_nrows * size;
            displs[p] = curr_displ;
            curr_displ += rcnts[p];
        }
        for (int j = 0; j < size; j++)
        {
            global_grid[j] = BC_TOP;
            global_grid[(n + 1) * size + j] = BC_BOTTOM;
        }
    }
    MPI_Gatherv(&u[size], nrows * size, MPI_DOUBLE,
                global_grid, rcnts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        print_grid(global_grid, n, "Final Solution");

        // FILE *fp = fopen("solution/sor_mpi_solution.dat", "w");
        // if (fp)
        // {
        //     for (int i = 0; i < size; i++)
        //     {
        //         for (int j = 0; j < size; j++)
        //         {
        //             fprintf(fp, "%d %d %.6f\n", i, j, global_grid[i * size + j]);
        //         }
        //         fprintf(fp, "\n");
        //     }
        //     fclose(fp);
        //     printf("Solution written to sor_mpi_solution.dat\n");
        // }

        free(global_grid);
        free(rcnts);
        free(displs);
    }
    free_grid(u);
    MPI_Finalize();
    return 0;
}