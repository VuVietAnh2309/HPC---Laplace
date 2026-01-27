/*
 * laplace_common.h - Unified utilities for Laplace solvers
 */

#ifndef LAPLACE_COMMON_H
#define LAPLACE_COMMON_H

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

static inline double *alloc_grid(int rows, int cols)
{
    double *u = (double *)malloc(rows * cols * sizeof(double));
    if (!u)
    {
        fprintf(stderr, "Err: malloc failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return u;
}

static inline void free_grid(double *u)
{
    free(u);
}

static inline void init_grid(double *u, int nrows, int n, int rank, int nprocs)
{
    int size = n + 2;
    for (int i = 0; i < nrows + 2; i++)
    {
        for (int j = 0; j < size; j++)
        {
            u[i * size + j] = INIT_VAL;
        }
    }
    for (int i = 0; i < nrows + 2; i++)
    {
        u[i * size + 0] = BC_LEFT;
        u[i * size + (n + 1)] = BC_RIGHT;
    }
    if (rank == 0)
    {
        for (int j = 0; j < size; j++)
            u[0 * size + j] = BC_TOP;
    }
    if (rank == nprocs - 1)
    {
        for (int j = 0; j < size; j++)
            u[(nrows + 1) * size + j] = BC_BOTTOM;
    }
}

static inline void exchange_rows(double *u, int nrows, int n, int rank, int nprocs)
{
    int size = n + 2;
    MPI_Status st;
    if (rank > 0)
        MPI_Sendrecv(&u[1 * size], size, MPI_DOUBLE, rank - 1, 0,
                     &u[0 * size], size, MPI_DOUBLE, rank - 1, 1,
                     MPI_COMM_WORLD, &st);
    if (rank < nprocs - 1)
        MPI_Sendrecv(&u[nrows * size], size, MPI_DOUBLE, rank + 1, 1,
                     &u[(nrows + 1) * size], size, MPI_DOUBLE, rank + 1, 0,
                     MPI_COMM_WORLD, &st);
}

static inline void print_grid(double *u, int n, const char *label)
{
    int size = n + 2;
    printf("\n%s (%d x %d):\n", label, size, size);
    if (n <= 20)
    {
        for (int row = 0; row < size; row++)
        {
            for (int col = 0; col < size; col++)
                printf("%7.2f ", u[row * size + col]);
            printf("\n");
        }
    }
    else
    {
        printf("(Corner-only because of n > 20)\n");
        printf("Top-left:\n");
        for (int row = 0; row < 5; row++)
        {
            for (int col = 0; col < 5; col++)
            {
                printf("%7.2f ", u[row * size + col]);
            }
            printf("\n");
        }
        printf("Bottom-right:\n");
        for (int row = size - 5; row < size; row++)
        {
            for (int col = size - 5; col < size; col++)
            {
                printf("%7.2f ", u[row * size + col]);
            }
            printf("\n");
        }
    }
}

static inline void print_result(const char *method, int n, int iter, double delta, double t)
{
    printf("\n=== %s Results ===\n", method);
    printf("Grid: %d x %d, Iterations: %d, Final delta: %.2e, Time: %.4f sec\n",
           n, n, iter, delta, t);
}

#endif