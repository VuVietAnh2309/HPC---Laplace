#ifndef LAPLACE_COMMON_H
#define LAPLACE_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define DEFAULT_N 100
#define DEFAULT_MAX_ITER 100000
#define DEFAULT_TOL 1e-6
#define BC_TOP 0.0
#define BC_BOTTOM 100.0
#define BC_LEFT 0.0
#define BC_RIGHT 100.0
#define INIT_VAL 50.0

static inline double *alloc_grid(int n)
{
    int size = n + 2;
    double *u = (double *)malloc(size * size * sizeof(double));
    if (!u)
    {
        fprintf(stderr, "Err: malloc failed\n");
        exit(1);
    }
    return u;
}

static inline void free_grid(double *u)
{
    free(u);
}

static inline void init_grid(double *u, int n)
{
    int size = n + 2;
    for (int row = 0; row < size; row++)
    {
        for (int col = 0; col < size; col++)
        {
            u[row * size + col] = INIT_VAL;
        }
    }
    for (int col = 0; col < size; col++)
    {
        u[0 * size + col] = BC_TOP;
        u[(n + 1) * size + col] = BC_BOTTOM;
    }
    for (int row = 0; row < size; row++)
    {
        u[row * size + 0] = BC_LEFT;
        u[row * size + (n + 1)] = BC_RIGHT;
    }
}

static inline void copy_grid(double *u_new, double *u_old, int n)
{
    int size = n + 2;
    memcpy(u_new, u_old, size * size * sizeof(double));
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
    printf("Grid: %d x %d, Iterations: %d, Final delta: %.2e, Time: %.4f sec\n", n, n, iter, delta, t);
}

static inline double get_time()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

#endif