/*
 * Common utilities for Laplace solvers
 */

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

// boundaries: top=0, bottom=100, left=0, right=100
#define BC_TOP    0.0
#define BC_BOTTOM 100.0
#define BC_LEFT   0.0
#define BC_RIGHT  100.0
#define INIT_VAL  50.0

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static inline double** alloc_grid(int n) {
    double **g = malloc((n + 2) * sizeof(double*));
    if (!g) { fprintf(stderr, "malloc failed\n"); exit(1); }
    for (int i = 0; i < n + 2; i++) {
        g[i] = malloc((n + 2) * sizeof(double));
        if (!g[i]) { fprintf(stderr, "malloc failed\n"); exit(1); }
    }
    return g;
}

static inline void free_grid(double **g, int n) {
    for (int i = 0; i < n + 2; i++) free(g[i]);
    free(g);
}

static inline void init_grid(double **u, int n) {
    for (int i = 0; i < n + 2; i++)
        for (int j = 0; j < n + 2; j++)
            u[i][j] = INIT_VAL;

    for (int j = 0; j < n + 2; j++) {
        u[0][j] = BC_TOP;
        u[n + 1][j] = BC_BOTTOM;
    }
    for (int i = 0; i < n + 2; i++) {
        u[i][0] = BC_LEFT;
        u[i][n + 1] = BC_RIGHT;
    }
}

static inline void copy_grid(double **dst, double **src, int n) {
    for (int i = 0; i < n + 2; i++)
        memcpy(dst[i], src[i], (n + 2) * sizeof(double));
}

static inline void print_matrix(double **u, int n, const char *label) {
    int size = n + 2;
    printf("\n%s (%d x %d):\n", label, size, size);

    if (n <= 20) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++)
                printf("%7.2f ", u[i][j]);
            printf("\n");
        }
    } else {
        printf("(Matrix too large, showing corners)\n");
        printf("Top-left 5x5:\n");
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++)
                printf("%7.2f ", u[i][j]);
            printf("\n");
        }
        printf("\nBottom-right 5x5:\n");
        for (int i = n - 3; i < size; i++) {
            for (int j = n - 3; j < size; j++)
                printf("%7.2f ", u[i][j]);
            printf("\n");
        }
    }
}

static inline void print_result(const char *method, int n, int iter, double delta, double t) {
    printf("\n=== %s Results ===\n", method);
    printf("Grid: %d x %d\n", n, n);
    printf("Iterations: %d\n", iter);
    printf("Final delta: %.2e\n", delta);
    printf("Time: %.4f sec\n", t);
}

static inline double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

#endif
