/*
 * SOR (Successive Over-Relaxation) for Laplace Equation
 * Compile: gcc -O3 -o sor sor.c -lm
 * Run:     ./sor 100 1e-6 10000 [redblack=1] [omega]
 */

#include "laplace_common.h"

double sor_step_rowwise(double *u, int n, double omega)
{
    int size = n + 2;
    double max_diff = 0.0;
    for (int row = 1; row <= n; row++)
    {
        int curr = row * size;
        int above = (row - 1) * size;
        int below = (row + 1) * size;
        for (int col = 1; col <= n; col++)
        {
            double old = u[curr + col];
            double avg = 0.25 * (u[above + col] + u[below + col] +
                                 u[curr + col - 1] + u[curr + col + 1]);
            u[curr + col] = old + omega * (avg - old);
            double diff = fabs(u[curr + col] - old);
            if (diff > max_diff)
            {
                max_diff = diff;
            }
        }
    }
    return max_diff;
}

double sor_step_redblack(double *u, int n, double omega)
{
    int size = n + 2;
    double max_diff = 0.0;

    // 0 for red, 1 for black
    for (int color = 0; color < 2; color++)
    {
        for (int row = 1; row <= n; row++)
        {
            int col_start = (row % 2 == (color == 0 ? 1 : 0)) ? 1 : 2;
            int curr = row * size;
            int above = (row - 1) * size;
            int below = (row + 1) * size;
            for (int col = col_start; col <= n; col += 2)
            {
                double old = u[curr + col];
                double avg = 0.25 * (u[above + col] + u[below + col] +
                                     u[curr + col - 1] + u[curr + col + 1]);
                u[curr + col] = old + omega * (avg - old);
                double diff = fabs(u[curr + col] - old);
                if (diff > max_diff)
                {
                    max_diff = diff;
                }
            }
        }
    }
    return max_diff;
}

int main(int argc, char *argv[])
{
    int n = (argc > 1) ? atoi(argv[1]) : DEFAULT_N;
    double tol = (argc > 2) ? atof(argv[2]) : DEFAULT_TOL;
    int max_iter = (argc > 3) ? atoi(argv[3]) : DEFAULT_MAX_ITER;
    int redblack = (argc > 4) ? atoi(argv[4]) : 1;
    double omega = (argc > 5) ? atof(argv[5]) : (2.0 - 2.0 * M_PI / n);

    printf("SOR: %dx%d grid, omega=%.4f, mode=%s\n", n, n, omega,
           redblack ? "red-black" : "row-wise");

    double *u = alloc_grid(n);
    init_grid(u, n);

    double t0 = get_time();
    double delta;
    int iter;
    for (iter = 1; iter <= max_iter; iter++)
    {
        delta = redblack ? sor_step_redblack(u, n, omega)
                         : sor_step_rowwise(u, n, omega);
        if (delta < tol)
        {
            break;
        }
        if (iter % 1000 == 0)
        {
            printf("iter %d: delta=%.2e\n", iter, delta);
        }
    }

    double t1 = get_time();
    print_result("SOR", n, iter, delta, t1 - t0);
    print_grid(u, n, "Final Solution");

    FILE *fp = fopen("solution/sor_solution.dat", "w");
    if (fp)
    {
        int size = n + 2;
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                fprintf(fp, "%d %d %.6f\n", i, j, u[i * size + j]);
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
        printf("Solution written to sor_solution.dat\n");
    }

    free_grid(u);
    return 0;
}