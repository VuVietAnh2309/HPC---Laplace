# Giải Phương Trình Laplace - Tính Toán Hiệu Năng Cao

Dự án implement các phương pháp số giải phương trình Laplace cho môn học **Tính toán Hiệu năng Cao** (High Performance Computing) trong chương trình Thạc sĩ.

## Cấu trúc dự án

```
HPC/
├── docs/                          # Tài liệu phân tích
│   ├── 01-problem-analysis.md     # Phân tích bài toán
│   ├── 02-numerical-methods.md    # Các phương pháp số
│   └── 03-parallelization.md      # Chiến lược song song hóa
├── src/
│   ├── sequential/                # Code tuần tự
│   │   ├── laplace_common.h       # Header chung
│   │   ├── jacobi.c               # Jacobi iteration
│   │   ├── gauss_seidel.c         # Gauss-Seidel iteration
│   │   └── sor.c                  # SOR với Red-Black ordering
│   └── parallel/                  # Code song song (MPI)
│       ├── jacobi_mpi.c           # Parallel Jacobi
│       └── sor_mpi.c              # Parallel SOR Red-Black
├── paper/                         # Tài liệu tham khảo (PDF)
├── bin/                           # Executables (sau khi build)
├── Makefile                       # Build system
└── README.md                      # File này
```

## Yêu cầu hệ thống

- **GCC** (hoặc compiler C tương đương)
- **MPI** (OpenMPI hoặc MPICH) cho phiên bản song song
- **Make**

### Cài đặt trên macOS
```bash
# Xcode Command Line Tools
xcode-select --install

# OpenMPI via Homebrew
brew install open-mpi
```

### Cài đặt trên Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install build-essential openmpi-bin libopenmpi-dev
```

## Build

```bash
# Build tất cả (sequential + parallel)
make all

# Chỉ build sequential
make sequential

# Chỉ build parallel (cần MPI)
make parallel

# Clean
make clean
```

## Chạy chương trình

### Phiên bản Sequential

```bash
# Jacobi
./bin/jacobi [grid_size] [tolerance] [max_iterations]
./bin/jacobi 100 1e-6 50000

# Gauss-Seidel
./bin/gauss_seidel [grid_size] [tolerance] [max_iterations]
./bin/gauss_seidel 100 1e-6 50000

# SOR (Successive Over-Relaxation)
./bin/sor [grid_size] [tolerance] [max_iterations] [use_redblack] [omega]
./bin/sor 100 1e-6 10000 1        # Với red-black ordering, omega tự động tối ưu
./bin/sor 100 1e-6 10000 1 1.9    # Chỉ định omega = 1.9
```

### Phiên bản Parallel (MPI)

```bash
# Parallel Jacobi
mpirun -np [num_processes] ./bin/jacobi_mpi [grid_size] [tolerance] [max_iterations]
mpirun -np 4 ./bin/jacobi_mpi 200 1e-6 50000

# Parallel SOR với Red-Black ordering
mpirun -np [num_processes] ./bin/sor_mpi [grid_size] [tolerance] [max_iterations] [omega]
mpirun -np 4 ./bin/sor_mpi 200 1e-6 10000
```

## So sánh các phương pháp

```bash
# So sánh tốc độ hội tụ
make compare

# Test khả năng mở rộng (scalability)
make scalability
```

## Bài toán mẫu

Giải phương trình Laplace trên miền vuông [0,1] × [0,1] với điều kiện biên:
- **Biên trên**: u = 0
- **Biên dưới**: u = 100
- **Biên trái**: u = 0
- **Biên phải**: u = 100

## Kết quả mong đợi

### So sánh số vòng lặp (grid 50×50, tolerance 10⁻⁶)

| Phương pháp | Iterations | Thời gian (tương đối) |
|-------------|------------|----------------------|
| Jacobi | ~15,000 | 1.0x |
| Gauss-Seidel | ~7,500 | 0.5x |
| SOR (ω ≈ 1.87) | ~300 | 0.02x |

### Scalability test (Parallel Jacobi, grid 100×100)

| Processors | Speedup | Efficiency |
|------------|---------|------------|
| 1 | 1.0 | 100% |
| 2 | ~1.8 | ~90% |
| 4 | ~3.2 | ~80% |
| 8 | ~5.5 | ~69% |

## Output files

Sau khi chạy, chương trình tạo ra các file `.dat` chứa nghiệm:
- `jacobi_solution.dat`
- `gauss_seidel_solution.dat`
- `sor_solution.dat`
- `jacobi_mpi_solution.dat`
- `sor_mpi_solution.dat`

### Visualize với Gnuplot

```gnuplot
# Vẽ heatmap
set pm3d map
splot 'sor_solution.dat' using 1:2:3 with pm3d title 'Temperature'

# Hoặc contour plot
set contour base
set cntrparam levels 20
splot 'sor_solution.dat' using 1:2:3 with lines title 'Temperature'
```

## Tài liệu tham khảo

1. **Paper chính**: `paper/Laplace-Equations.pdf` - Numerical Methods lecture notes
2. **Paper tham khảo**: `paper/Numerical Solution of Laplaces Equation.pdf` - Per Brinch Hansen, 1992
3. Press et al., "Numerical Recipes in C", Chapter 19 (Relaxation Methods)
4. David Young, "Iterative methods for solving partial difference equations", 1954

## Thuật toán chính

### Jacobi Iteration
```
u_new[i][j] = 0.25 * (u_old[i-1][j] + u_old[i+1][j] + u_old[i][j-1] + u_old[i][j+1])
```

### Gauss-Seidel Iteration
```
u[i][j] = 0.25 * (u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1])
# Sử dụng giá trị mới ngay khi có
```

### SOR (Successive Over-Relaxation)
```
aver = 0.25 * (u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1])
res = aver - u[i][j]
u[i][j] = u[i][j] + omega * res

# Optimal omega: ω = 2 - 2π/n
```

### Red-Black Ordering (cho song song hóa)
```
Phase 1: Cập nhật tất cả điểm RED (i+j chẵn)
Phase 2: Cập nhật tất cả điểm BLACK (i+j lẻ)
# Mỗi phase có thể thực hiện song song hoàn toàn
```

## License

Educational use only - Master's Course Project.
