# Giải Phương Trình Laplace - Tính Toán Hiệu Năng Cao

Dự án implement phương pháp SOR (Successive Over-Relaxation) với Red-Black ordering để giải phương trình Laplace cho môn học **Tính toán Hiệu năng Cao** (High Performance Computing).

## Cấu trúc dự án
```
HPC/
├── docs/                          # Tài liệu phân tích
│   ├── 01-problem-analysis.md     # Phân tích bài toán
│   ├── 02-numerical-methods.md    # Các phương pháp số
│   └── 03-parallelization.md      # Chiến lược song song hóa
├── src/                           # Source code
│   ├── sequential/                # Code tuần tự
│   │   ├── laplace_common.h       # Header chung
│   │   └── sor.c                  # SOR với Red-Black ordering
│   └── parallel/                  # Code song song (MPI)
│       ├── laplace_common_mpi.h   # Header chung MPI
│       └── sor_mpi.c              # Parallel SOR Red-Black
├── solution/                     # Nghiệm số và visualization
│   ├── sor_solution.dat           # Sequential solution
│   ├── sor_mpi_solution.dat       # Parallel solution
│   ├── heatmap.png                # Heatmap visualization
│   ├── contour.png                # Contour plot
│   └── visualize.gp               # Gnuplot script
├── paper/                         # Tài liệu tham khảo (PDF)
├── bin/                           # Executables (sau khi build)
├── Makefile                       # Build system
└── README.md                      # File này
```

## Yêu cầu hệ thống

- **GCC** (hoặc compiler C tương đương)
- **MPI** (OpenMPI hoặc MPICH) cho phiên bản song song
- **Make**
- **Gnuplot** (để visualize kết quả)

### Cài đặt trên macOS
```bash
xcode-select --install
brew install open-mpi gnuplot
```

### Cài đặt trên Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install build-essential openmpi-bin libopenmpi-dev gnuplot
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
./bin/sor [grid_size] [tolerance] [max_iterations] [use_redblack] [omega]

# Ví dụ
./bin/sor 100 1e-6 10000 1        # Red-black ordering, omega tự động tối ưu
./bin/sor 100 1e-6 10000 1 1.9    # Chỉ định omega = 1.9
```

### Phiên bản Parallel (MPI)
```bash
mpirun -np [num_processes] ./bin/sor_mpi [grid_size] [tolerance] [max_iterations] [omega]

# Ví dụ
mpirun -np 4 ./bin/sor_mpi 200 1e-6 10000
```

## Chạy test và benchmark
```bash
# Test sequential
make test-sequential

# Test parallel
make test-parallel

# Test khả năng mở rộng (scalability)
make scalability
```

## Bài toán

Giải phương trình Laplace trên miền vuông [0,1] × [0,1] với điều kiện biên:
- **Biên trên**: u = 0
- **Biên dưới**: u = 100
- **Biên trái**: u = 0
- **Biên phải**: u = 100

## Visualize kết quả

Sau khi chạy, nghiệm được lưu vào thư mục `solution/`:
- `sor_solution.dat` - nghiệm sequential
- `sor_mpi_solution.dat` - nghiệm parallel

### Tạo visualization
```bash
make visualize
```

Kết quả: `heatmap.png`, `contour.png` trong thư mục `solution/`

## Thuật toán SOR với Red-Black Ordering

### SOR (Successive Over-Relaxation)
```c
aver = 0.25 * (u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1])
res = aver - u[i][j]
u[i][j] = u[i][j] + omega * res

// Optimal omega: ω = 2 - 2π/n
```

### Red-Black Ordering (cho song song hóa)
```
Phase 1: Cập nhật tất cả điểm RED (i+j chẵn)
Phase 2: Cập nhật tất cả điểm BLACK (i+j lẻ)
# Mỗi phase có thể thực hiện song song hoàn toàn
```

## Tài liệu tham khảo

1. **Paper chính**: `paper/Laplace-Equations.pdf` - Numerical Methods lecture notes
2. **Paper tham khảo**: `paper/Numerical Solution of Laplaces Equation.pdf` - Per Brinch Hansen, 1992
3. Press et al., "Numerical Recipes in C", Chapter 19 (Relaxation Methods)
4. David Young, "Iterative methods for solving partial difference equations", 1954