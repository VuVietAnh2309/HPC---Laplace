# Song Song Hóa Giải Phương Trình Laplace

## 1. Tại sao cần Song song hóa?

### 1.1 Giới hạn của tính toán tuần tự
Với lưới n×n:
- Jacobi: O(n⁴) operations tổng
- Mỗi iteration: O(n²) operations
- Với n = 1000, cần ~10¹² operations → không khả thi với máy đơn

### 1.2 Đặc tính song song tự nhiên
Phương trình Laplace có **data locality**:
- Mỗi điểm chỉ phụ thuộc 4 láng giềng trực tiếp
- Các điểm không liền kề có thể tính độc lập
- Phù hợp với mô hình **SPMD** (Single Program Multiple Data)

## 2. Domain Decomposition

### 2.1 Ý tưởng
Chia miền tính toán thành các **subdomain**, mỗi processor xử lý một subdomain.

### 2.2 1D Decomposition (Strip/Row-wise)
```
    Processor 0     ┌─────────────────┐  ← ghost row from P1
                    │  subdomain 0    │
                    │                 │
                    └─────────────────┘  → send to P1
    ─────────────────────────────────────
    Processor 1     ┌─────────────────┐  ← ghost row from P0
                    │  subdomain 1    │
                    │                 │
                    └─────────────────┘  → send to P2
    ─────────────────────────────────────
    Processor 2     ┌─────────────────┐  ← ghost row from P1
                    │  subdomain 2    │
                    │                 │
                    └─────────────────┘
```

**Ưu điểm:**
- Đơn giản implement
- Communication pattern đơn giản (chỉ với 2 neighbors)

**Nhược điểm:**
- Surface-to-volume ratio không tối ưu

### 2.3 2D Decomposition (Block)
```
    ┌─────┬─────┬─────┐
    │ P0  │ P1  │ P2  │
    ├─────┼─────┼─────┤
    │ P3  │ P4  │ P5  │
    ├─────┼─────┼─────┤
    │ P6  │ P7  │ P8  │
    └─────┴─────┴─────┘
```

**Ưu điểm:**
- Tối ưu surface-to-volume ratio
- Scale tốt hơn với số processor lớn

**Nhược điểm:**
- Communication phức tạp hơn (4 neighbors)
- Cần xử lý góc

## 3. Ghost Cells (Halo Exchange)

### 3.1 Khái niệm
**Ghost cells** (hay **halo**) là các điểm biên được sao chép từ processor láng giềng.

```
    Processor i storage:
    ┌───────────────────────────────┐
    │ ghost │ local data  │ ghost  │
    │ (from │  (own)      │ (from  │
    │  i-1) │             │  i+1)  │
    └───────────────────────────────┘
```

### 3.2 Tại sao cần Ghost cells?
Để tính giá trị tại biên của subdomain, cần biết giá trị từ subdomain lân cận:

```
    subdomain i        boundary        subdomain i+1
    ─────────────────────────────────────────────────
    ... c[n-1]  c[n]  │  c[0]   c[1] ...
                  ↑      ↑
              cần c[0]  cần c[n]
              từ i+1    từ i
```

### 3.3 Halo Exchange Pattern
```
Phase 1: Send boundaries
  P0 → P1: send right boundary
  P1 → P0: send left boundary
  P1 → P2: send right boundary
  P2 → P1: send left boundary
  ...

Phase 2: Receive into ghost cells
  P0: receive into left ghost (from P-1 or boundary condition)
  P0: receive into right ghost (from P1)
  ...
```

## 4. Parallel Jacobi Iteration

### 4.1 Algorithm
```
main() {
    decompose_lattice()
    initialize_lattice_sites()
    set_boundary_conditions()

    do {
        exchange_boundary_strips_with_neighbors()

        for all grid points in this processor {
            update according to Jacobi iteration
            calculate local δ
        }

        // Global reduction for stopping criterion
        δ_global = MPI_Reduce(δ_local, MAX)

    } while (δ_global > tolerance)

    print_results_to_file()
}
```

### 4.2 MPI Implementation Outline
```c
// Initialization
MPI_Init(&argc, &argv);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);

// Determine local domain
rows_per_proc = N / size;
local_start = rank * rows_per_proc;
local_end = local_start + rows_per_proc;

// Allocate local arrays (with ghost rows)
double *u_old = malloc((rows_per_proc + 2) * N * sizeof(double));
double *u_new = malloc((rows_per_proc + 2) * N * sizeof(double));

// Main iteration loop
do {
    // Halo exchange
    if (rank > 0)
        MPI_Sendrecv(u_old[1], N, MPI_DOUBLE, rank-1, 0,
                     u_old[0], N, MPI_DOUBLE, rank-1, 0,
                     MPI_COMM_WORLD, &status);

    if (rank < size-1)
        MPI_Sendrecv(u_old[rows_per_proc], N, MPI_DOUBLE, rank+1, 0,
                     u_old[rows_per_proc+1], N, MPI_DOUBLE, rank+1, 0,
                     MPI_COMM_WORLD, &status);

    // Jacobi update
    local_delta = 0.0;
    for (i = 1; i <= rows_per_proc; i++) {
        for (j = 1; j < N-1; j++) {
            u_new[i][j] = 0.25 * (u_old[i-1][j] + u_old[i+1][j] +
                                  u_old[i][j-1] + u_old[i][j+1]);
            diff = fabs(u_new[i][j] - u_old[i][j]);
            if (diff > local_delta) local_delta = diff;
        }
    }

    // Global reduction
    MPI_Allreduce(&local_delta, &global_delta, 1, MPI_DOUBLE,
                  MPI_MAX, MPI_COMM_WORLD);

    // Swap arrays
    swap(&u_old, &u_new);

} while (global_delta > tolerance);
```

## 5. Parallel Gauss-Seidel với Red-Black Ordering

### 5.1 Thách thức
Gauss-Seidel chuẩn (row-wise) là **inherently sequential** - không thể song song hóa trực tiếp.

### 5.2 Giải pháp: Red-Black Ordering
```
Phase 1: Update all RED points (in parallel)
    - RED points only depend on BLACK points
    - All RED updates are independent

Exchange boundaries (RED values)

Phase 2: Update all BLACK points (in parallel)
    - BLACK points only depend on RED points
    - All BLACK updates are independent

Exchange boundaries (BLACK values)
```

### 5.3 Algorithm
```
do {
    // Phase 1: Update RED
    exchange_boundary_strips()
    for all RED grid points in this processor {
        update according to Gauss-Seidel
        calculate local δ_red
    }

    // Phase 2: Update BLACK
    exchange_boundary_strips()
    for all BLACK grid points in this processor {
        update according to Gauss-Seidel
        calculate local δ_black
    }

    δ = max(δ_red, δ_black)
    δ_global = MPI_Allreduce(δ, MAX)

} while (δ_global > tolerance)
```

### 5.4 Note về Communication
- Jacobi: 1 exchange per iteration
- Red-Black: 2 exchanges per iteration (sau mỗi phase)
- Nhưng tổng communication volume giống nhau (chỉ trao đổi nửa số điểm mỗi lần)

## 6. Parallel SOR

### 6.1 SOR với Red-Black
Giống Gauss-Seidel Red-Black, chỉ thay đổi update formula:

```c
// Thay vì:
u_new = 0.25 * (u_n + u_s + u_e + u_w)

// Dùng:
aver = 0.25 * (u_n + u_s + u_e + u_w)
res = aver - u_old
u_new = u_old + omega * res
```

### 6.2 Chọn ω tối ưu
```
omega_opt = 2.0 - 2.0 * PI / n
```

Với n là kích thước lưới tổng (không phải local).

## 7. Stopping Criterion Song song

### 7.1 Vấn đề
- Mỗi processor chỉ biết δ_local của mình
- Cần δ_global để quyết định dừng

### 7.2 Giải pháp: Global Reduction
```c
MPI_Allreduce(&local_delta, &global_delta, 1, MPI_DOUBLE,
              MPI_MAX, MPI_COMM_WORLD);
```

### 7.3 Tối ưu hóa
Thay vì check mỗi iteration, có thể check mỗi q iterations:
```c
if (iter % q == 0) {
    MPI_Allreduce(&local_delta, &global_delta, ...);
    if (global_delta < tolerance) break;
}
```

Điều này giảm overhead của global communication.

## 8. Load Balancing

### 8.1 Vấn đề
Nếu n không chia hết cho p (số processors):
- Một số processor có nhiều hàng hơn
- Gây mất cân bằng tải

### 8.2 Giải pháp
```c
base_rows = n / p
extra = n % p

if (rank < extra)
    local_rows = base_rows + 1
else
    local_rows = base_rows
```

## 9. Communication Patterns

### 9.1 Point-to-point (MPI_Send/Recv)
```c
if (rank > 0) {
    MPI_Send(top_row, N, MPI_DOUBLE, rank-1, tag, comm);
    MPI_Recv(ghost_top, N, MPI_DOUBLE, rank-1, tag, comm, &status);
}
```

### 9.2 Sendrecv (tránh deadlock)
```c
MPI_Sendrecv(top_row, N, MPI_DOUBLE, rank-1, tag1,
             ghost_bottom, N, MPI_DOUBLE, rank+1, tag2,
             comm, &status);
```

### 9.3 Non-blocking (overlap computation and communication)
```c
MPI_Isend(boundary, N, MPI_DOUBLE, neighbor, tag, comm, &req_send);
MPI_Irecv(ghost, N, MPI_DOUBLE, neighbor, tag, comm, &req_recv);

// Compute interior points while communication happens
compute_interior();

MPI_Wait(&req_send, &status);
MPI_Wait(&req_recv, &status);

// Now compute boundary points
compute_boundary();
```

## 10. Performance Analysis

### 10.1 Speedup
```
Speedup(p) = T_serial / T_parallel(p)
```

### 10.2 Efficiency
```
Efficiency(p) = Speedup(p) / p
```

### 10.3 Amdahl's Law
Nếu fraction f của code là serial:
```
Speedup_max = 1 / f
```

### 10.4 Communication Overhead
Cho 1D decomposition với p processors:
- Computation: O(n²/p) per iteration
- Communication: O(n) per iteration (2 row exchanges)
- Communication-to-computation ratio: O(p/n)

Khi p tăng, communication overhead tăng.

## 11. Best Practices

1. **Overlap communication và computation**
   - Dùng non-blocking MPI calls
   - Tính interior trong khi chờ boundary exchange

2. **Minimize synchronization**
   - Không check convergence mỗi iteration
   - Batch multiple iterations before sync

3. **Choose right decomposition**
   - 1D cho small-medium problems
   - 2D cho large problems với many processors

4. **Profile và tune**
   - Đo communication time vs computation time
   - Adjust domain size per processor
