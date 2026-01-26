# Các Phương Pháp Số Giải Phương Trình Laplace

## 1. Tổng quan

Có hai hướng tiếp cận chính để giải hệ phương trình **Ax = b**:

| Phương pháp | Độ chính xác | Độ phức tạp | Song song hóa |
|-------------|--------------|-------------|---------------|
| **Trực tiếp** (Direct) | Chính xác (trong sai số máy) | O(N³) dense, O(N²) banded | Khó |
| **Lặp** (Iterative) | Xấp xỉ (đến ngưỡng ε) | O(N) - O(N²) | Dễ |

## 2. Phương pháp Trực tiếp

### 2.1 Gaussian Elimination
Biến đổi ma trận về dạng tam giác trên rồi thế ngược:

```
Ax = b  →  Ux = c  (dạng tam giác trên)
```

**Các bước:**
1. Chọn pivot (partial pivoting để ổn định)
2. Khử để tạo số 0 dưới đường chéo
3. Back substitution

### 2.2 LU Factorization
Phân tích A = LU:
- **L**: Ma trận tam giác dưới (lower)
- **U**: Ma trận tam giác trên (upper)

```
Ax = b  →  LUx = b
Bước 1: Ly = b  (forward substitution)
Bước 2: Ux = y  (backward substitution)
```

### 2.3 Thư viện song song
- **LAPACK**: Ma trận dày, tuần tự
- **ScaLAPACK**: Ma trận dày, song song (distributed memory)
- **BLAS 1/2/3**: Các phép toán cơ bản

## 3. Phương pháp Lặp (Iterative Methods)

### 3.1 Ý tưởng chung
Từ **Ax = b**, xây dựng lặp:
```
x^(n+1) = Φ(x^(n))
```

Với **A = D + E + F** (D: đường chéo, E: tam giác dưới, F: tam giác trên):
```
x^(n+1) = (I - B⁻¹A)x^(n) + B⁻¹b
```

### 3.2 Jacobi Iteration

**Công thức:**
```
c_{l,m}^{(n+1)} = 1/4 [c_{l+1,m}^{(n)} + c_{l-1,m}^{(n)} + c_{l,m+1}^{(n)} + c_{l,m-1}^{(n)}]
```

**Đặc điểm:**
- **B = D** (chỉ dùng đường chéo)
- Cần 2 mảng: old values và new values
- **Dễ song song hóa nhất** - tất cả điểm cập nhật độc lập
- Hội tụ chậm: O(N²) iterations

**Pseudo-code:**
```
do {
    δ = 0
    for i = 0 to max:
        for j = 0 to max:
            if (boundary) continue

            west  = c[i-1][j]  (or periodic)
            east  = c[i+1][j]  (or periodic)
            south = c[i][j-1]  (or fixed: c0)
            north = c[i][j+1]  (or fixed: cL)

            c_new[i][j] = 0.25 * (west + east + south + north)

            if |c_new[i][j] - c[i][j]| > δ:
                δ = |c_new[i][j] - c[i][j]|

    swap(c, c_new)
} while (δ > tolerance)
```

### 3.3 Gauss-Seidel Iteration

**Công thức:**
```
c_{l,m}^{(n+1)} = 1/4 [c_{l+1,m}^{(n)} + c_{l-1,m}^{(n+1)} + c_{l,m+1}^{(n)} + c_{l,m-1}^{(n+1)}]
```

**Đặc điểm:**
- **B = D + E** (dùng cả tam giác dưới)
- Cập nhật **in-place** - dùng giá trị mới ngay khi có
- Chỉ cần 1 mảng
- Hội tụ nhanh gấp 2 lần Jacobi
- **Khó song song hóa** theo row-wise ordering

**So sánh với Jacobi:**
```
Jacobi:        c_new = f(c_old, c_old, c_old, c_old)
Gauss-Seidel:  c_new = f(c_old, c_NEW, c_old, c_NEW)
                              ↑           ↑
                        đã cập nhật   đã cập nhật
```

### 3.4 Successive Over-Relaxation (SOR)

**Công thức:**
```
c_{l,m}^{(n+1)} = ω/4 [c_{l+1,m}^{(n)} + c_{l-1,m}^{(n+1)} + c_{l,m+1}^{(n)} + c_{l,m-1}^{(n+1)}] + (1-ω)c_{l,m}^{(n)}
```

Hay viết theo residual:
```
aver = (u_n + u_s + u_e + u_w) / 4
res  = aver - u_c
u_new = u_c + ω * res
```

**Tham số ω (relaxation factor):**
- ω = 1: Gauss-Seidel
- 0 < ω < 1: Under-relaxation
- 1 < ω < 2: **Over-relaxation** (nhanh hơn)
- ω ≥ 2: Không hội tụ

**Giá trị tối ưu (Young, 1954):**
```
ω_opt = 2 - 2π/n
```

Với n = 250: ω_opt ≈ 1.97

**Hiệu quả:**
- SOR với ω_opt: **O(n) iterations** (so với O(n²) của Jacobi)
- Giảm thời gian tính toán đáng kể

## 4. So sánh Hiệu quả

### 4.1 Số vòng lặp cần thiết (N = 40)
```
p (precision)  | Jacobi  | Gauss-Seidel | SOR (ω=1.9)
---------------|---------|--------------|-------------
2              |   ~800  |    ~400      |    ~30
4              |  ~5000  |   ~2500      |   ~100
6              | ~15000  |   ~7500      |   ~200
8              | ~25000  |  ~12500      |   ~300
10             | ~30000  |  ~15000      |   ~400
```

### 4.2 Mean Error (N = 40)
SOR đạt sai số nhỏ hơn nhiều với cùng stopping criterion:
- Jacobi (p=6): error ≈ 10⁻⁴
- SOR (p=6): error ≈ 10⁻⁷

## 5. Red-Black Ordering (Parity Ordering)

### 5.1 Vấn đề với Gauss-Seidel
- Row-wise update là tuần tự
- Không thể song song hóa trực tiếp

### 5.2 Giải pháp: Checkerboard pattern
```
  -  1  0  1  0  -
  1  0  1  0  1  0
  0  1  0  1  0  1
  1  0  1  0  1  0
  0  1  0  1  0  1
  -  0  1  0  1  -

  0 = even (red)
  1 = odd (black)
```

**Tính chất quan trọng:**
- Mỗi điểm **red** chỉ phụ thuộc vào các điểm **black** và ngược lại
- Có thể cập nhật **tất cả red** song song, rồi **tất cả black** song song

### 5.3 Algorithm với Red-Black ordering
```
do {
    // Phase 1: Update all RED points
    parallel for all red points (i,j):
        u[i,j] = next(u[i,j], neighbors)  // neighbors are all black

    // Phase 2: Update all BLACK points
    parallel for all black points (i,j):
        u[i,j] = next(u[i,j], neighbors)  // neighbors are all red

} while (not converged)
```

### 5.4 Coarse-grained Red-Black
Thay vì red-black cho từng điểm, có thể áp dụng cho từng **domain**:

```
    P₀    P₁    P₂
  ┌─────┬─────┬─────┐
  │ RED │BLACK│ RED │
  │     │     │     │
  └─────┴─────┴─────┘
```

- Domain RED cập nhật trước
- Trao đổi biên
- Domain BLACK cập nhật sau

## 6. Biểu diễn Ma trận

### 6.1 Jacobi trong ký hiệu ma trận
```
B = D  →  a_{ii} * x_i^{(n+1)} = -Σ_{j≠i} a_{ij} * x_j^{(n)} + b_i
```

### 6.2 Gauss-Seidel trong ký hiệu ma trận
```
B = D + E  →  a_{ii} * x_i^{(n+1)} = -Σ_{j<i} a_{ij} * x_j^{(n+1)} - Σ_{j>i} a_{ij} * x_j^{(n)} + b_i
```

### 6.3 SOR trong ký hiệu ma trận
```
B = (1/ω)D + E  →  tương tự với hệ số ω
```

## 7. Kết luận

| Phương pháp | Iterations | Memory | Parallelism | Recommended |
|-------------|------------|--------|-------------|-------------|
| Jacobi | O(n²) | 2 arrays | Excellent | Learning |
| Gauss-Seidel | O(n²)/2 | 1 array | Poor (row-wise) | Sequential |
| GS Red-Black | O(n²)/2 | 1 array | Good | Parallel |
| SOR | O(n) | 1 array | Good (RB) | **Production** |

**Khuyến nghị:**
1. **Học tập**: Bắt đầu với Jacobi (đơn giản nhất)
2. **Sequential**: Gauss-Seidel hoặc SOR
3. **Parallel**: SOR với Red-Black ordering
4. **Large scale**: Multigrid methods (O(N) time)
