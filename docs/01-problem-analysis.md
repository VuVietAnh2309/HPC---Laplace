# Phân Tích Bài Toán: Giải Phương Trình Laplace

## 1. Giới thiệu

### 1.1 Bối cảnh
Phương trình Laplace là một trong những phương trình vi phân riêng phần quan trọng nhất trong khoa học và kỹ thuật. Nó mô tả:
- Trường điện thế tĩnh
- Trường hấp dẫn
- Dòng chảy nhiệt ổn định (steady-state heat flow)
- Dòng chảy chất lỏng lý tưởng

### 1.2 Phương trình Laplace
Phương trình Laplace trong không gian 2D:

```
∇²u = ∂²u/∂x² + ∂²u/∂y² = 0
```

Đây là phương trình khuếch tán không phụ thuộc thời gian (time-independent diffusion equation).

## 2. Mô tả Bài toán

### 2.1 Bài toán dòng nhiệt ổn định
Xét một tấm phẳng mỏng, đồng nhất với:
- Hai mặt được cách nhiệt hoàn toàn
- Biên có nhiệt độ cố định (Dirichlet boundary conditions)
- Cần tìm phân bố nhiệt độ cân bằng u(x,y) bên trong

### 2.2 Miền tính toán
```
        u₁ = 0 (biên trên)
    ┌─────────────────┐
    │                 │
u₄=0│     u = ?       │ u₃ = 100 (biên phải)
    │                 │
    └─────────────────┘
        u₂ = 100 (biên dưới)
```

### 2.3 Điều kiện biên
- **Biên cố định (Dirichlet)**: Nhiệt độ được chỉ định trước trên biên
- **Biên tuần hoàn (Periodic)**: c(x,y) = c(x+L,y) trong một số bài toán

## 3. Rời rạc hóa (Discretization)

### 3.1 Lưới vuông (Square Grid)
Chia miền tính toán thành lưới (n+2) × (n+2):
- n × n điểm bên trong (unknown)
- 4n điểm biên (known)
- 4 điểm góc (không sử dụng)

```
  -  +  +  +  +  -
  +  ?  ?  ?  ?  +
  +  ?  ?  ?  ?  +
  +  ?  ?  ?  ?  +
  +  ?  ?  ?  ?  +
  -  +  +  +  +  -

  - : góc (không dùng)
  + : biên (đã biết)
  ? : bên trong (cần tìm)
```

### 3.2 Sai phân hữu hạn (Finite Difference)
Sử dụng khai triển Taylor, ta có xấp xỉ:

```
∂²u/∂x² ≈ (u_{i+1,j} - 2u_{i,j} + u_{i-1,j}) / h²
∂²u/∂y² ≈ (u_{i,j+1} - 2u_{i,j} + u_{i,j-1}) / h²
```

### 3.3 Phương trình sai phân
Thay vào phương trình Laplace:

```
u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j} = 0
```

Hay tương đương:

```
u_{i,j} = (u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1}) / 4
```

**Ý nghĩa vật lý**: Nhiệt độ tại mỗi điểm bằng trung bình của 4 điểm lân cận.

### 3.4 Stencil 5 điểm
```
         u_n (north)
           │
u_w ─── u_c ─── u_e
(west)     │    (east)
         u_s (south)
```

## 4. Hệ phương trình tuyến tính

### 4.1 Dạng ma trận
Hệ n² phương trình có thể viết dưới dạng:

```
Ax = b
```

Trong đó:
- **A**: Ma trận N×N (N = n²) - ma trận thưa, có cấu trúc băng
- **x**: Vector nghiệm (nhiệt độ tại các điểm bên trong)
- **b**: Vector vế phải (chứa điều kiện biên)

### 4.2 Cấu trúc ma trận A (1D case)
Với bài toán 1D, ma trận A là tam đường chéo:
```
⎛  2  -1   0   0  ...  0 ⎞ ⎛c₁⎞   ⎛C₀⎞
⎜ -1   2  -1   0  ...  0 ⎟ ⎜c₂⎟   ⎜ 0⎟
⎜  0  -1   2  -1  ...  0 ⎟ ⎜c₃⎟ = ⎜ 0⎟
⎜  ⋮       ⋱   ⋱   ⋱   ⋮⎟ ⎜ ⋮⎟   ⎜ ⋮⎟
⎝  0  ...  0  -1   2    ⎠  ⎝cₙ⎠   ⎝Cₗ⎠
```

### 4.3 Độ phức tạp
| Phương pháp | Ma trận dày | Ma trận thưa |
|-------------|-------------|--------------|
| Direct (LU) | O(N³)       | O(N²)        |
| Iterative   | O(N²)       | O(N)         |

Với lưới n×n (N = n²):
- Direct method: O(n⁶) - không thực tế cho n lớn
- Iterative method: O(n²) đến O(n⁴) tùy phương pháp

## 5. Tại sao cần Tính toán Song song?

### 5.1 Thách thức về quy mô
- Lưới 250×250: 62,500 ẩn số
- Lưới 1000×1000: 1,000,000 ẩn số
- Direct method không khả thi cho các bài toán lớn

### 5.2 Ưu điểm của phương pháp lặp
1. **Độ phức tạp thấp hơn**: O(N) thay vì O(N³)
2. **Dễ song song hóa**: Các điểm có thể cập nhật độc lập
3. **Tiết kiệm bộ nhớ**: Chỉ cần lưu ma trận thưa

### 5.3 Mô hình song song
- **Domain decomposition**: Chia lưới thành các miền con
- **Mỗi processor xử lý một miền**
- **Trao đổi biên**: Các processor láng giềng trao đổi giá trị biên

## 6. Tiêu chí hội tụ

### 6.1 Stopping criterion
Lặp cho đến khi:
```
max|c_{i,j}^{(n+1)} - c_{i,j}^{(n)}| < ε
```

Với ε = 10^(-p), p là số chữ số chính xác mong muốn.

### 6.2 Nghiệm chính xác (test case)
Với điều kiện biên tuần hoàn theo x và cố định theo y:
- c(x, y=0) = 0
- c(x, y=1) = 1
- c(x,y) = c(x+1,y)

Nghiệm chính xác: **c(x,y) = y** (tuyến tính)

## 7. Kết luận

Bài toán giải phương trình Laplace:
1. Có ý nghĩa vật lý quan trọng
2. Đòi hỏi phương pháp số để giải trên máy tính
3. Phù hợp để song song hóa do tính chất địa phương của stencil
4. Là nền tảng cho nhiều ứng dụng HPC thực tế
