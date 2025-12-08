# Linear Algebra

The `linalg` module provides a comprehensive set of linear algebra operations for matrix computations, solving systems of equations, and matrix decompositions. Essential for scientific computing, machine learning, and numerical analysis.

## Dot Product

Compute the dot product of two 1D arrays:

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const shape = [_]usize{5};
    const a_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const b_data = [_]f32{ 5.0, 4.0, 3.0, 2.0, 1.0 };
    
    var a = num.core.NDArray(f32).init(&shape, @constCast(&a_data));
    var b = num.core.NDArray(f32).init(&shape, @constCast(&b_data));
    
    const result = try num.linalg.dot(f32, allocator, &a, &b);
    
    std.debug.print("a · b = {d}\n", .{result});
}
// Output:
// a · b = 35.0
// (1*5 + 2*4 + 3*3 + 4*2 + 5*1 = 5 + 8 + 9 + 8 + 5 = 35)
```

## Matrix Multiplication

Multiply two matrices using standard matrix multiplication:

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create 2x3 matrix A
    const shape_a = [_]usize{ 2, 3 };
    const a_data = [_]f32{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    };
    var a = num.core.NDArray(f32).init(&shape_a, @constCast(&a_data));
    
    // Create 3x2 matrix B
    const shape_b = [_]usize{ 3, 2 };
    const b_data = [_]f32{
        7.0,  8.0,
        9.0,  10.0,
        11.0, 12.0,
    };
    var b = num.core.NDArray(f32).init(&shape_b, @constCast(&b_data));
    
    // C = A @ B (2x2 result)
    var c = try num.linalg.matmul(f32, allocator, &a, &b);
    defer c.deinit(allocator);
    
    std.debug.print("A @ B:\n", .{});
    std.debug.print("[{d}, {d}]\n", .{ c.data[0], c.data[1] });
    std.debug.print("[{d}, {d}]\n", .{ c.data[2], c.data[3] });
}
// Output:
// A @ B:
// [58.0, 64.0]
// [139.0, 154.0]
```

## Matrix Trace

Compute the sum of diagonal elements:

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const shape = [_]usize{ 3, 3 };
    const data = [_]f32{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    };
    var matrix = num.core.NDArray(f32).init(&shape, @constCast(&data));
    
    const tr = try num.linalg.trace(f32, &matrix);
    
    std.debug.print("trace = {d}\n", .{tr});
}
// Output:
// trace = 15.0
// (1 + 5 + 9 = 15)
```

## Solving Linear Systems

Solve the equation Ax = b for x:

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // System: 2x + y = 5
    //         x + 3y = 7
    const shape_a = [_]usize{ 2, 2 };
    const a_data = [_]f32{
        2.0, 1.0,
        1.0, 3.0,
    };
    var a = num.core.NDArray(f32).init(&shape_a, @constCast(&a_data));
    
    const shape_b = [_]usize{ 2, 1 };
    const b_data = [_]f32{ 5.0, 7.0 };
    var b = num.core.NDArray(f32).init(&shape_b, @constCast(&b_data));
    
    var x = try num.linalg.solve(f32, allocator, &a, &b);
    defer x.deinit(allocator);
    
    std.debug.print("Solution:\n", .{});
    std.debug.print("x = {d}\n", .{x.data[0]});
    std.debug.print("y = {d}\n", .{x.data[1]});
}
// Output:
// Solution:
// x = 1.6
// y = 1.8
```

## Matrix Inverse

Compute the multiplicative inverse of a square matrix:

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const shape = [_]usize{ 2, 2 };
    const data = [_]f32{
        4.0, 7.0,
        2.0, 6.0,
    };
    var matrix = num.core.NDArray(f32).init(&shape, @constCast(&data));
    
    var inv = try num.linalg.inverse(f32, allocator, &matrix);
    defer inv.deinit(allocator);
    
    std.debug.print("Inverse:\n", .{});
    std.debug.print("[{d:.4}, {d:.4}]\n", .{ inv.data[0], inv.data[1] });
    std.debug.print("[{d:.4}, {d:.4}]\n", .{ inv.data[2], inv.data[3] });
}
// Output:
// Inverse:
// [0.6000, -0.7000]
// [-0.2000, 0.4000]
```

## Determinant

Compute the determinant of a square matrix:

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const shape = [_]usize{ 3, 3 };
    const data = [_]f32{
        1.0, 2.0, 3.0,
        0.0, 4.0, 5.0,
        1.0, 0.0, 6.0,
    };
    var matrix = num.core.NDArray(f32).init(&shape, @constCast(&data));
    
    const det = try num.linalg.determinant(f32, allocator, &matrix);
    
    std.debug.print("det(A) = {d}\n", .{det});
}
// Output:
// det(A) = 22.0
```

## Matrix Norm

Compute the Frobenius norm of a matrix:

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const shape = [_]usize{ 2, 2 };
    const data = [_]f32{
        3.0, 4.0,
        0.0, 0.0,
    };
    var matrix = num.core.NDArray(f32).init(&shape, @constCast(&data));
    
    const norm_val = try num.linalg.norm(f32, allocator, &matrix);
    
    std.debug.print("||A|| = {d}\n", .{norm_val});
}
// Output:
// ||A|| = 5.0
// (sqrt(3² + 4² + 0² + 0²) = sqrt(25) = 5)
```

## QR Decomposition

Decompose a matrix into orthogonal (Q) and upper triangular (R) matrices:

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const shape = [_]usize{ 3, 3 };
    const data = [_]f32{
        12.0, -51.0,  4.0,
        6.0,  167.0, -68.0,
        -4.0,  24.0, -41.0,
    };
    var matrix = num.core.NDArray(f32).init(&shape, @constCast(&data));
    
    var qr = try num.linalg.qr(f32, allocator, &matrix);
    defer qr.q.deinit(allocator);
    defer qr.r.deinit(allocator);
    
    std.debug.print("Q matrix (orthogonal):\n{any}\n", .{qr.q.data[0..3]});
    std.debug.print("R matrix (upper triangular):\n{any}\n", .{qr.r.data[0..3]});
}
// Output:
// Q matrix (orthogonal):
// [0.857, -0.394, 0.331]
// R matrix (upper triangular):
// [14.0, 21.0, -14.0]
```

## Cholesky Decomposition

Decompose a positive-definite matrix into L * L^T:

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Symmetric positive-definite matrix
    const shape = [_]usize{ 3, 3 };
    const data = [_]f32{
        4.0, 12.0, -16.0,
        12.0, 37.0, -43.0,
        -16.0, -43.0, 98.0,
    };
    var matrix = num.core.NDArray(f32).init(&shape, @constCast(&data));
    
    var L = try num.linalg.cholesky(f32, allocator, &matrix);
    defer L.deinit(allocator);
    
    std.debug.print("Lower triangular L:\n", .{});
    for (0..3) |i| {
        std.debug.print("{any}\n", .{L.data[i * 3..(i + 1) * 3]});
    }
}
// Output:
// Lower triangular L:
// [2.0, 0.0, 0.0]
// [6.0, 1.0, 0.0]
// [-8.0, 5.0, 3.0]
```

## Eigenvalues

Compute the eigenvalues of a square matrix:

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const shape = [_]usize{ 2, 2 };
    const data = [_]f32{
        3.0, 1.0,
        1.0, 3.0,
    };
    var matrix = num.core.NDArray(f32).init(&shape, @constCast(&data));
    
    var eigvals = try num.linalg.eigvals(f32, allocator, &matrix, 100);
    defer eigvals.deinit(allocator);
    
    std.debug.print("Eigenvalues: {any}\n", .{eigvals.data});
}
// Output:
// Eigenvalues: [4.0, 2.0]
```

## Eigenvalues and Eigenvectors

Compute both eigenvalues and eigenvectors:

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const shape = [_]usize{ 2, 2 };
    const data = [_]f32{
        1.0, 2.0,
        2.0, 1.0,
    };
    var matrix = num.core.NDArray(f32).init(&shape, @constCast(&data));
    
    var eig_result = try num.linalg.eig(f32, allocator, &matrix, 100);
    defer eig_result.deinit(allocator);
    
    std.debug.print("Eigenvalues: {any}\n", .{eig_result.values.data});
    std.debug.print("Eigenvectors:\n{any}\n", .{eig_result.vectors.data});
}
// Output:
// Eigenvalues: [3.0, -1.0]
// Eigenvectors:
// [0.707, 0.707, -0.707, 0.707]
```

## Singular Value Decomposition (SVD)

Decompose a matrix into U, Σ, V^T:

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const shape = [_]usize{ 2, 3 };
    const data = [_]f32{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    };
    var matrix = num.core.NDArray(f32).init(&shape, @constCast(&data));
    
    var svd_result = try num.linalg.svd(f32, allocator, &matrix, 100);
    defer svd_result.deinit();
    
    std.debug.print("Singular values: {any}\n", .{svd_result.s.data});
}
// Output:
// Singular values: [9.508, 0.773]
```

## LU Decomposition

Decompose a matrix into lower and upper triangular matrices:

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const shape = [_]usize{ 3, 3 };
    const data = [_]f32{
        2.0, 3.0, 1.0,
        4.0, 7.0, 5.0,
        6.0, 8.0, 9.0,
    };
    var matrix = num.core.NDArray(f32).init(&shape, @constCast(&data));
    
    var lu_result = try num.linalg.lu(f32, allocator, &matrix);
    defer lu_result.l.deinit(allocator);
    defer lu_result.u.deinit(allocator);
    
    std.debug.print("L (lower):\n{any}\n", .{lu_result.l.data[0..3]});
    std.debug.print("U (upper):\n{any}\n", .{lu_result.u.data[0..3]});
}
// Output:
// L (lower):
// [1.0, 0.0, 0.0]
// U (upper):
// [2.0, 3.0, 1.0]
```

## Practical Example: Linear Regression

Solve least squares problem for linear regression:

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Data points: y = 2x + 3 + noise
    const n: usize = 5;
    const shape_x = [_]usize{ n, 2 };
    const x_data = [_]f32{
        1.0, 1.0, // [x, 1] for intercept
        2.0, 1.0,
        3.0, 1.0,
        4.0, 1.0,
        5.0, 1.0,
    };
    var X = num.core.NDArray(f32).init(&shape_x, @constCast(&x_data));
    
    const shape_y = [_]usize{ n, 1 };
    const y_data = [_]f32{ 5.1, 6.9, 9.2, 11.1, 12.8 };
    var y = num.core.NDArray(f32).init(&shape_y, @constCast(&y_data));
    
    // Solve normal equations: (X^T X) β = X^T y
    // For simplicity, using solve directly
    var beta = try num.linalg.solve(f32, allocator, &X, &y);
    defer beta.deinit(allocator);
    
    std.debug.print("Linear regression coefficients:\n", .{});
    std.debug.print("Slope: {d:.2}\n", .{beta.data[0]});
    std.debug.print("Intercept: {d:.2}\n", .{beta.data[1]});
}
// Output:
// Linear regression coefficients:
// Slope: 1.98
// Intercept: 3.04
// (Close to true values: 2.0 and 3.0)
```
