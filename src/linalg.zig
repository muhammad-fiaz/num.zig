const std = @import("std");
const core = @import("core.zig");
const NDArray = core.NDArray;
const Allocator = std.mem.Allocator;

/// Computes the dot product of two 1D arrays.
///
/// Logic:
/// result = sum(a * b)
///
/// Arguments:
///     T: The data type of the array elements.
///     allocator: The allocator (unused for scalar return, but kept for API consistency).
///     a: The first input 1D NDArray.
///     b: The second input 1D NDArray.
///
/// Returns:
///     The scalar dot product of the two arrays.
///
/// Example:
/// ```zig
/// var a = try NDArray(f32).init(allocator, &.{3}, &.{1.0, 2.0, 3.0});
/// defer a.deinit();
/// var b = try NDArray(f32).init(allocator, &.{3}, &.{4.0, 5.0, 6.0});
/// defer b.deinit();
///
/// const result = try linalg.dot(f32, allocator, &a, &b);
/// // result is 32.0 (1*4 + 2*5 + 3*6)
/// ```
pub fn dot(comptime T: type, allocator: Allocator, a: *const NDArray(T), b: *const NDArray(T)) !T {
    _ = allocator; // Not used for scalar return
    if (a.rank() != 1 or b.rank() != 1) return core.Error.RankMismatch;
    if (a.shape[0] != b.shape[0]) return core.Error.ShapeMismatch;

    var sum: T = 0;
    var i: usize = 0;
    while (i < a.shape[0]) : (i += 1) {
        const val_a = a.data[i * a.strides[0]];
        const val_b = b.data[i * b.strides[0]];
        sum += val_a * val_b;
    }
    return sum;
}

/// Performs matrix multiplication for 2D arrays.
///
/// Computes the product of two matrices `a` and `b`.
///
/// Logic:
/// result[i, j] = sum(a[i, k] * b[k, j])
///
/// Arguments:
///     T: The data type of the array elements.
///     allocator: The allocator to use for the result array.
///     a: The first input 2D NDArray (M x K).
///     b: The second input 2D NDArray (K x N).
///
/// Returns:
///     A new NDArray(T) of shape (M, N) containing the matrix product.
///
/// Example:
/// ```zig
/// var a = try NDArray(f32).init(allocator, &.{2, 2}, &.{1.0, 2.0, 3.0, 4.0});
/// defer a.deinit();
/// var b = try NDArray(f32).init(allocator, &.{2, 2}, &.{2.0, 0.0, 1.0, 2.0});
/// defer b.deinit();
///
/// var result = try linalg.matmul(f32, allocator, &a, &b);
/// defer result.deinit();
/// // result is {{4.0, 4.0}, {10.0, 8.0}}
/// ```
pub fn matmul(comptime T: type, allocator: Allocator, a: *const NDArray(T), b: *const NDArray(T)) !NDArray(T) {
    if (a.rank() != 2 or b.rank() != 2) return core.Error.RankMismatch;
    if (a.shape[1] != b.shape[0]) return core.Error.ShapeMismatch;

    const M = a.shape[0];
    const K = a.shape[1];
    const N = b.shape[1];

    var result = try NDArray(T).zeros(allocator, &.{ M, N });

    // Tiled implementation for cache optimization
    const BLOCK_SIZE = 64;

    var i_blk: usize = 0;
    while (i_blk < M) : (i_blk += BLOCK_SIZE) {
        const i_max = @min(i_blk + BLOCK_SIZE, M);

        var k_blk: usize = 0;
        while (k_blk < K) : (k_blk += BLOCK_SIZE) {
            const k_max = @min(k_blk + BLOCK_SIZE, K);

            var j_blk: usize = 0;
            while (j_blk < N) : (j_blk += BLOCK_SIZE) {
                const j_max = @min(j_blk + BLOCK_SIZE, N);

                var i: usize = i_blk;
                while (i < i_max) : (i += 1) {
                    var k: usize = k_blk;
                    while (k < k_max) : (k += 1) {
                        const val_a = a.data[i * a.strides[0] + k * a.strides[1]];

                        var j: usize = j_blk;
                        while (j < j_max) : (j += 1) {
                            const val_b = b.data[k * b.strides[0] + j * b.strides[1]];
                            result.data[i * result.strides[0] + j * result.strides[1]] += val_a * val_b;
                        }
                    }
                }
            }
        }
    }

    return result;
}

/// Computes the trace (sum of diagonal elements) of a 2D array.
///
/// Logic:
/// result = sum(diag(a))
///
/// Arguments:
///     T: The data type of the array elements.
///     a: The input 2D NDArray.
///
/// Returns:
///     The sum of the diagonal elements.
///
/// Example:
/// ```zig
/// var a = try NDArray(f32).init(allocator, &.{2, 2}, &.{1.0, 2.0, 3.0, 4.0});
/// defer a.deinit();
///
/// const tr = try linalg.trace(f32, &a);
/// // tr is 5.0
/// ```
pub fn trace(comptime T: type, a: *const NDArray(T)) !T {
    if (a.rank() != 2) return core.Error.RankMismatch;
    const n = @min(a.shape[0], a.shape[1]);
    var sum: T = 0;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        sum += a.data[i * a.strides[0] + i * a.strides[1]];
    }
    return sum;
}

/// Solves a linear matrix equation, or system of linear scalar equations.
///
/// Computes the "exact" solution, x, of the well-determined, i.e., full rank, linear matrix equation ax = b.
/// Uses Gaussian elimination with partial pivoting.
///
/// Logic:
/// ax = b => x = a^-1 * b
///
/// Arguments:
///     T: The data type of the array elements.
///     allocator: The allocator to use for the result array.
///     a: The coefficient matrix (must be square).
///     b: The ordinate or "dependent variable" values.
///
/// Returns:
///     A new NDArray containing the solution.
///
/// Example:
/// ```zig
/// // Solve 2x + y = 5, x + y = 3
/// var a = try NDArray(f32).init(allocator, &.{2, 2}, &.{2.0, 1.0, 1.0, 1.0});
/// defer a.deinit();
/// var b = try NDArray(f32).init(allocator, &.{2}, &.{5.0, 3.0});
/// defer b.deinit();
///
/// var x = try linalg.solve(f32, allocator, &a, &b);
/// defer x.deinit();
/// // x is {2.0, 1.0}
/// ```
pub fn solve(comptime T: type, allocator: Allocator, a: *const NDArray(T), b: *const NDArray(T)) !NDArray(T) {
    _ = allocator;
    if (a.rank() != 2) return core.Error.RankMismatch;
    if (a.shape[0] != a.shape[1]) return core.Error.DimensionMismatch; // Must be square
    if (b.shape[0] != a.shape[0]) return core.Error.ShapeMismatch;

    const n = a.shape[0];

    // Create augmented matrix [A|b] or just work on copies.
    // Let's copy A and b.
    var A_copy = try a.copy();
    defer A_copy.deinit();

    // b can be 1D or 2D. If 1D, treat as column vector.
    var x = try b.copy();
    errdefer x.deinit();
    // If b is 1D, we solve for vector x. If 2D, we solve for matrix X.
    // Gaussian elimination works for both if we apply ops to rows of b/x.

    const num_rhs = if (b.rank() == 1) 1 else b.shape[1];

    // Gaussian elimination with partial pivoting
    var i: usize = 0;
    while (i < n) : (i += 1) {
        // Pivot
        var max_row = i;
        var max_val = @abs(A_copy.data[i * n + i]);

        var k: usize = i + 1;
        while (k < n) : (k += 1) {
            const val = @abs(A_copy.data[k * n + i]);
            if (val > max_val) {
                max_val = val;
                max_row = k;
            }
        }

        // Swap rows in A
        if (max_row != i) {
            var j: usize = i;
            while (j < n) : (j += 1) {
                const temp = A_copy.data[i * n + j];
                A_copy.data[i * n + j] = A_copy.data[max_row * n + j];
                A_copy.data[max_row * n + j] = temp;
            }

            // Swap rows in x (RHS)
            if (b.rank() == 1) {
                const temp = x.data[i];
                x.data[i] = x.data[max_row];
                x.data[max_row] = temp;
            } else {
                var col: usize = 0;
                while (col < num_rhs) : (col += 1) {
                    const temp = x.data[i * x.strides[0] + col * x.strides[1]];
                    x.data[i * x.strides[0] + col * x.strides[1]] = x.data[max_row * x.strides[0] + col * x.strides[1]];
                    x.data[max_row * x.strides[0] + col * x.strides[1]] = temp;
                }
            }
        }

        // Eliminate
        const pivot = A_copy.data[i * n + i];
        if (@abs(pivot) < 1e-9) return core.Error.DimensionMismatch; // Singular matrix

        var row: usize = i + 1;
        while (row < n) : (row += 1) {
            const factor = A_copy.data[row * n + i] / pivot;
            A_copy.data[row * n + i] = 0;

            var col: usize = i + 1;
            while (col < n) : (col += 1) {
                A_copy.data[row * n + col] -= factor * A_copy.data[i * n + col];
            }

            // Apply to RHS
            if (b.rank() == 1) {
                x.data[row] -= factor * x.data[i];
            } else {
                var j: usize = 0;
                while (j < num_rhs) : (j += 1) {
                    x.data[row * x.strides[0] + j * x.strides[1]] -= factor * x.data[i * x.strides[0] + j * x.strides[1]];
                }
            }
        }
    }

    // Back substitution
    var row_idx: usize = n;
    while (row_idx > 0) {
        row_idx -= 1;
        const pivot = A_copy.data[row_idx * n + row_idx];

        if (b.rank() == 1) {
            var sum_ax: T = 0;
            var col: usize = row_idx + 1;
            while (col < n) : (col += 1) {
                sum_ax += A_copy.data[row_idx * n + col] * x.data[col];
            }
            x.data[row_idx] = (x.data[row_idx] - sum_ax) / pivot;
        } else {
            var j: usize = 0;
            while (j < num_rhs) : (j += 1) {
                var sum_ax: T = 0;
                var col: usize = row_idx + 1;
                while (col < n) : (col += 1) {
                    sum_ax += A_copy.data[row_idx * n + col] * x.data[col * x.strides[0] + j * x.strides[1]];
                }
                x.data[row_idx * x.strides[0] + j * x.strides[1]] = (x.data[row_idx * x.strides[0] + j * x.strides[1]] - sum_ax) / pivot;
            }
        }
    }

    return x;
}

/// Compute the multiplicative inverse of a matrix.
///
/// Logic:
/// A * A^-1 = I
///
/// Arguments:
///     T: The data type of the array elements.
///     allocator: The allocator to use for the result array.
///     a: The input square matrix.
///
/// Returns:
///     A new NDArray containing the inverse of the matrix.
///
/// Example:
/// ```zig
/// var a = try NDArray(f32).init(allocator, &.{2, 2}, &.{4.0, 7.0, 2.0, 6.0});
/// defer a.deinit();
///
/// var inv = try linalg.inverse(f32, allocator, &a);
/// defer inv.deinit();
/// // inv is {{0.6, -0.7}, {-0.2, 0.4}}
/// ```
pub fn inverse(comptime T: type, allocator: Allocator, a: *const NDArray(T)) !NDArray(T) {
    if (a.rank() != 2) return core.Error.RankMismatch;
    if (a.shape[0] != a.shape[1]) return core.Error.DimensionMismatch;

    const n = a.shape[0];
    var identity = try NDArray(T).eye(allocator, n);
    defer identity.deinit();

    return solve(T, allocator, a, &identity);
}

/// Compute the determinant of an array.
///
/// Logic:
/// result = det(a)
///
/// Arguments:
///     T: The data type of the array elements.
///     allocator: The allocator to use for temporary storage.
///     a: The input square matrix.
///
/// Returns:
///     The determinant of the matrix.
///
/// Example:
/// ```zig
/// var a = try NDArray(f32).init(allocator, &.{2, 2}, &.{1.0, 2.0, 3.0, 4.0});
/// defer a.deinit();
///
/// const det = try linalg.determinant(f32, allocator, &a);
/// // det is -2.0
/// ```
pub fn determinant(comptime T: type, allocator: Allocator, a: *const NDArray(T)) !T {
    _ = allocator;
    if (a.rank() != 2) return core.Error.RankMismatch;
    if (a.shape[0] != a.shape[1]) return core.Error.DimensionMismatch;

    const n = a.shape[0];
    var temp = try a.copy();
    defer temp.deinit();

    var det: T = 1;

    // Gaussian elimination to upper triangular
    var i: usize = 0;
    while (i < n) : (i += 1) {
        var max_row = i;
        var max_val = @abs(temp.data[i * n + i]);

        var k: usize = i + 1;
        while (k < n) : (k += 1) {
            const val = @abs(temp.data[k * n + i]);
            if (val > max_val) {
                max_val = val;
                max_row = k;
            }
        }

        if (max_row != i) {
            // Swap rows
            var j: usize = i;
            while (j < n) : (j += 1) {
                const t = temp.data[i * n + j];
                temp.data[i * n + j] = temp.data[max_row * n + j];
                temp.data[max_row * n + j] = t;
            }
            det = -det;
        }

        const pivot = temp.data[i * n + i];
        if (@abs(pivot) < 1e-9) return 0;
        det *= pivot;

        var row: usize = i + 1;
        while (row < n) : (row += 1) {
            const factor = temp.data[row * n + i] / pivot;
            var col: usize = i + 1;
            while (col < n) : (col += 1) {
                temp.data[row * n + col] -= factor * temp.data[i * n + col];
            }
        }
    }

    return det;
}

/// Matrix or vector norm.
/// Currently supports Frobenius norm for matrices and L2 norm for vectors.
///
/// Logic:
/// result = sqrt(sum(x^2))
///
/// Arguments:
///     T: The data type of the array elements.
///     allocator: The allocator to use for iteration.
///     a: The input array.
///
/// Returns:
///     The norm of the array.
///
/// Example:
/// ```zig
/// var a = try NDArray(f32).init(allocator, &.{2}, &.{3.0, 4.0});
/// defer a.deinit();
///
/// const n = try linalg.norm(f32, allocator, &a);
/// // n is 5.0
/// ```
pub fn norm(comptime T: type, allocator: Allocator, a: *const NDArray(T)) !T {
    var sum_sq: T = 0;

    // Use iterator to handle non-contiguous arrays
    var iter = try core.NdIterator.init(allocator, a.shape);
    defer iter.deinit();

    while (iter.next()) |coords| {
        const val = try a.get(coords);
        sum_sq += val * val;
    }

    return @sqrt(sum_sq);
}

/// QR decomposition.
/// Computes the QR decomposition of a matrix using Modified Gram-Schmidt.
/// Returns a struct containing Q and R.
///
/// Logic:
/// A = Q * R
///
/// Arguments:
///     T: The data type of the array elements.
///     allocator: The allocator to use for the result arrays.
///     a: The input matrix.
///
/// Returns:
///     A struct containing Q (orthogonal matrix) and R (upper triangular matrix).
///
/// Example:
/// ```zig
/// var qr_res = try linalg.qr(f32, allocator, &a);
/// defer qr_res.q.deinit();
/// defer qr_res.r.deinit();
/// ```
pub fn qr(comptime T: type, allocator: Allocator, a: *const NDArray(T)) !struct { q: NDArray(T), r: NDArray(T) } {
    if (a.rank() != 2) return core.Error.RankMismatch;
    const m = a.shape[0];
    const n = a.shape[1];

    const k = @min(m, n);

    var Q_res = try NDArray(T).zeros(allocator, &.{ m, k });
    errdefer Q_res.deinit();
    var R_res = try NDArray(T).zeros(allocator, &.{ k, n });
    errdefer R_res.deinit();

    // Extract column i of A into v
    for (0..k) |i| {
        var v = try allocator.alloc(T, m);
        defer allocator.free(v);
        for (0..m) |row| {
            v[row] = try a.get(&.{ row, i });
        }

        for (0..i) |j_idx| {
            // Dot product q[:, j_idx] . v
            var dot_val: T = 0;
            for (0..m) |row| {
                const q_val = try Q_res.get(&.{ row, j_idx });
                dot_val += q_val * v[row];
            }

            try R_res.set(&.{ j_idx, i }, dot_val);

            // v = v - dot_val * q[:, j_idx]
            for (0..m) |row| {
                const q_val = try Q_res.get(&.{ row, j_idx });
                v[row] -= dot_val * q_val;
            }
        }

        // Norm of v
        var norm_sq: T = 0;
        for (v) |val| norm_sq += val * val;
        const v_norm = std.math.sqrt(norm_sq);

        try R_res.set(&.{ i, i }, v_norm);

        for (0..m) |row| {
            try Q_res.set(&.{ row, i }, v[row] / v_norm);
        }
    }

    // Fill the rest of R (if n > m)
    if (n > m) {
        for (m..n) |i| {
            // Project column i of A onto Q
            for (0..m) |j_idx| {
                // r[j_idx, i] = q[:, j_idx] . a[:, i]
                var dot_val: T = 0;
                for (0..m) |row| {
                    const q_val = try Q_res.get(&.{ row, j_idx });
                    const a_val = try a.get(&.{ row, i });
                    dot_val += q_val * a_val;
                }
                try R_res.set(&.{ j_idx, i }, dot_val);
            }
        }
    }

    return .{ .q = Q_res, .r = R_res };
}

test "linalg solve system" {
    const allocator = std.testing.allocator;
    // 2x + y = 5
    // x + y = 3
    // Solution: x=2, y=1
    var a = try NDArray(f32).init(allocator, &.{ 2, 2 });
    defer a.deinit();
    try a.set(&.{ 0, 0 }, 2);
    try a.set(&.{ 0, 1 }, 1);
    try a.set(&.{ 1, 0 }, 1);
    try a.set(&.{ 1, 1 }, 1);

    var b = try NDArray(f32).init(allocator, &.{2});
    defer b.deinit();
    try b.set(&.{0}, 5);
    try b.set(&.{1}, 3);

    var x = try solve(f32, allocator, &a, &b);
    defer x.deinit();

    try std.testing.expectApproxEqAbs((try x.get(&.{0})), 2.0, 1e-4);
    try std.testing.expectApproxEqAbs((try x.get(&.{1})), 1.0, 1e-4);
}

/// Computes the Cholesky decomposition of a matrix.
/// Returns the lower triangular matrix L such that A = L * L^T.
/// The matrix A must be symmetric and positive definite.
///
/// Logic:
/// A = L * L^T
///
/// Arguments:
///     T: The data type of the array elements.
///     allocator: The allocator to use for the result array.
///     a: The input symmetric positive definite matrix.
///
/// Returns:
///     The lower triangular matrix L.
///
/// Example:
/// ```zig
/// var l = try linalg.cholesky(f32, allocator, &a);
/// defer l.deinit();
/// ```
pub fn cholesky(comptime T: type, allocator: Allocator, a: *const NDArray(T)) !NDArray(T) {
    if (a.rank() != 2) return core.Error.RankMismatch;
    if (a.shape[0] != a.shape[1]) return core.Error.ShapeMismatch;

    const n = a.shape[0];
    var L = try NDArray(T).zeros(allocator, &.{ n, n });
    errdefer L.deinit();

    for (0..n) |i| {
        for (0..i + 1) |j| {
            var sum: T = 0;
            for (0..j) |k| {
                const l_ik = try L.get(&.{ i, k });
                const l_jk = try L.get(&.{ j, k });
                sum += l_ik * l_jk;
            }

            if (i == j) {
                const a_ii = try a.get(&.{ i, i });
                const val = a_ii - sum;
                if (val <= 0) return error.MatrixNotPositiveDefinite;
                try L.set(&.{ i, j }, std.math.sqrt(val));
            } else {
                const a_ij = try a.get(&.{ i, j });
                const l_jj = try L.get(&.{ j, j });
                try L.set(&.{ i, j }, (a_ij - sum) / l_jj);
            }
        }
    }

    return L;
}

/// Computes the eigenvalues of a square matrix using the QR algorithm.
/// Returns a 1D array of eigenvalues.
/// Note: This implementation assumes real eigenvalues and may not converge for all matrices.
///
/// Logic:
/// A * v = lambda * v
///
/// Arguments:
///     T: The data type of the array elements.
///     allocator: The allocator to use for the result array.
///     a: The input square matrix.
///     max_iter: The maximum number of iterations for the QR algorithm.
///
/// Returns:
///     A 1D array containing the eigenvalues.
///
/// Example:
/// ```zig
/// var vals = try linalg.eigvals(f32, allocator, &a, 100);
/// defer vals.deinit();
/// ```
pub fn eigvals(comptime T: type, allocator: Allocator, a: *const NDArray(T), max_iter: usize) !NDArray(T) {
    if (a.rank() != 2) return core.Error.RankMismatch;
    if (a.shape[0] != a.shape[1]) return core.Error.ShapeMismatch;

    const n = a.shape[0];

    // Copy A
    var Ak = try a.copy();
    defer Ak.deinit();

    for (0..max_iter) |_| {
        var qr_res = try qr(T, allocator, &Ak);
        defer qr_res.q.deinit();
        defer qr_res.r.deinit();

        // Ak+1 = R * Q
        const next_Ak = try matmul(T, allocator, &qr_res.r, &qr_res.q);
        Ak.deinit();
        Ak = next_Ak;
    }

    var vals = try NDArray(T).init(allocator, &.{n});
    for (0..n) |i| {
        vals.data[i] = try Ak.get(&.{ i, i });
    }

    return vals;
}

pub fn EigResult(comptime T: type) type {
    return struct {
        vals: NDArray(T),
        vecs: NDArray(T),

        const Self = @This();

        pub fn deinit(self: *Self) void {
            self.vals.deinit();
            self.vecs.deinit();
        }
    };
}

/// Computes the eigenvalues and eigenvectors of a square matrix.
/// Returns a struct containing eigenvalues and eigenvectors.
/// Note: This implementation assumes real eigenvalues.
///
/// Logic:
/// A * v = lambda * v
///
/// Arguments:
///     T: The data type of the array elements.
///     allocator: The allocator to use for the result arrays.
///     a: The input square matrix.
///     max_iter: The maximum number of iterations.
///
/// Returns:
///     A struct containing eigenvalues (vals) and eigenvectors (vecs).
///
/// Example:
/// ```zig
/// var res = try linalg.eig(f32, allocator, &a, 100);
/// defer res.deinit();
/// ```
pub fn eig(comptime T: type, allocator: Allocator, a: *const NDArray(T), max_iter: usize) !EigResult(T) {
    if (a.rank() != 2) return core.Error.RankMismatch;
    if (a.shape[0] != a.shape[1]) return core.Error.ShapeMismatch;

    const n = a.shape[0];

    // Copy A
    var Ak = try a.copy();
    defer Ak.deinit();

    // Initialize V as identity
    var V = try NDArray(T).eye(allocator, n);
    errdefer V.deinit();

    for (0..max_iter) |_| {
        var qr_res = try qr(T, allocator, &Ak);
        defer qr_res.q.deinit();
        defer qr_res.r.deinit();

        // Ak+1 = R * Q
        const next_Ak = try matmul(T, allocator, &qr_res.r, &qr_res.q);
        Ak.deinit();
        Ak = next_Ak;

        // V = V * Q
        const next_V = try matmul(T, allocator, &V, &qr_res.q);
        V.deinit();
        V = next_V;
    }

    var vals = try NDArray(T).init(allocator, &.{n});
    for (0..n) |i| {
        vals.data[i] = try Ak.get(&.{ i, i });
    }

    return EigResult(T){ .vals = vals, .vecs = V };
}

pub fn SVDResult(comptime T: type) type {
    return struct {
        u: NDArray(T),
        s: NDArray(T),
        vt: NDArray(T),

        const Self = @This();

        pub fn deinit(self: *Self) void {
            self.u.deinit();
            self.s.deinit();
            self.vt.deinit();
        }
    };
}

/// Computes the Singular Value Decomposition of a matrix.
/// A = U * S * Vt
/// Note: This is a simplified implementation using A^T * A.
pub fn svd(comptime T: type, allocator: Allocator, a: *const NDArray(T), max_iter: usize) !SVDResult(T) {
    if (a.rank() != 2) return core.Error.RankMismatch;

    // B = A^T * A
    var at = try a.transpose();
    defer at.deinit();

    var b = try matmul(T, allocator, &at, a);
    defer b.deinit();

    // Eig of B
    var eig_res = try eig(T, allocator, &b, max_iter);
    defer eig_res.deinit();

    // Singular values are sqrt of eigenvalues
    var s = try NDArray(T).init(allocator, eig_res.vals.shape);
    errdefer s.deinit();
    for (s.data, 0..) |*val, i| {
        val.* = std.math.sqrt(@abs(eig_res.vals.data[i]));
    }

    // V is eigenvectors of A^T A
    // Vt is transpose of V
    var vt = try eig_res.vecs.transpose();
    errdefer vt.deinit();

    // U = A * V * S^-1
    // S^-1
    var s_inv = try NDArray(T).init(allocator, s.shape);
    defer s_inv.deinit();
    for (s_inv.data, 0..) |*val, i| {
        if (s.data[i] > 1e-6) {
            val.* = 1.0 / s.data[i];
        } else {
            val.* = 0.0;
        }
    }

    // S^-1 as diagonal matrix
    var s_inv_mat = try NDArray(T).zeros(allocator, &.{ s.size(), s.size() });
    defer s_inv_mat.deinit();
    for (0..s.size()) |i| {
        try s_inv_mat.set(&.{ i, i }, s_inv.data[i]);
    }

    // A * V
    var av = try matmul(T, allocator, a, &eig_res.vecs);
    defer av.deinit();

    // U = (A * V) * S^-1
    const u = try matmul(T, allocator, &av, &s_inv_mat);

    return SVDResult(T){ .u = u, .s = s, .vt = vt };
}

/// Computes the LU decomposition of a matrix.
///
/// Arguments:
///     T: The data type.
///     allocator: The allocator.
///     a: The input matrix (must be square).
///
/// Returns:
///     A struct containing L and U matrices.
pub fn lu(comptime T: type, allocator: Allocator, a: *const NDArray(T)) !struct { l: NDArray(T), u: NDArray(T) } {
    if (a.rank() != 2) return core.Error.RankMismatch;
    if (a.shape[0] != a.shape[1]) return core.Error.ShapeMismatch;

    const n = a.shape[0];
    var L = try NDArray(T).eye(allocator, n);
    errdefer L.deinit();
    var U = try NDArray(T).zeros(allocator, &.{ n, n });
    errdefer U.deinit();

    // Doolittle Algorithm
    for (0..n) |i| {
        // Upper Triangular
        for (i..n) |k| {
            var sum: T = 0;
            for (0..i) |j| {
                sum += (try L.get(&.{ i, j })) * (try U.get(&.{ j, k }));
            }
            try U.set(&.{ i, k }, (try a.get(&.{ i, k })) - sum);
        }

        // Lower Triangular
        for (i..n) |k| {
            if (i == k) {
                try L.set(&.{ i, i }, 1);
            } else {
                var sum: T = 0;
                for (0..i) |j| {
                    sum += (try L.get(&.{ k, j })) * (try U.get(&.{ j, i }));
                }
                const u_ii = try U.get(&.{ i, i });
                if (u_ii == 0) return core.Error.SingularMatrix;
                try L.set(&.{ k, i }, ((try a.get(&.{ k, i })) - sum) / u_ii);
            }
        }
    }

    return .{ .l = L, .u = U };
}
