//! Matrix decompositions: LU, QR, and Cholesky.
//!
//! Provides fundamental linear algebra factorizations for square and rectangular matrices:
//! - LU with partial pivoting (PA = LU)
//! - QR factorization via Householder reflections (A = QR)
//! - Cholesky factorization for symmetric positive-definite matrices (A = L L^T)

const std = @import("std");
const Array = @import("../core/array.zig").Array;
const zeros = @import("../core/array.zig").zeros;
const eye = @import("../core/array.zig").eye;
const Shape = @import("../core/shape.zig").Shape;
const ShapeError = @import("../core/error.zig").ShapeError;
const LinalgError = @import("../core/error.zig").LinalgError;
const DTypeError = @import("../core/error.zig").DTypeError;
const IndexError = @import("../core/error.zig").IndexError;
const DType = @import("../core/dtype.zig").DType;

pub const LuResult = struct {
    p: Array,
    l: Array,
    u: Array,

    pub fn deinit(self: *LuResult) void {
        self.p.deinit();
        self.l.deinit();
        self.u.deinit();
    }
};

pub const QrResult = struct {
    q: Array,
    r: Array,

    pub fn deinit(self: *QrResult) void {
        self.q.deinit();
        self.r.deinit();
    }
};

/// Computes the LU decomposition with partial pivoting: P * A = L * U.
pub fn lu(a: Array) (ShapeError || LinalgError || DTypeError || IndexError || std.mem.Allocator.Error)!LuResult {
    const s = a.shape();
    if (s.ndim != 2) return ShapeError.InvalidDimension;
    if (s.dims[0] != s.dims[1]) return LinalgError.MatrixNotSquare;

    const n = s.dims[0];
    const float_dtype: DType = if (a.dtype == .f32) .f32 else .f64;

    var p = try eye(a.allocator, .{ .n = n, .dtype = float_dtype });
    errdefer p.deinit();

    var l = try eye(a.allocator, .{ .n = n, .dtype = float_dtype });
    errdefer l.deinit();

    var u = try zeros(a.allocator, .{ .shape = &.{ n, n }, .dtype = float_dtype });
    errdefer u.deinit();

    // Copy A into U
    for (0..n) |i| {
        for (0..n) |j| {
            const val = try a.get(f64, &.{ i, j });
            try u.set(f64, &.{ i, j }, val);
        }
    }

    for (0..n) |k| {
        // Find pivot in column k
        var pivot_row: usize = k;
        var max_val = @abs(try u.get(f64, &.{ k, k }));

        for (k + 1..n) |r| {
            const val = @abs(try u.get(f64, &.{ r, k }));
            if (val > max_val) {
                max_val = val;
                pivot_row = r;
            }
        }

        if (max_val < 1e-15) return LinalgError.SingularMatrix;

        // Swap rows in U, P, and already-computed elements of L
        if (pivot_row != k) {
            for (0..n) |j| {
                const u_k = try u.get(f64, &.{ k, j });
                const u_p = try u.get(f64, &.{ pivot_row, j });
                try u.set(f64, &.{ k, j }, u_p);
                try u.set(f64, &.{ pivot_row, j }, u_k);

                const p_k = try p.get(f64, &.{ k, j });
                const p_p = try p.get(f64, &.{ pivot_row, j });
                try p.set(f64, &.{ k, j }, p_p);
                try p.set(f64, &.{ pivot_row, j }, p_k);
            }

            for (0..k) |j| {
                const l_k = try l.get(f64, &.{ k, j });
                const l_p = try l.get(f64, &.{ pivot_row, j });
                try l.set(f64, &.{ k, j }, l_p);
                try l.set(f64, &.{ pivot_row, j }, l_k);
            }
        }

        // Elimination
        const u_kk = try u.get(f64, &.{ k, k });
        for (k + 1..n) |r| {
            const factor = (try u.get(f64, &.{ r, k })) / u_kk;
            try l.set(f64, &.{ r, k }, factor);
            try u.set(f64, &.{ r, k }, 0.0);

            for (k + 1..n) |c| {
                const cur = try u.get(f64, &.{ r, c });
                const top = try u.get(f64, &.{ k, c });
                try u.set(f64, &.{ r, c }, cur - factor * top);
            }
        }
    }

    return LuResult{ .p = p, .l = l, .u = u };
}

/// Computes the QR decomposition: A = Q * R using Householder reflections.
pub fn qr(a: Array) (ShapeError || LinalgError || DTypeError || IndexError || std.mem.Allocator.Error)!QrResult {
    const s = a.shape();
    if (s.ndim != 2) return ShapeError.InvalidDimension;

    const m = s.dims[0];
    const n = s.dims[1];
    const k_min = @min(m, n);
    const float_dtype: DType = if (a.dtype == .f32) .f32 else .f64;

    var q = try eye(a.allocator, .{ .n = m, .dtype = float_dtype });
    errdefer q.deinit();

    var r = try zeros(a.allocator, .{ .shape = &.{ m, n }, .dtype = float_dtype });
    errdefer r.deinit();

    // Copy A into R
    for (0..m) |i| {
        for (0..n) |j| {
            const val = try a.get(f64, &.{ i, j });
            try r.set(f64, &.{ i, j }, val);
        }
    }

    // Temporary vector for Householder reflection
    var v = try a.allocator.alloc(f64, m);
    defer a.allocator.free(v);

    for (0..k_min) |k| {
        var norm_sq: f64 = 0;
        for (k..m) |i| {
            const val = try r.get(f64, &.{ i, k });
            v[i] = val;
            norm_sq += val * val;
        }

        const alpha = std.math.sqrt(norm_sq);
        if (alpha < 1e-15) continue;

        const sign: f64 = if (v[k] >= 0) 1.0 else -1.0;
        v[k] += sign * alpha;

        // Normalize v
        var v_norm_sq: f64 = 0;
        for (k..m) |i| {
            v_norm_sq += v[i] * v[i];
        }

        if (v_norm_sq < 1e-15) continue;
        const beta = 2.0 / v_norm_sq;

        // Apply Householder matrix H = I - beta * v * v^T to R: R = H * R
        for (k..n) |j| {
            var dot_prod: f64 = 0;
            for (k..m) |i| {
                dot_prod += v[i] * (try r.get(f64, &.{ i, j }));
            }
            for (k..m) |i| {
                const cur = try r.get(f64, &.{ i, j });
                try r.set(f64, &.{ i, j }, cur - beta * dot_prod * v[i]);
            }
        }

        // Apply Householder matrix to Q: Q = Q * H
        for (0..m) |i| {
            var dot_prod: f64 = 0;
            for (k..m) |j| {
                dot_prod += (try q.get(f64, &.{ i, j })) * v[j];
            }
            for (k..m) |j| {
                const cur = try q.get(f64, &.{ i, j });
                try q.set(f64, &.{ i, j }, cur - beta * dot_prod * v[j]);
            }
        }
    }

    return QrResult{ .q = q, .r = r };
}

/// Computes the Cholesky decomposition of a symmetric positive-definite matrix: A = L * L^T.
pub fn cholesky(a: Array) (ShapeError || LinalgError || DTypeError || IndexError || std.mem.Allocator.Error)!Array {
    const s = a.shape();
    if (s.ndim != 2) return ShapeError.InvalidDimension;
    if (s.dims[0] != s.dims[1]) return LinalgError.MatrixNotSquare;

    const n = s.dims[0];
    const float_dtype: DType = if (a.dtype == .f32) .f32 else .f64;

    var l = try zeros(a.allocator, .{ .shape = &.{ n, n }, .dtype = float_dtype });
    errdefer l.deinit();

    for (0..n) |i| {
        for (0..i + 1) |j| {
            var sum_val: f64 = 0;
            for (0..j) |k| {
                const l_ik = try l.get(f64, &.{ i, k });
                const l_jk = try l.get(f64, &.{ j, k });
                sum_val += l_ik * l_jk;
            }

            const a_ij = try a.get(f64, &.{ i, j });

            if (i == j) {
                const diff = a_ij - sum_val;
                if (diff <= 0) return LinalgError.NotPositiveDefinite;
                try l.set(f64, &.{ i, j }, std.math.sqrt(diff));
            } else {
                const l_jj = try l.get(f64, &.{ j, j });
                if (@abs(l_jj) < 1e-15) return LinalgError.NotPositiveDefinite;
                try l.set(f64, &.{ i, j }, (a_ij - sum_val) / l_jj);
            }
        }
    }

    return l;
}

test "lu, qr, and cholesky decompositions" {
    const allocator = std.testing.allocator;
    const fromSlice = @import("../core/array.zig").fromSlice;
    const matmul = @import("matmul.zig").matmul;

    // Test LU decomposition on 3x3
    const data_a = [_]f64{
        2, 1, 1,
        4, 3, 3,
        8, 7, 9,
    };
    var a = try fromSlice(allocator, f64, .{ .data = &data_a, .shape = &.{ 3, 3 } });
    defer a.deinit();

    var lu_res = try lu(a);
    defer lu_res.deinit();

    // Verify P * A = L * U
    var pa = try matmul(lu_res.p, a, .{});
    defer pa.deinit();

    var lu_prod = try matmul(lu_res.l, lu_res.u, .{});
    defer lu_prod.deinit();

    for (0..3) |i| {
        for (0..3) |j| {
            try std.testing.expectApproxEqAbs(try pa.get(f64, &.{ i, j }), try lu_prod.get(f64, &.{ i, j }), 1e-5);
        }
    }

    // Test QR decomposition on 3x3
    var qr_res = try qr(a);
    defer qr_res.deinit();

    var qr_prod = try matmul(qr_res.q, qr_res.r, .{});
    defer qr_prod.deinit();

    for (0..3) |i| {
        for (0..3) |j| {
            try std.testing.expectApproxEqAbs(try a.get(f64, &.{ i, j }), try qr_prod.get(f64, &.{ i, j }), 1e-5);
        }
    }

    // Test Cholesky on symmetric positive-definite 2x2: [[4, 2], [2, 5]]
    // L should be [[2, 0], [1, 2]]
    const spd_data = [_]f64{ 4, 2, 2, 5 };
    var spd = try fromSlice(allocator, f64, .{ .data = &spd_data, .shape = &.{ 2, 2 } });
    defer spd.deinit();

    var chol = try cholesky(spd);
    defer chol.deinit();

    try std.testing.expectApproxEqAbs(@as(f64, 2.0), try chol.get(f64, &.{ 0, 0 }), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), try chol.get(f64, &.{ 0, 1 }), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), try chol.get(f64, &.{ 1, 0 }), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), try chol.get(f64, &.{ 1, 1 }), 1e-6);
}
