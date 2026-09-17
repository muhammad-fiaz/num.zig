//! Linear system solvers, matrix inversion, and determinant computation.
//!
//! Solves Ax = b using LU decomposition with partial pivoting and forward/backward substitution,
//! computes matrix inverses via identity solves, and evaluates determinants.

const std = @import("std");
const Array = @import("../core/array.zig").Array;
const zeros = @import("../core/array.zig").zeros;
const empty = @import("../core/array.zig").empty;
const eye = @import("../core/array.zig").eye;
const Shape = @import("../core/shape.zig").Shape;
const ShapeError = @import("../core/error.zig").ShapeError;
const LinalgError = @import("../core/error.zig").LinalgError;
const DTypeError = @import("../core/error.zig").DTypeError;
const IndexError = @import("../core/error.zig").IndexError;
const DType = @import("../core/dtype.zig").DType;
const lu = @import("decompose.zig").lu;

/// Solves a linear system of equations: A * x = b.
pub fn solve(a: Array, b: Array) (ShapeError || LinalgError || DTypeError || IndexError || std.mem.Allocator.Error)!Array {
    const s_a = a.shape();
    const s_b = b.shape();

    if (s_a.ndim != 2) return ShapeError.InvalidDimension;
    if (s_a.dims[0] != s_a.dims[1]) return LinalgError.MatrixNotSquare;

    const n = s_a.dims[0];
    if (s_b.dims[0] != n) return LinalgError.IncompatibleDimensions;

    const is_b_1d = s_b.ndim == 1;
    const n_rhs: usize = if (is_b_1d) 1 else s_b.dims[1];

    var lu_res = try lu(a);
    defer lu_res.deinit();

    const float_dtype: DType = if (a.dtype == .f32) .f32 else .f64;

    // Apply permutation P to b: Pb = P * b
    var pb = try zeros(a.allocator, .{ .shape = &.{ n, n_rhs }, .dtype = float_dtype });
    defer pb.deinit();

    for (0..n) |i| {
        for (0..n) |j| {
            const p_ij = try lu_res.p.get(f64, &.{ i, j });
            if (p_ij != 0) {
                for (0..n_rhs) |col| {
                    const b_val = if (is_b_1d)
                        try b.get(f64, &.{j})
                    else
                        try b.get(f64, &.{ j, col });
                    try pb.set(f64, &.{ i, col }, b_val);
                }
                break;
            }
        }
    }

    // Forward substitution: L * y = Pb
    var y = try zeros(a.allocator, .{ .shape = &.{ n, n_rhs }, .dtype = float_dtype });
    defer y.deinit();

    for (0..n) |i| {
        for (0..n_rhs) |col| {
            var sum: f64 = 0;
            for (0..i) |j| {
                const l_ij = try lu_res.l.get(f64, &.{ i, j });
                const y_jc = try y.get(f64, &.{ j, col });
                sum += l_ij * y_jc;
            }
            const pb_val = try pb.get(f64, &.{ i, col });
            try y.set(f64, &.{ i, col }, pb_val - sum);
        }
    }

    // Back substitution: U * x = y
    var x = if (is_b_1d)
        try zeros(a.allocator, .{ .shape = &.{n}, .dtype = float_dtype })
    else
        try zeros(a.allocator, .{ .shape = &.{ n, n_rhs }, .dtype = float_dtype });
    errdefer x.deinit();

    var i: usize = n;
    while (i > 0) {
        i -= 1;
        const u_ii = try lu_res.u.get(f64, &.{ i, i });
        if (@abs(u_ii) < 1e-15) return LinalgError.SingularMatrix;

        for (0..n_rhs) |col| {
            var sum: f64 = 0;
            for (i + 1..n) |j| {
                const u_ij = try lu_res.u.get(f64, &.{ i, j });
                const x_jc = if (is_b_1d)
                    try x.get(f64, &.{j})
                else
                    try x.get(f64, &.{ j, col });
                sum += u_ij * x_jc;
            }
            const y_ic = try y.get(f64, &.{ i, col });
            const x_val = (y_ic - sum) / u_ii;

            if (is_b_1d) {
                try x.set(f64, &.{i}, x_val);
            } else {
                try x.set(f64, &.{ i, col }, x_val);
            }
        }
    }

    return x;
}

/// Solves a triangular system: A * x = b where A is upper or lower triangular.
/// No decomposition is performed; forward/backward substitution runs directly.
pub fn solveTriangular(
    a: Array,
    b: Array,
    options: struct { lower: bool = true },
) (ShapeError || LinalgError || DTypeError || IndexError || std.mem.Allocator.Error)!Array {
    const s_a = a.shape();
    const s_b = b.shape();
    if (s_a.ndim != 2) return ShapeError.InvalidDimension;
    if (s_a.dims[0] != s_a.dims[1]) return LinalgError.MatrixNotSquare;
    const n = s_a.dims[0];
    if (s_b.dims[0] != n) return LinalgError.IncompatibleDimensions;

    const is_b_1d = s_b.ndim == 1;
    const n_rhs: usize = if (is_b_1d) 1 else s_b.dims[1];
    const float_dtype: DType = if (a.dtype == .f32) .f32 else .f64;

    var x = if (is_b_1d)
        try zeros(a.allocator, .{ .shape = &.{n}, .dtype = float_dtype })
    else
        try zeros(a.allocator, .{ .shape = &.{ n, n_rhs }, .dtype = float_dtype });
    errdefer x.deinit();

    if (options.lower) {
        for (0..n) |i| {
            const a_ii = try a.getAsFloat(&.{ i, i });
            if (@abs(a_ii) < 1e-15) return LinalgError.SingularMatrix;
            for (0..n_rhs) |col| {
                var sum: f64 = 0;
                for (0..i) |j| {
                    const a_ij = try a.getAsFloat(&.{ i, j });
                    const x_jc = if (is_b_1d) try x.getAsFloat(&.{j}) else try x.getAsFloat(&.{ j, col });
                    sum += a_ij * x_jc;
                }
                const b_ic = if (is_b_1d) try b.getAsFloat(&.{i}) else try b.getAsFloat(&.{ i, col });
                const val = (b_ic - sum) / a_ii;
                if (is_b_1d) try x.setFromFloat(&.{i}, val) else try x.setFromFloat(&.{ i, col }, val);
            }
        }
    } else {
        var i: usize = n;
        while (i > 0) {
            i -= 1;
            const a_ii = try a.getAsFloat(&.{ i, i });
            if (@abs(a_ii) < 1e-15) return LinalgError.SingularMatrix;
            for (0..n_rhs) |col| {
                var sum: f64 = 0;
                for (i + 1..n) |j| {
                    const a_ij = try a.getAsFloat(&.{ i, j });
                    const x_jc = if (is_b_1d) try x.getAsFloat(&.{j}) else try x.getAsFloat(&.{ j, col });
                    sum += a_ij * x_jc;
                }
                const b_ic = if (is_b_1d) try b.getAsFloat(&.{i}) else try b.getAsFloat(&.{ i, col });
                const val = (b_ic - sum) / a_ii;
                if (is_b_1d) try x.setFromFloat(&.{i}, val) else try x.setFromFloat(&.{ i, col }, val);
            }
        }
    }
    return x;
}

/// Solves a symmetric positive-definite system via Cholesky factorization.
/// Reuses the shared Cholesky and triangular-substitution kernels.
pub fn solveSpd(a: Array, b: Array) (ShapeError || LinalgError || DTypeError || IndexError || std.mem.Allocator.Error)!Array {
    const cholesky = @import("decompose.zig").cholesky;
    const transpose = @import("../manip/transpose.zig").transpose;
    var l = try cholesky(a);
    defer l.deinit();
    var y = try solveTriangular(l, b, .{ .lower = true });
    defer y.deinit();
    var lt = try transpose(l, .{});
    defer lt.deinit();
    return solveTriangular(lt, y, .{ .lower = false });
}

/// Least-squares solution to an overdetermined system: min ||A x - b||.
/// Solves the normal equations (A^T A) x = A^T b reusing transpose, matmul, and LU solve.
pub fn lstsq(a: Array, b: Array) (ShapeError || LinalgError || DTypeError || IndexError || std.mem.Allocator.Error)!Array {
    const transpose = @import("../manip/transpose.zig").transpose;
    const matmul = @import("matmul.zig").matmul;
    const s_a = a.shape();
    if (s_a.ndim != 2) return ShapeError.InvalidDimension;
    var at = try transpose(a, .{});
    defer at.deinit();
    var ata = try matmul(at, a, .{});
    defer ata.deinit();
    var atb = try matmul(at, b, .{});
    defer atb.deinit();
    return solve(ata, atb);
}

/// Computes the multiplicative inverse of a square matrix.
pub fn inv(a: Array) (ShapeError || LinalgError || DTypeError || IndexError || std.mem.Allocator.Error)!Array {
    const s = a.shape();
    if (s.ndim != 2) return ShapeError.InvalidDimension;
    if (s.dims[0] != s.dims[1]) return LinalgError.MatrixNotSquare;

    const n = s.dims[0];
    const float_dtype: DType = if (a.dtype == .f32) .f32 else .f64;

    var ident = try eye(a.allocator, .{ .n = n, .dtype = float_dtype });
    defer ident.deinit();

    return solve(a, ident);
}

/// Computes the determinant of a square matrix. Returns a 0D scalar array.
pub fn det(a: Array) (ShapeError || LinalgError || DTypeError || IndexError || std.mem.Allocator.Error)!Array {
    const s = a.shape();
    if (s.ndim != 2) return ShapeError.InvalidDimension;
    if (s.dims[0] != s.dims[1]) return LinalgError.MatrixNotSquare;

    const n = s.dims[0];
    const float_dtype: DType = if (a.dtype == .f32) .f32 else .f64;

    if (n == 0) {
        var res = try empty(a.allocator, .{ .shape = &.{}, .dtype = float_dtype });
        try res.set(f64, &.{}, 1.0);
        return res;
    }

    var lu_res = lu(a) catch |err| switch (err) {
        LinalgError.SingularMatrix => {
            var res = try empty(a.allocator, .{ .shape = &.{}, .dtype = float_dtype });
            try res.set(f64, &.{}, 0.0);
            return res;
        },
        else => return err,
    };
    defer lu_res.deinit();

    // Count row swaps in P
    var num_swaps: usize = 0;
    for (0..n) |i| {
        for (0..n) |j| {
            if ((try lu_res.p.get(f64, &.{ i, j })) != 0) {
                if (i != j) num_swaps += 1;
                break;
            }
        }
    }
    // Each swap affects two rows, so number of transpositions is swaps / 2
    const sign: f64 = if ((num_swaps / 2) % 2 == 1) -1.0 else 1.0;

    var det_val: f64 = sign;
    for (0..n) |i| {
        det_val *= try lu_res.u.get(f64, &.{ i, i });
    }

    var out = try empty(a.allocator, .{ .shape = &.{}, .dtype = float_dtype });
    try out.set(f64, &.{}, det_val);
    return out;
}

pub const SlogdetResult = struct {
    sign: Array,
    logabsdet: Array,

    pub fn deinit(self: *SlogdetResult) void {
        self.sign.deinit();
        self.logabsdet.deinit();
    }
};

/// Computes the sign and natural logarithm of the determinant of a square matrix.
pub fn slogdet(a: Array) (ShapeError || LinalgError || DTypeError || IndexError || std.mem.Allocator.Error)!SlogdetResult {
    const s = a.shape();
    if (s.ndim != 2) return ShapeError.InvalidDimension;
    if (s.dims[0] != s.dims[1]) return LinalgError.MatrixNotSquare;

    const n = s.dims[0];
    const float_dtype: DType = if (a.dtype == .f32) .f32 else .f64;

    if (n == 0) {
        var s_arr = try empty(a.allocator, .{ .shape = &.{}, .dtype = float_dtype });
        try s_arr.set(f64, &.{}, 1.0);
        var l_arr = try empty(a.allocator, .{ .shape = &.{}, .dtype = float_dtype });
        try l_arr.set(f64, &.{}, 0.0);
        return SlogdetResult{ .sign = s_arr, .logabsdet = l_arr };
    }

    var lu_res = lu(a) catch |err| switch (err) {
        LinalgError.SingularMatrix => {
            var s_arr = try empty(a.allocator, .{ .shape = &.{}, .dtype = float_dtype });
            try s_arr.set(f64, &.{}, 0.0);
            var l_arr = try empty(a.allocator, .{ .shape = &.{}, .dtype = float_dtype });
            try l_arr.set(f64, &.{}, -std.math.inf(f64));
            return SlogdetResult{ .sign = s_arr, .logabsdet = l_arr };
        },
        else => return err,
    };
    defer lu_res.deinit();

    var num_swaps: usize = 0;
    for (0..n) |i| {
        for (0..n) |j| {
            if ((try lu_res.p.get(f64, &.{ i, j })) != 0) {
                if (i != j) num_swaps += 1;
                break;
            }
        }
    }
    var sign_val: f64 = if ((num_swaps / 2) % 2 == 1) -1.0 else 1.0;
    var logabs_val: f64 = 0.0;

    for (0..n) |i| {
        const u_diag = try lu_res.u.get(f64, &.{ i, i });
        if (u_diag == 0.0) {
            sign_val = 0.0;
            logabs_val = -std.math.inf(f64);
            break;
        } else if (u_diag < 0.0) {
            sign_val = -sign_val;
            logabs_val += @log(-u_diag);
        } else {
            logabs_val += @log(u_diag);
        }
    }

    var sign_arr = try empty(a.allocator, .{ .shape = &.{}, .dtype = float_dtype });
    try sign_arr.set(f64, &.{}, sign_val);
    var logabs_arr = try empty(a.allocator, .{ .shape = &.{}, .dtype = float_dtype });
    try logabs_arr.set(f64, &.{}, logabs_val);

    return SlogdetResult{
        .sign = sign_arr,
        .logabsdet = logabs_arr,
    };
}

/// Computes the sum along diagonals of a 2D matrix. Returns a 0D scalar array.
pub fn trace(a: Array) (ShapeError || std.mem.Allocator.Error)!Array {
    const s = a.shape();
    if (s.ndim != 2) return ShapeError.InvalidDimension;
    const n = @min(s.dims[0], s.dims[1]);
    const float_dtype: DType = if (a.dtype == .f32) .f32 else .f64;

    var sum_val: f64 = 0.0;
    for (0..n) |i| {
        const val = a.getAsFloat(&.{ i, i }) catch unreachable;
        sum_val += val;
    }

    var out = try empty(a.allocator, .{ .shape = &.{}, .dtype = float_dtype });
    out.setFromFloat(&.{}, sum_val) catch unreachable;
    return out;
}

/// Computes the matrix rank via Gaussian elimination with row pivoting.
pub fn matrixRank(a: Array, options: struct { tol: ?f64 = null }) (ShapeError || std.mem.Allocator.Error)!usize {
    const s = a.shape();
    if (s.ndim != 2) return ShapeError.InvalidDimension;
    const rows = s.dims[0];
    const cols = s.dims[1];
    if (rows == 0 or cols == 0) return 0;

    var mat = try a.clone();
    defer mat.deinit();

    const tol = options.tol orelse 1e-10;
    var rank_count: usize = 0;
    var lead: usize = 0;

    for (0..rows) |r| {
        if (lead >= cols) break;
        var i = r;
        while (@abs(mat.getAsFloat(&.{ i, lead }) catch 0.0) < tol) {
            i += 1;
            if (i == rows) {
                i = r;
                lead += 1;
                if (lead == cols) return rank_count;
            }
        }

        // Swap rows i and r
        for (0..cols) |c| {
            const v1 = mat.getAsFloat(&.{ i, c }) catch 0.0;
            const v2 = mat.getAsFloat(&.{ r, c }) catch 0.0;
            mat.setFromFloat(&.{ i, c }, v2) catch unreachable;
            mat.setFromFloat(&.{ r, c }, v1) catch unreachable;
        }

        const div_val = mat.getAsFloat(&.{ r, lead }) catch 1.0;
        if (@abs(div_val) > tol) {
            for (0..cols) |c| {
                const cur = mat.getAsFloat(&.{ r, c }) catch 0.0;
                mat.setFromFloat(&.{ r, c }, cur / div_val) catch unreachable;
            }

            for (0..rows) |other_r| {
                if (other_r != r) {
                    const factor = mat.getAsFloat(&.{ other_r, lead }) catch 0.0;
                    for (0..cols) |c| {
                        const cur = mat.getAsFloat(&.{ other_r, c }) catch 0.0;
                        const pivot_val = mat.getAsFloat(&.{ r, c }) catch 0.0;
                        mat.setFromFloat(&.{ other_r, c }, cur - factor * pivot_val) catch unreachable;
                    }
                }
            }
            rank_count += 1;
        }
        lead += 1;
    }

    return rank_count;
}

/// Computes the matrix power A^n for square 2D matrix A and integer n.
pub fn matrixPower(a: Array, n: isize) (ShapeError || LinalgError || DTypeError || IndexError || std.mem.Allocator.Error)!Array {
    const s = a.shape();
    if (s.ndim != 2) return ShapeError.InvalidDimension;
    if (s.dims[0] != s.dims[1]) return LinalgError.MatrixNotSquare;
    const dim = s.dims[0];

    const matmul = @import("matmul.zig").matmul;

    if (n == 0) {
        return eye(a.allocator, .{ .n = dim, .dtype = a.dtype });
    }

    var base = if (n < 0) try inv(a) else try a.clone();
    defer base.deinit();

    var exp: usize = @intCast(@abs(n));
    var res = try eye(a.allocator, .{ .n = dim, .dtype = a.dtype });
    errdefer res.deinit();

    while (exp > 0) {
        if ((exp & 1) == 1) {
            const next_res = try matmul(res, base, .{});
            res.deinit();
            res = next_res;
        }
        exp >>= 1;
        if (exp > 0) {
            const next_base = try matmul(base, base, .{});
            base.deinit();
            base = next_base;
        }
    }

    return res;
}

/// Computes the Moore-Penrose pseudoinverse of a matrix using Singular Value Decomposition.
pub fn pinv(a: Array, options: struct { rcond: f64 = 1e-15 }) (ShapeError || LinalgError || DTypeError || IndexError || std.mem.Allocator.Error)!Array {
    const s = a.shape();
    if (s.ndim != 2) return ShapeError.InvalidDimension;
    const rows = s.dims[0];
    const cols = s.dims[1];

    const svd = @import("svd.zig").svd;
    const matmul = @import("matmul.zig").matmul;
    const transpose = @import("../manip/transpose.zig").transpose;

    var svd_res = try svd(a, .{});
    defer svd_res.deinit();

    const u = svd_res.u;
    const sigma = svd_res.s;
    const vt = svd_res.vt;

    const k = sigma.elementCount();
    if (k == 0) {
        return zeros(a.allocator, .{ .shape = &.{ cols, rows }, .dtype = a.dtype });
    }

    const s0 = try sigma.getAsFloat(&.{0});
    const cutoff = options.rcond * s0;

    // Construct Sigma^+ of shape [cols, rows]
    var s_plus = try zeros(a.allocator, .{ .shape = &.{ cols, rows }, .dtype = a.dtype });
    defer s_plus.deinit();

    for (0..k) |i| {
        const val = try sigma.getAsFloat(&.{i});
        if (val > cutoff) {
            try s_plus.setFromFloat(&.{ i, i }, 1.0 / val);
        }
    }

    // A^+ = V * Sigma^+ * U^T = Vt^T * Sigma^+ * U^T
    var v = try transpose(vt, .{});
    defer v.deinit();

    var ut = try transpose(u, .{});
    defer ut.deinit();

    var v_s_plus = try matmul(v, s_plus, .{});
    defer v_s_plus.deinit();

    return matmul(v_s_plus, ut, .{});
}

test "linear solve, inverse, and determinant" {
    const allocator = std.testing.allocator;
    const fromSlice = @import("../core/array.zig").fromSlice;
    const matmul = @import("matmul.zig").matmul;

    // 2x2 system:
    // 2x + y = 5
    // x + 3y = 5
    // Solution: x = 2, y = 1
    const a_data = [_]f64{ 2, 1, 1, 3 };
    var a = try fromSlice(allocator, f64, .{ .data = &a_data, .shape = &.{ 2, 2 } });
    defer a.deinit();

    const b_data = [_]f64{ 5, 5 };
    var b = try fromSlice(allocator, f64, .{ .data = &b_data, .shape = &.{2} });
    defer b.deinit();

    var x = try solve(a, b);
    defer x.deinit();

    try std.testing.expectApproxEqAbs(@as(f64, 2.0), try x.get(f64, &.{0}), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), try x.get(f64, &.{1}), 1e-6);

    // Matrix inverse: a * inv(a) = I
    var a_inv = try inv(a);
    defer a_inv.deinit();

    var prod = try matmul(a, a_inv, .{});
    defer prod.deinit();

    try std.testing.expectApproxEqAbs(@as(f64, 1.0), try prod.get(f64, &.{ 0, 0 }), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), try prod.get(f64, &.{ 0, 1 }), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), try prod.get(f64, &.{ 1, 0 }), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), try prod.get(f64, &.{ 1, 1 }), 1e-6);

    // Determinant of [[2, 1], [1, 3]] = 2*3 - 1*1 = 5.0
    var d = try det(a);
    defer d.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), try d.get(f64, &.{}), 1e-6);

    // Matrix power: a^2 = [[2, 1], [1, 3]] * [[2, 1], [1, 3]] = [[5, 5], [5, 10]]
    var a2 = try matrixPower(a, 2);
    defer a2.deinit();
    try std.testing.expectEqual(@as(f64, 5.0), try a2.get(f64, &.{ 0, 0 }));
    try std.testing.expectEqual(@as(f64, 5.0), try a2.get(f64, &.{ 0, 1 }));
    try std.testing.expectEqual(@as(f64, 5.0), try a2.get(f64, &.{ 1, 0 }));
    try std.testing.expectEqual(@as(f64, 10.0), try a2.get(f64, &.{ 1, 1 }));

    // Pinv: a * pinv(a) should be Identity for invertible a
    var a_pinv = try pinv(a, .{});
    defer a_pinv.deinit();
    var pinv_prod = try matmul(a, a_pinv, .{});
    defer pinv_prod.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), try pinv_prod.get(f64, &.{ 0, 0 }), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), try pinv_prod.get(f64, &.{ 0, 1 }), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), try pinv_prod.get(f64, &.{ 1, 0 }), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), try pinv_prod.get(f64, &.{ 1, 1 }), 1e-5);

    // slogdet: det(A) = 5.0 -> sign = 1.0, logabsdet = ln(5.0) ~ 1.6094379
    var sl = try slogdet(a);
    defer sl.deinit();
    try std.testing.expectEqual(@as(f64, 1.0), try sl.sign.get(f64, &.{}));
    try std.testing.expectApproxEqAbs(@as(f64, @log(5.0)), try sl.logabsdet.get(f64, &.{}), 1e-6);
}

test "triangular, SPD, and least-squares solvers" {
    const allocator = std.testing.allocator;
    const fromSlice = @import("../core/array.zig").fromSlice;
    const matmul = @import("matmul.zig").matmul;

    // Lower-triangular: [[2, 0], [1, 3]] * [x0, x1] = [4, 7] -> x = [2, 5/3]
    const l_data = [_]f64{ 2, 0, 1, 3 };
    var l = try fromSlice(allocator, f64, .{ .data = &l_data, .shape = &.{ 2, 2 } });
    defer l.deinit();
    const lb_data = [_]f64{ 4, 7 };
    var lb = try fromSlice(allocator, f64, .{ .data = &lb_data, .shape = &.{2} });
    defer lb.deinit();
    var xl = try solveTriangular(l, lb, .{ .lower = true });
    defer xl.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), try xl.get(f64, &.{0}), 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0 / 3.0), try xl.get(f64, &.{1}), 1e-9);

    // Upper-triangular: [[2, 1], [0, 3]] * x = [5, 6] -> x = [1.5, 2]
    const u_data = [_]f64{ 2, 1, 0, 3 };
    var u = try fromSlice(allocator, f64, .{ .data = &u_data, .shape = &.{ 2, 2 } });
    defer u.deinit();
    const ub_data = [_]f64{ 5, 6 };
    var ub = try fromSlice(allocator, f64, .{ .data = &ub_data, .shape = &.{2} });
    defer ub.deinit();
    var xu = try solveTriangular(u, ub, .{ .lower = false });
    defer xu.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), try xu.get(f64, &.{0}), 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), try xu.get(f64, &.{1}), 1e-9);

    // SPD: [[4, 1], [1, 3]] * x = [5, 4] -> x = [1, 1]
    const s_data = [_]f64{ 4, 1, 1, 3 };
    var s_mat = try fromSlice(allocator, f64, .{ .data = &s_data, .shape = &.{ 2, 2 } });
    defer s_mat.deinit();
    const sb_data = [_]f64{ 5, 4 };
    var sb = try fromSlice(allocator, f64, .{ .data = &sb_data, .shape = &.{2} });
    defer sb.deinit();
    var xs = try solveSpd(s_mat, sb);
    defer xs.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), try xs.get(f64, &.{0}), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), try xs.get(f64, &.{1}), 1e-6);

    // Least-squares: fit y = 2x + 1 through (0,1),(1,3),(2,5)
    const a_data = [_]f64{ 0, 1, 1, 1, 2, 1 };
    var amat = try fromSlice(allocator, f64, .{ .data = &a_data, .shape = &.{ 3, 2 } });
    defer amat.deinit();
    const y_data = [_]f64{ 1, 3, 5 };
    var y = try fromSlice(allocator, f64, .{ .data = &y_data, .shape = &.{3} });
    defer y.deinit();
    var xls = try lstsq(amat, y);
    defer xls.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), try xls.get(f64, &.{0}), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), try xls.get(f64, &.{1}), 1e-6);

    // Residual check: A xls ≈ y
    var pred = try matmul(amat, xls, .{});
    defer pred.deinit();
    for (0..3) |i| {
        try std.testing.expectApproxEqAbs(y_data[i], try pred.get(f64, &.{i}), 1e-6);
    }

    // f32 dtype is preserved end to end
    const f32_a = [_]f32{ 2, 0, 1, 3 };
    var fa = try fromSlice(allocator, f32, .{ .data = &f32_a, .shape = &.{ 2, 2 } });
    defer fa.deinit();
    const f32_b = [_]f32{ 4, 7 };
    var fb = try fromSlice(allocator, f32, .{ .data = &f32_b, .shape = &.{2} });
    defer fb.deinit();
    var fx = try solveTriangular(fa, fb, .{ .lower = true });
    defer fx.deinit();
    try std.testing.expect(fx.dtype == .f32);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), try fx.get(f32, &.{0}), 1e-5);
}
