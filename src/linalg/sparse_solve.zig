//! Iterative sparse linear solvers for Compressed Sparse Row (CSR) matrices.
//!
//! Provides Conjugate Gradient (CG) for symmetric positive-definite systems and
//! Generalized Minimal Residual (GMRES) for general non-symmetric square systems.

const std = @import("std");
const Array = @import("../core/array.zig").Array;
const zeros = @import("../core/array.zig").zeros;
const empty = @import("../core/array.zig").empty;
const fromSlice = @import("../core/array.zig").fromSlice;
const CsrMatrix = @import("../core/sparse.zig").CsrMatrix;
const ShapeError = @import("../core/error.zig").ShapeError;
const LinalgError = @import("../core/error.zig").LinalgError;
const DTypeError = @import("../core/error.zig").DTypeError;
const IndexError = @import("../core/error.zig").IndexError;

/// Result of an iterative sparse linear solve.
pub const SparseSolveResult = struct {
    x: Array,
    iterations: usize,
    converged: bool,
    residualNorm: f64,

    pub fn deinit(self: *SparseSolveResult) void {
        self.x.deinit();
    }
};

/// Options for Conjugate Gradient solver.
pub const CgOptions = struct {
    maxIter: usize = 1000,
    tol: f64 = 1e-6,
    x0: ?Array = null,
};

/// Options for GMRES solver.
pub const GmresOptions = struct {
    restart: usize = 30,
    maxIter: usize = 1000,
    tol: f64 = 1e-6,
    x0: ?Array = null,
};

/// Solves A x = b for symmetric positive-definite A using Conjugate Gradient.
pub fn cg(
    A: CsrMatrix,
    b: Array,
    options: CgOptions,
) (ShapeError || LinalgError || DTypeError || IndexError || std.mem.Allocator.Error)!SparseSolveResult {
    if (A.rows != A.cols) return LinalgError.MatrixNotSquare;
    if (b.ndim != 1 or b.shape_dims[0] != A.rows) return ShapeError.IncompatibleShapes;

    const n = A.rows;
    if (n == 0) return ShapeError.EmptyArray;

    const allocator = A.allocator;

    // Allocate solution vector x
    var x = try zeros(allocator, .{ .shape = &.{n}, .dtype = .f64 });
    errdefer x.deinit();

    if (options.x0) |x0_arr| {
        if (x0_arr.ndim != 1 or x0_arr.shape_dims[0] != n) return ShapeError.IncompatibleShapes;
        for (0..n) |i| {
            try x.setFromFloat(&.{i}, try x0_arr.getAsFloat(&.{i}));
        }
    }

    const x_slice = x.asSlice(f64) catch unreachable;

    // Compute b_norm
    var b_norm_sq: f64 = 0.0;
    for (0..n) |i| {
        const val = try b.getAsFloat(&.{i});
        b_norm_sq += val * val;
    }
    const b_norm = @sqrt(b_norm_sq);
    const threshold = options.tol * (if (b_norm > 0) b_norm else 1.0);

    // Compute initial residual r = b - A*x
    const r = try allocator.alloc(f64, n);
    defer allocator.free(r);

    var Ax0 = try A.dotVector(x);
    defer Ax0.deinit();

    var rho: f64 = 0.0;
    for (0..n) |i| {
        const b_i = try b.getAsFloat(&.{i});
        const Ax_i = try Ax0.getAsFloat(&.{i});
        r[i] = b_i - Ax_i;
        rho += r[i] * r[i];
    }

    if (@sqrt(rho) <= threshold) {
        return SparseSolveResult{
            .x = x,
            .iterations = 0,
            .converged = true,
            .residualNorm = @sqrt(rho),
        };
    }

    // Direction vector p
    const p = try allocator.alloc(f64, n);
    defer allocator.free(p);
    @memcpy(p, r);

    // Workspace for Ap
    var p_arr = try fromSlice(allocator, f64, .{ .data = p, .shape = &.{n} });
    defer p_arr.deinit();

    var iter: usize = 0;
    var converged = false;
    var res_norm: f64 = @sqrt(rho);

    while (iter < options.maxIter) : (iter += 1) {
        // Ap = A * p
        @memcpy(p_arr.asSlice(f64) catch unreachable, p);
        var Ap_arr = try A.dotVector(p_arr);
        defer Ap_arr.deinit();
        const Ap_slice = Ap_arr.asConstSlice(f64) catch unreachable;

        // pAp = p . Ap
        var pAp: f64 = 0.0;
        for (0..n) |i| {
            pAp += p[i] * Ap_slice[i];
        }

        if (pAp <= 0.0) {
            // Indefinite or singular matrix detected
            break;
        }

        const alpha = rho / pAp;

        // x += alpha * p
        // r -= alpha * Ap
        var rho_new: f64 = 0.0;
        for (0..n) |i| {
            x_slice[i] += alpha * p[i];
            r[i] -= alpha * Ap_slice[i];
            rho_new += r[i] * r[i];
        }

        res_norm = @sqrt(rho_new);
        if (res_norm <= threshold) {
            converged = true;
            iter += 1;
            break;
        }

        const beta = rho_new / rho;
        for (0..n) |i| {
            p[i] = r[i] + beta * p[i];
        }

        rho = rho_new;
    }

    return SparseSolveResult{
        .x = x,
        .iterations = iter,
        .converged = converged,
        .residualNorm = res_norm,
    };
}

/// Solves A x = b for general square non-symmetric A using Restarted GMRES.
pub fn gmres(
    A: CsrMatrix,
    b: Array,
    options: GmresOptions,
) (ShapeError || LinalgError || DTypeError || IndexError || std.mem.Allocator.Error)!SparseSolveResult {
    if (A.rows != A.cols) return LinalgError.MatrixNotSquare;
    if (b.ndim != 1 or b.shape_dims[0] != A.rows) return ShapeError.IncompatibleShapes;

    const n = A.rows;
    if (n == 0) return ShapeError.EmptyArray;

    const allocator = A.allocator;
    const m = @min(options.restart, n);

    var x = try zeros(allocator, .{ .shape = &.{n}, .dtype = .f64 });
    errdefer x.deinit();

    if (options.x0) |x0_arr| {
        if (x0_arr.ndim != 1 or x0_arr.shape_dims[0] != n) return ShapeError.IncompatibleShapes;
        for (0..n) |i| {
            try x.setFromFloat(&.{i}, try x0_arr.getAsFloat(&.{i}));
        }
    }

    const x_slice = x.asSlice(f64) catch unreachable;

    // b_norm
    var b_norm_sq: f64 = 0.0;
    for (0..n) |i| {
        const val = try b.getAsFloat(&.{i});
        b_norm_sq += val * val;
    }
    const b_norm = @sqrt(b_norm_sq);
    const threshold = options.tol * (if (b_norm > 0) b_norm else 1.0);

    // Allocate Arnoldi basis: (m + 1) * n
    const V = try allocator.alloc(f64, (m + 1) * n);
    defer allocator.free(V);

    // Hessenberg matrix H: (m + 1) * m
    const H = try allocator.alloc(f64, (m + 1) * m);
    defer allocator.free(H);

    // Givens rotations
    const cs = try allocator.alloc(f64, m);
    defer allocator.free(cs);
    const sn = try allocator.alloc(f64, m);
    defer allocator.free(sn);

    // Right hand side vector g of length m + 1
    const g = try allocator.alloc(f64, m + 1);
    defer allocator.free(g);

    // Vector for dotVector
    var v_arr = try empty(allocator, .{ .shape = &.{n}, .dtype = .f64 });
    defer v_arr.deinit();

    var total_iter: usize = 0;
    var converged = false;
    var res_norm: f64 = 0.0;

    while (total_iter < options.maxIter) {
        // r = b - A*x
        var Ax = try A.dotVector(x);
        defer Ax.deinit();

        var r_norm_sq: f64 = 0.0;
        for (0..n) |i| {
            const b_i = try b.getAsFloat(&.{i});
            const Ax_i = try Ax.getAsFloat(&.{i});
            const r_i = b_i - Ax_i;
            V[0 * n + i] = r_i;
            r_norm_sq += r_i * r_i;
        }

        res_norm = @sqrt(r_norm_sq);
        if (res_norm <= threshold) {
            converged = true;
            break;
        }

        // v_0 = r / r_norm
        for (0..n) |i| {
            V[0 * n + i] /= res_norm;
        }

        @memset(g, 0.0);
        g[0] = res_norm;

        var k: usize = 0;
        while (k < m and total_iter < options.maxIter) : (k += 1) {
            total_iter += 1;

            // w = A * v_k
            const vk = V[k * n .. (k + 1) * n];
            @memcpy(v_arr.asSlice(f64) catch unreachable, vk);
            var w_arr = try A.dotVector(v_arr);
            defer w_arr.deinit();
            const w = w_arr.asSlice(f64) catch unreachable;

            // Modified Gram-Schmidt orthogonalization
            for (0..k + 1) |i| {
                const vi = V[i * n .. (i + 1) * n];
                var h_val: f64 = 0.0;
                for (0..n) |idx| {
                    h_val += vi[idx] * w[idx];
                }
                H[i * m + k] = h_val;
                for (0..n) |idx| {
                    w[idx] -= h_val * vi[idx];
                }
            }

            var w_norm_sq: f64 = 0.0;
            for (0..n) |idx| {
                w_norm_sq += w[idx] * w[idx];
            }
            const h_next = @sqrt(w_norm_sq);
            H[(k + 1) * m + k] = h_next;

            if (h_next > 1e-15) {
                const v_next = V[(k + 1) * n .. (k + 2) * n];
                for (0..n) |idx| {
                    v_next[idx] = w[idx] / h_next;
                }
            }

            // Apply existing Givens rotations to column k
            for (0..k) |i| {
                const temp = cs[i] * H[i * m + k] + sn[i] * H[(i + 1) * m + k];
                H[(i + 1) * m + k] = -sn[i] * H[i * m + k] + cs[i] * H[(i + 1) * m + k];
                H[i * m + k] = temp;
            }

            // Generate new Givens rotation to zero out H[(k + 1) * m + k]
            const h1 = H[k * m + k];
            const h2 = H[(k + 1) * m + k];
            const denom = @sqrt(h1 * h1 + h2 * h2);

            if (denom > 1e-15) {
                cs[k] = h1 / denom;
                sn[k] = h2 / denom;
            } else {
                cs[k] = 1.0;
                sn[k] = 0.0;
            }

            H[k * m + k] = cs[k] * h1 + sn[k] * h2;
            H[(k + 1) * m + k] = 0.0;

            // Apply rotation to g
            g[k + 1] = -sn[k] * g[k];
            g[k] = cs[k] * g[k];

            res_norm = @abs(g[k + 1]);
            if (res_norm <= threshold) {
                k += 1;
                break;
            }
        }

        // Back-substitution to solve H * y = g (upper triangular system)
        const y_sol = try allocator.alloc(f64, k);
        defer allocator.free(y_sol);

        var row: usize = k;
        while (row > 0) {
            row -= 1;
            var sum = g[row];
            for (row + 1..k) |col| {
                sum -= H[row * m + col] * y_sol[col];
            }
            y_sol[row] = if (@abs(H[row * m + row]) > 1e-15) sum / H[row * m + row] else 0.0;
        }

        // x += sum(y_i * v_i)
        for (0..k) |i| {
            const vi = V[i * n .. (i + 1) * n];
            const yi = y_sol[i];
            for (0..n) |idx| {
                x_slice[idx] += yi * vi[idx];
            }
        }

        if (res_norm <= threshold) {
            converged = true;
            break;
        }
    }

    return SparseSolveResult{
        .x = x,
        .iterations = total_iter,
        .converged = converged,
        .residualNorm = res_norm,
    };
}

test "cg on 2x2 SPD matrix" {
    const allocator = std.testing.allocator;

    // A = [ 4, 1 ]
    //     [ 1, 3 ]
    // b = [ 5, 4 ]
    // Exact solution: x = [ 1, 1 ]
    var dense = try zeros(allocator, .{ .shape = &.{ 2, 2 }, .dtype = .f64 });
    defer dense.deinit();

    try dense.setFromFloat(&.{ 0, 0 }, 4.0);
    try dense.setFromFloat(&.{ 0, 1 }, 1.0);
    try dense.setFromFloat(&.{ 1, 0 }, 1.0);
    try dense.setFromFloat(&.{ 1, 1 }, 3.0);

    var A = try CsrMatrix.fromDense(allocator, dense, 1e-9);
    defer A.deinit();

    const b_data = [_]f64{ 5.0, 4.0 };
    var b = try fromSlice(allocator, f64, .{ .data = &b_data, .shape = &.{2} });
    defer b.deinit();

    var res = try cg(A, b, .{ .tol = 1e-7 });
    defer res.deinit();

    try std.testing.expect(res.converged);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), try res.x.getAsFloat(&.{0}), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), try res.x.getAsFloat(&.{1}), 1e-5);
}

test "gmres on 2x2 non-symmetric matrix" {
    const allocator = std.testing.allocator;

    // A = [ 3, 2 ]
    //     [ 1, 4 ]
    // b = [ 8, 6 ]
    // 3*2 + 2*1 = 8
    // 1*2 + 4*1 = 6 -> x = [ 2, 1 ]
    var dense = try zeros(allocator, .{ .shape = &.{ 2, 2 }, .dtype = .f64 });
    defer dense.deinit();

    try dense.setFromFloat(&.{ 0, 0 }, 3.0);
    try dense.setFromFloat(&.{ 0, 1 }, 2.0);
    try dense.setFromFloat(&.{ 1, 0 }, 1.0);
    try dense.setFromFloat(&.{ 1, 1 }, 4.0);

    var A = try CsrMatrix.fromDense(allocator, dense, 1e-9);
    defer A.deinit();

    const b_data = [_]f64{ 8.0, 6.0 };
    var b = try fromSlice(allocator, f64, .{ .data = &b_data, .shape = &.{2} });
    defer b.deinit();

    var res = try gmres(A, b, .{ .tol = 1e-7 });
    defer res.deinit();

    try std.testing.expect(res.converged);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), try res.x.getAsFloat(&.{0}), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), try res.x.getAsFloat(&.{1}), 1e-5);
}
