//! Polynomial least-squares curve fitting.
//!
//! Fits a polynomial of specified degree to 1D data points (x, y)
//! via QR factorization of the Vandermonde matrix.

const std = @import("std");
const Array = @import("../core/array.zig").Array;
const zeros = @import("../core/array.zig").zeros;
const empty = @import("../core/array.zig").empty;
const fromSlice = @import("../core/array.zig").fromSlice;
const DType = @import("../core/dtype.zig").DType;
const Shape = @import("../core/shape.zig").Shape;
const ShapeError = @import("../core/error.zig").ShapeError;
const DTypeError = @import("../core/error.zig").DTypeError;
const IndexError = @import("../core/error.zig").IndexError;
const LinalgError = @import("../core/error.zig").LinalgError;

const qr = @import("../linalg/decompose.zig").qr;
const transpose = @import("../manip/transpose.zig").transpose;
const matmul = @import("../linalg/matmul.zig").matmul;
const expandDims = @import("../manip/reshape.zig").expandDims;

/// Fits a polynomial of degree `deg` to points `(x, y)` using least-squares via QR decomposition.
/// Returns coefficients with highest degree first: [c_deg, ..., c_0].
pub fn fit(
    x: Array,
    y: Array,
    deg: usize,
) (ShapeError || DTypeError || IndexError || LinalgError || std.mem.Allocator.Error)!Array {
    if (x.ndim != 1 or y.ndim != 1) return ShapeError.InvalidDimension;
    if (x.shape_dims[0] != y.shape_dims[0]) return ShapeError.IncompatibleShapes;

    const N = x.shape_dims[0];
    const K = deg + 1;
    if (N < K) return ShapeError.InvalidDimension;

    // 1. Construct Vandermonde matrix V of size (N, K): V[i, j] = x[i] ^ (deg - j)
    var v_mat = try empty(x.allocator, .{
        .shape = &.{ N, K },
        .dtype = .f64,
    });
    defer v_mat.deinit();

    for (0..N) |i| {
        const xi = try x.get(f64, &.{i});
        var cur_pow: f64 = 1.0;
        var j = K;
        while (j > 0) {
            j -= 1;
            try v_mat.set(f64, &.{ i, j }, cur_pow);
            cur_pow *= xi;
        }
    }

    // 2. Compute QR factorization: V = Q * R
    var qr_res = try qr(v_mat);
    defer qr_res.deinit();

    // 3. Compute rhs = Q^T * y: (K, N) x (N, 1) -> (K, 1)
    const q_t = try transpose(qr_res.q, .{});

    const y_2d = try expandDims(y, .{ .axis = 1 });
    var qty = try matmul(q_t, y_2d, .{ .dtype = .f64 });
    defer qty.deinit();

    // 4. Back-substitution: R * c = qty
    var coeffs = try zeros(x.allocator, .{ .shape = &.{K}, .dtype = .f64 });
    errdefer coeffs.deinit();

    var i = K;
    while (i > 0) {
        i -= 1;
        const r_ii = try qr_res.r.get(f64, &.{ i, i });
        if (@abs(r_ii) < 1e-15) return LinalgError.SingularMatrix;

        var sum_val: f64 = 0;
        for (i + 1..K) |j| {
            const r_ij = try qr_res.r.get(f64, &.{ i, j });
            const c_j = try coeffs.get(f64, &.{j});
            sum_val += r_ij * c_j;
        }

        const b_i = try qty.get(f64, &.{ i, 0 });
        const c_i = (b_i - sum_val) / r_ii;
        try coeffs.set(f64, &.{i}, c_i);
    }

    return coeffs;
}

test "fit linear regression" {
    const allocator = std.testing.allocator;

    // Line: y = 2x + 1
    const x_vals = [_]f64{ 0.0, 1.0, 2.0, 3.0, 4.0 };
    const y_vals = [_]f64{ 1.0, 3.0, 5.0, 7.0, 9.0 };

    var x = try fromSlice(allocator, f64, .{ .data = &x_vals, .shape = &.{5} });
    defer x.deinit();

    var y = try fromSlice(allocator, f64, .{ .data = &y_vals, .shape = &.{5} });
    defer y.deinit();

    var f = try fit(x, y, 1);
    defer f.deinit();

    // Slope ~ 2.0, Intercept ~ 1.0
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), try f.get(f64, &.{0}), 1e-4);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), try f.get(f64, &.{1}), 1e-4);
}
