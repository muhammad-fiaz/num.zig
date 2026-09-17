//! Polynomial evaluation, differentiation, and integration.
//!
//! Implements Horner's method evaluation (val), differentiation (der),
//! and indefinite integration (integ).

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

const add_fn = @import("../ops/elementwise.zig").add;
const mul_fn = @import("../ops/elementwise.zig").multiply;
const ravel = @import("../manip/reshape.zig").ravel;

/// Evaluates a polynomial at specified points using Horner's method.
/// `coeffs` has highest degree first: [c_n, c_{n-1}, ..., c_0].
pub fn val(
    coeffs: Array,
    x: Array,
) (ShapeError || DTypeError || IndexError || std.mem.Allocator.Error)!Array {
    if (coeffs.elementCount() == 0) return ShapeError.EmptyArray;

    const n_coeffs = coeffs.elementCount();
    var flat_coeffs = try ravel(coeffs);
    defer flat_coeffs.deinit();

    // Start accumulator with highest degree coefficient: c_n
    const c_lead = try flat_coeffs.get(f64, &.{0});
    const c_lead_slice = [_]f64{c_lead};
    var acc = try fromSlice(x.allocator, f64, .{ .data = &c_lead_slice, .shape = &.{} });
    errdefer acc.deinit();

    // Promote accumulator to match x shape
    var ones_x = try @import("../core/array.zig").ones(x.allocator, .{ .shape = x.shapeSlice(), .dtype = .f64 });
    defer ones_x.deinit();

    const current = try mul_fn(acc, ones_x, .{ .dtype = .f64 });
    acc.deinit();
    acc = current;

    for (1..n_coeffs) |i| {
        // acc = acc * x + coeffs[i]
        var next_acc = try mul_fn(acc, x, .{ .dtype = .f64 });
        acc.deinit();

        const c_i = try flat_coeffs.get(f64, &.{i});
        const c_slice = [_]f64{c_i};
        var c_arr = try fromSlice(x.allocator, f64, .{ .data = &c_slice, .shape = &.{} });
        defer c_arr.deinit();

        acc = try add_fn(next_acc, c_arr, .{ .dtype = .f64 });
        next_acc.deinit();
    }

    return acc;
}

/// Computes the m-th derivative of a polynomial.
pub fn der(
    coeffs: Array,
    m: usize,
) (ShapeError || DTypeError || IndexError || std.mem.Allocator.Error)!Array {
    if (coeffs.ndim != 1) return ShapeError.InvalidDimension;
    const n = coeffs.shape_dims[0];
    if (n == 0) return ShapeError.EmptyArray;

    if (m == 0) return coeffs.clone();
    if (m >= n) {
        // Degree zero
        const res = try zeros(coeffs.allocator, .{ .shape = &.{1}, .dtype = .f64 });
        return res;
    }

    var current = try coeffs.clone();
    errdefer current.deinit();

    for (0..m) |_| {
        const cur_n = current.shape_dims[0];
        const next_n = cur_n - 1;
        var next_arr = try empty(coeffs.allocator, .{ .shape = &.{next_n}, .dtype = .f64 });
        errdefer next_arr.deinit();

        for (0..next_n) |i| {
            const power: f64 = @floatFromInt(next_n - i);
            const v = try current.get(f64, &.{i});
            try next_arr.set(f64, &.{i}, v * power);
        }

        current.deinit();
        current = next_arr;
    }

    return current;
}

/// Computes the indefinite integral of a polynomial.
pub fn integ(
    coeffs: Array,
    k: f64,
) (ShapeError || DTypeError || IndexError || std.mem.Allocator.Error)!Array {
    if (coeffs.ndim != 1) return ShapeError.InvalidDimension;
    const n = coeffs.shape_dims[0];

    var out = try zeros(coeffs.allocator, .{ .shape = &.{n + 1}, .dtype = .f64 });
    errdefer out.deinit();

    for (0..n) |i| {
        const power: f64 = @floatFromInt(n - i);
        const v = try coeffs.get(f64, &.{i});
        try out.set(f64, &.{i}, v / power);
    }

    try out.set(f64, &.{n}, k);
    return out;
}

test "val using Horner's method" {
    const allocator = std.testing.allocator;

    // p(x) = 2x^2 - 3x + 5: coeffs = [2.0, -3.0, 5.0]
    const c = [_]f64{ 2.0, -3.0, 5.0 };
    var coeffs = try fromSlice(allocator, f64, .{ .data = &c, .shape = &.{3} });
    defer coeffs.deinit();

    // At x = 2: 2*(4) - 3*(2) + 5 = 8 - 6 + 5 = 7.0
    const x_val = [_]f64{2.0};
    var x = try fromSlice(allocator, f64, .{ .data = &x_val, .shape = &.{} });
    defer x.deinit();

    var v = try val(coeffs, x);
    defer v.deinit();

    try std.testing.expectApproxEqAbs(@as(f64, 7.0), try v.get(f64, &.{}), 1e-5);
}

test "der and integ" {
    const allocator = std.testing.allocator;

    // p(x) = 3x^2 + 4x + 5 -> p'(x) = 6x + 4
    const c = [_]f64{ 3.0, 4.0, 5.0 };
    var coeffs = try fromSlice(allocator, f64, .{ .data = &c, .shape = &.{3} });
    defer coeffs.deinit();

    var d = try der(coeffs, 1);
    defer d.deinit();

    try std.testing.expectEqual(@as(usize, 2), d.elementCount());
    try std.testing.expectApproxEqAbs(@as(f64, 6.0), try d.get(f64, &.{0}), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), try d.get(f64, &.{1}), 1e-5);

    // Integral of 6x + 4 with constant 5 -> 3x^2 + 4x + 5
    var itg = try integ(d, 5.0);
    defer itg.deinit();

    try std.testing.expectEqual(@as(usize, 3), itg.elementCount());
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), try itg.get(f64, &.{0}), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), try itg.get(f64, &.{1}), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), try itg.get(f64, &.{2}), 1e-5);
}
