//! Polynomial root finding and coefficient arithmetic.
//!
//! Coefficients use highest-degree-first ordering: `[c_n, ..., c_0]`.
//! `roots` returns complex roots (`c128`) using analytic formulas for
//! degrees 1-2 and the Durand-Kerner iteration otherwise.

const std = @import("std");
const Array = @import("../core/array.zig").Array;
const empty = @import("../core/array.zig").empty;
const fromSlice = @import("../core/array.zig").fromSlice;
const ShapeError = @import("../core/error.zig").ShapeError;
const DTypeError = @import("../core/error.zig").DTypeError;
const IndexError = @import("../core/error.zig").IndexError;

const C128 = std.math.Complex(f64);

fn coeffsToSlice(coeffs: Array, buf: []f64) !void {
    if (coeffs.ndim != 1) return ShapeError.InvalidDimension;
    if (coeffs.shape_dims[0] != buf.len) return ShapeError.IncompatibleShapes;
    for (0..buf.len) |i| buf[i] = try coeffs.getAsFloat(&.{i});
}

/// Finds polynomial roots. Returns a 1D `c128` array of length `deg`.
/// Linear and quadratic cases are analytic; higher degrees use Durand-Kerner
/// with a fixed iteration budget (1000) and 1e-12 convergence tolerance.
pub fn roots(coeffs: Array) (ShapeError || DTypeError || IndexError || std.mem.Allocator.Error)!Array {
    if (coeffs.ndim != 1) return ShapeError.InvalidDimension;
    const n = coeffs.shape_dims[0];
    if (n == 0) return ShapeError.EmptyArray;
    if (n == 1) {
        // Non-zero constant has no roots; zero constant is degenerate.
        const c0 = try coeffs.getAsFloat(&.{0});
        if (c0 == 0.0) return ShapeError.InvalidDimension;
        return empty(coeffs.allocator, .{ .shape = &.{0}, .dtype = .c128 });
    }
    const deg = n - 1;
    const lead = try coeffs.getAsFloat(&.{0});
    if (lead == 0.0) return ShapeError.InvalidDimension;

    // Normalize to monic: p[i] / lead
    const poly = try coeffs.allocator.alloc(f64, n);
    defer coeffs.allocator.free(poly);
    for (0..n) |i| poly[i] = (try coeffs.getAsFloat(&.{i})) / lead;

    var out = try empty(coeffs.allocator, .{ .shape = &.{deg}, .dtype = .c128 });
    errdefer out.deinit();
    const out_slice = try out.asSlice(C128);

    if (deg == 1) {
        out_slice[0] = C128.init(-poly[1], 0.0);
        return out;
    }
    if (deg == 2) {
        const b = poly[1];
        const c = poly[2];
        const disc = b * b - 4.0 * c;
        if (disc >= 0) {
            const s = @sqrt(disc);
            out_slice[0] = C128.init((-b + s) / 2.0, 0.0);
            out_slice[1] = C128.init((-b - s) / 2.0, 0.0);
        } else {
            const s = @sqrt(-disc);
            out_slice[0] = C128.init(-b / 2.0, s / 2.0);
            out_slice[1] = C128.init(-b / 2.0, -s / 2.0);
        }
        return out;
    }

    // Durand-Kerner for deg >= 3.
    const angle_step = 2.0 * std.math.pi / @as(f64, @floatFromInt(deg));
    for (0..deg) |i| {
        const a = angle_step * @as(f64, @floatFromInt(i)) + 0.4;
        out_slice[i] = C128.init(@cos(a) * 0.8, @sin(a) * 0.8);
    }
    var k: usize = 0;
    while (k < 1000) : (k += 1) {
        var max_change: f64 = 0.0;
        for (0..deg) |i| {
            const xi = out_slice[i];
            // Evaluate monic polynomial at xi.
            var px = C128.init(poly[0], 0.0);
            for (1..n) |j| {
                px = px.mul(xi).add(C128.init(poly[j], 0.0));
            }
            var denom = C128.init(1.0, 0.0);
            for (0..deg) |j| {
                if (j == i) continue;
                denom = denom.mul(xi.sub(out_slice[j]));
            }
            const denom_abs = denom.re * denom.re + denom.im * denom.im;
            if (denom_abs < 1e-24) continue;
            const delta = px.div(denom);
            const next = xi.sub(delta);
            const change = @abs(next.re - xi.re) + @abs(next.im - xi.im);
            if (change > max_change) max_change = change;
            out_slice[i] = next;
        }
        if (max_change < 1e-12) break;
    }
    return out;
}

fn alignLengths(allocator: std.mem.Allocator, a: []const f64, b: []const f64) !struct { x: []f64, y: []f64, n: usize } {
    const n = @max(a.len, b.len);
    const x = try allocator.alloc(f64, n);
    errdefer allocator.free(x);
    const y = try allocator.alloc(f64, n);
    @memset(x, 0.0);
    @memset(y, 0.0);
    @memcpy(x[n - a.len ..], a);
    @memcpy(y[n - b.len ..], b);
    return .{ .x = x, .y = y, .n = n };
}

/// Adds two polynomials (highest-degree-first). Returns trimmed leading zeros (keeps one).
pub fn add(a: Array, b: Array) (ShapeError || DTypeError || IndexError || std.mem.Allocator.Error)!Array {
    if (a.ndim != 1 or b.ndim != 1) return ShapeError.InvalidDimension;
    const na = a.shape_dims[0];
    const nb = b.shape_dims[0];
    if (na == 0 or nb == 0) return ShapeError.EmptyArray;
    const abuf = try a.allocator.alloc(f64, na);
    defer a.allocator.free(abuf);
    const bbuf = try a.allocator.alloc(f64, nb);
    defer a.allocator.free(bbuf);
    try coeffsToSlice(a, abuf);
    try coeffsToSlice(b, bbuf);
    const al = try alignLengths(a.allocator, abuf, bbuf);
    defer a.allocator.free(al.x);
    defer a.allocator.free(al.y);
    for (0..al.n) |i| al.x[i] += al.y[i];
    return fromTrimmed(a.allocator, al.x);
}

/// Subtracts polynomial `b` from `a`.
pub fn sub(a: Array, b: Array) (ShapeError || DTypeError || IndexError || std.mem.Allocator.Error)!Array {
    if (a.ndim != 1 or b.ndim != 1) return ShapeError.InvalidDimension;
    const na = a.shape_dims[0];
    const nb = b.shape_dims[0];
    if (na == 0 or nb == 0) return ShapeError.EmptyArray;
    const abuf = try a.allocator.alloc(f64, na);
    defer a.allocator.free(abuf);
    const bbuf = try a.allocator.alloc(f64, nb);
    defer a.allocator.free(bbuf);
    try coeffsToSlice(a, abuf);
    try coeffsToSlice(b, bbuf);
    const al = try alignLengths(a.allocator, abuf, bbuf);
    defer a.allocator.free(al.x);
    defer a.allocator.free(al.y);
    for (0..al.n) |i| al.x[i] -= al.y[i];
    return fromTrimmed(a.allocator, al.x);
}

/// Multiplies two polynomials (convolution of coefficients).
pub fn mul(a: Array, b: Array) (ShapeError || DTypeError || IndexError || std.mem.Allocator.Error)!Array {
    if (a.ndim != 1 or b.ndim != 1) return ShapeError.InvalidDimension;
    const na = a.shape_dims[0];
    const nb = b.shape_dims[0];
    if (na == 0 or nb == 0) return ShapeError.EmptyArray;
    const abuf = try a.allocator.alloc(f64, na);
    defer a.allocator.free(abuf);
    const bbuf = try a.allocator.alloc(f64, nb);
    defer a.allocator.free(bbuf);
    try coeffsToSlice(a, abuf);
    try coeffsToSlice(b, bbuf);
    const n = na + nb - 1;
    const out_buf = try a.allocator.alloc(f64, n);
    defer a.allocator.free(out_buf);
    @memset(out_buf, 0.0);
    for (0..na) |i| {
        for (0..nb) |j| out_buf[i + j] += abuf[i] * bbuf[j];
    }
    return fromTrimmed(a.allocator, out_buf);
}

fn fromTrimmed(allocator: std.mem.Allocator, coeffs: []const f64) (ShapeError || std.mem.Allocator.Error)!Array {
    var start: usize = 0;
    while (start + 1 < coeffs.len and coeffs[start] == 0.0) : (start += 1) {}
    return fromSlice(allocator, f64, .{ .data = coeffs[start..], .shape = &.{coeffs.len - start} });
}

test "poly roots linear and quadratic" {
    const allocator = std.testing.allocator;
    // x - 2 = 0 -> root 2
    const c1 = [_]f64{ 1.0, -2.0 };
    var p1 = try fromSlice(allocator, f64, .{ .data = &c1, .shape = &.{2} });
    defer p1.deinit();
    var r1 = try roots(p1);
    defer r1.deinit();
    try std.testing.expectEqual(@as(usize, 1), r1.elementCount());
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), (try r1.asSlice(C128))[0].re, 1e-9);

    // x^2 - 5x + 6 = (x-2)(x-3)
    const c2 = [_]f64{ 1.0, -5.0, 6.0 };
    var p2 = try fromSlice(allocator, f64, .{ .data = &c2, .shape = &.{3} });
    defer p2.deinit();
    var r2 = try roots(p2);
    defer r2.deinit();
    const rs = try r2.asSlice(C128);
    // order: larger first from formula
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), rs[0].re, 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), rs[1].re, 1e-9);
}

test "poly roots cubic residual" {
    const allocator = std.testing.allocator;
    // (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6
    const c = [_]f64{ 1.0, -6.0, 11.0, -6.0 };
    var p = try fromSlice(allocator, f64, .{ .data = &c, .shape = &.{4} });
    defer p.deinit();
    var r = try roots(p);
    defer r.deinit();
    try std.testing.expectEqual(@as(usize, 3), r.elementCount());
    const rs = try r.asSlice(C128);
    for (rs) |root| {
        // |p(root)| should be ~0
        var acc = C128.init(c[0], 0.0);
        for (c[1..]) |coef| acc = acc.mul(root).add(C128.init(coef, 0.0));
        const mag = @sqrt(acc.re * acc.re + acc.im * acc.im);
        try std.testing.expect(mag < 1e-6);
    }
}

test "poly coefficient add sub mul" {
    const allocator = std.testing.allocator;
    const a_vals = [_]f64{ 1.0, 2.0 }; // x + 2
    const b_vals = [_]f64{ 1.0, 3.0 }; // x + 3
    var a = try fromSlice(allocator, f64, .{ .data = &a_vals, .shape = &.{2} });
    defer a.deinit();
    var b = try fromSlice(allocator, f64, .{ .data = &b_vals, .shape = &.{2} });
    defer b.deinit();
    var s = try add(a, b);
    defer s.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 5.0 }, try s.asSlice(f64));
    var d = try sub(a, b);
    defer d.deinit();
    try std.testing.expectEqualSlices(f64, &.{-1.0}, try d.asSlice(f64));
    var m = try mul(a, b);
    defer m.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 5.0, 6.0 }, try m.asSlice(f64));
}
