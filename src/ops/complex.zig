//! Complex number helpers for `c64` / `c128` arrays.
//!
//! Provides `conj`/`conjugate`, `real`, `imag`, `magnitude` and `phase`.
//! All operations preserve views/ownership discipline by returning new owned
//! arrays. Comparisons and ordering on complex dtypes are unsupported.

const std = @import("std");
const Array = @import("../core/array.zig").Array;
const empty = @import("../core/array.zig").empty;
const DType = @import("../core/dtype.zig").DType;
const ShapeError = @import("../core/error.zig").ShapeError;
const DTypeError = @import("../core/error.zig").DTypeError;
const NdIterator = @import("../core/iterator.zig").NdIterator;

fn requireComplex(dtype: DType) DTypeError!void {
    if (!dtype.isComplex()) return DTypeError.UnsupportedDType;
}

/// Complex conjugate: negates the imaginary component. Preserves dtype.
pub fn conj(a: Array) (ShapeError || DTypeError || std.mem.Allocator.Error)!Array {
    try requireComplex(a.dtype);
    var out = try empty(a.allocator, .{ .shape = a.shapeSlice(), .dtype = a.dtype });
    errdefer out.deinit();
    if (a.dtype == .c64) {
        const T = std.math.Complex(f32);
        if (a.isContiguous()) {
            const sa = a.asConstSlice(T) catch unreachable;
            const so = out.asSlice(T) catch unreachable;
            for (sa, 0..) |v, i| so[i] = T.init(v.re, -v.im);
        } else {
            copyConj(T, a, out);
        }
    } else {
        const T = std.math.Complex(f64);
        if (a.isContiguous()) {
            const sa = a.asConstSlice(T) catch unreachable;
            const so = out.asSlice(T) catch unreachable;
            for (sa, 0..) |v, i| so[i] = T.init(v.re, -v.im);
        } else {
            copyConj(T, a, out);
        }
    }
    return out;
}

fn copyConj(comptime T: type, a: Array, out: Array) void {
    var it = NdIterator.init(a.shape(), a.strides());
    var out_it = NdIterator.init(out.shape(), out.strides());
    const in_ptr: [*]const T = @ptrCast(@alignCast(a.data_ptr));
    const out_ptr: [*]T = @ptrCast(@alignCast(out.data_ptr));
    while (it.next()) |item| {
        const o = out_it.next().?;
        const v = in_ptr[@as(usize, @intCast(item.offset))];
        out_ptr[@as(usize, @intCast(o.offset))] = T.init(v.re, -v.im);
    }
}

/// Alias for `conj`.
pub const conjugate = conj;

const ComplexUnaryKind = enum { re, im, magnitude, phase };

/// Single canonical complex-to-float traversal shared by real/imag/magnitude/phase.
/// Uses `std.math.hypot` and `std.math.atan2` (verified Zig 0.16.0 APIs) with a
/// contiguous fast path and a stride-aware fallback.
fn mapComplexToFloat(a: Array, comptime kind: ComplexUnaryKind, out: Array) void {
    if (a.dtype == .c64) {
        const C = std.math.Complex(f32);
        if (a.isContiguous()) {
            const sa = a.asConstSlice(C) catch unreachable;
            const so = out.asSlice(f32) catch unreachable;
            for (sa, 0..) |v, i| so[i] = switch (kind) {
                .re => v.re,
                .im => v.im,
                .magnitude => std.math.hypot(v.re, v.im),
                .phase => std.math.atan2(v.im, v.re),
            };
        } else {
            const in_ptr: [*]const C = @ptrCast(@alignCast(a.data_ptr));
            const out_ptr: [*]f32 = @ptrCast(@alignCast(out.data_ptr));
            var it = NdIterator.init(a.shape(), a.strides());
            var out_idx: usize = 0;
            while (it.next()) |item| {
                const v = in_ptr[@as(usize, @intCast(item.offset))];
                out_ptr[out_idx] = switch (kind) {
                    .re => v.re,
                    .im => v.im,
                    .magnitude => std.math.hypot(v.re, v.im),
                    .phase => std.math.atan2(v.im, v.re),
                };
                out_idx += 1;
            }
        }
    } else {
        const C = std.math.Complex(f64);
        if (a.isContiguous()) {
            const sa = a.asConstSlice(C) catch unreachable;
            const so = out.asSlice(f64) catch unreachable;
            for (sa, 0..) |v, i| so[i] = switch (kind) {
                .re => v.re,
                .im => v.im,
                .magnitude => std.math.hypot(v.re, v.im),
                .phase => std.math.atan2(v.im, v.re),
            };
        } else {
            const in_ptr: [*]const C = @ptrCast(@alignCast(a.data_ptr));
            const out_ptr: [*]f64 = @ptrCast(@alignCast(out.data_ptr));
            var it = NdIterator.init(a.shape(), a.strides());
            var out_idx: usize = 0;
            while (it.next()) |item| {
                const v = in_ptr[@as(usize, @intCast(item.offset))];
                out_ptr[out_idx] = switch (kind) {
                    .re => v.re,
                    .im => v.im,
                    .magnitude => std.math.hypot(v.re, v.im),
                    .phase => std.math.atan2(v.im, v.re),
                };
                out_idx += 1;
            }
        }
    }
}

fn allocComplexFloatOut(a: Array) (ShapeError || std.mem.Allocator.Error)!Array {
    const out_dtype: DType = if (a.dtype == .c64) .f32 else .f64;
    return empty(a.allocator, .{ .shape = a.shapeSlice(), .dtype = out_dtype });
}

/// Real component. `c64 -> f32`, `c128 -> f64`.
pub fn real(a: Array) (ShapeError || DTypeError || std.mem.Allocator.Error)!Array {
    try requireComplex(a.dtype);
    var out = try allocComplexFloatOut(a);
    errdefer out.deinit();
    mapComplexToFloat(a, .re, out);
    return out;
}

/// Imaginary component. `c64 -> f32`, `c128 -> f64`.
pub fn imag(a: Array) (ShapeError || DTypeError || std.mem.Allocator.Error)!Array {
    try requireComplex(a.dtype);
    var out = try allocComplexFloatOut(a);
    errdefer out.deinit();
    mapComplexToFloat(a, .im, out);
    return out;
}

/// Magnitude `|z| = hypot(re, im)`. `c64 -> f32`, `c128 -> f64`.
pub fn magnitude(a: Array) (ShapeError || DTypeError || std.mem.Allocator.Error)!Array {
    try requireComplex(a.dtype);
    var out = try allocComplexFloatOut(a);
    errdefer out.deinit();
    mapComplexToFloat(a, .magnitude, out);
    return out;
}

/// Phase `arg(z) = atan2(im, re)` in radians. `c64 -> f32`, `c128 -> f64`.
pub fn phase(a: Array) (ShapeError || DTypeError || std.mem.Allocator.Error)!Array {
    try requireComplex(a.dtype);
    var out = try allocComplexFloatOut(a);
    errdefer out.deinit();
    mapComplexToFloat(a, .phase, out);
    return out;
}

test "complex construction, conj, real/imag, magnitude, phase" {
    const fromSlice = @import("../core/array.zig").fromSlice;
    const allocator = std.testing.allocator;
    const C128 = std.math.Complex(f64);
    const vals = [_]C128{ C128.init(3.0, 4.0), C128.init(1.0, 0.0), C128.init(0.0, 1.0) };
    var a = try fromSlice(allocator, C128, .{ .data = &vals, .shape = &.{3} });
    defer a.deinit();
    try std.testing.expectEqual(DType.c128, a.dtype);

    var c = try conj(a);
    defer c.deinit();
    try std.testing.expectEqual(@as(f64, -4.0), (try c.asSlice(C128))[0].im);

    var re = try real(a);
    defer re.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 3.0, 1.0, 0.0 }, try re.asSlice(f64));

    var im = try imag(a);
    defer im.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 4.0, 0.0, 1.0 }, try im.asSlice(f64));

    var mag = try magnitude(a);
    defer mag.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), try mag.get(f64, &.{0}), 1e-12);

    var ph = try phase(a);
    defer ph.deinit();
    try std.testing.expectApproxEqAbs(std.math.atan2(@as(f64, 4.0), @as(f64, 3.0)), try ph.get(f64, &.{0}), 1e-12);

    // conjugate alias
    var c2 = try conjugate(a);
    defer c2.deinit();
    try std.testing.expectEqual((try c.asSlice(C128))[1].re, (try c2.asSlice(C128))[1].re);
}

test "complex rejects non-complex input" {
    const fromSlice = @import("../core/array.zig").fromSlice;
    const allocator = std.testing.allocator;
    const vals = [_]f64{ 1.0, 2.0 };
    var a = try fromSlice(allocator, f64, .{ .data = &vals, .shape = &.{2} });
    defer a.deinit();
    try std.testing.expectError(DTypeError.UnsupportedDType, conj(a));
    try std.testing.expectError(DTypeError.UnsupportedDType, real(a));
}
