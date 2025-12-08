const std = @import("std");
const core = @import("core.zig");
const NDArray = core.NDArray;
const Allocator = std.mem.Allocator;
pub const Complex = std.math.Complex;

/// Return the real part of the complex argument.
pub fn real(allocator: Allocator, comptime T: type, a: NDArray(Complex(T))) !NDArray(T) {
    var result = try NDArray(T).init(allocator, a.shape);
    for (a.data, 0..) |val, i| {
        result.data[i] = val.re;
    }
    return result;
}

/// Return the imaginary part of the complex argument.
pub fn imag(allocator: Allocator, comptime T: type, a: NDArray(Complex(T))) !NDArray(T) {
    var result = try NDArray(T).init(allocator, a.shape);
    for (a.data, 0..) |val, i| {
        result.data[i] = val.im;
    }
    return result;
}

/// Return the complex conjugate, element-wise.
pub fn conj(allocator: Allocator, comptime T: type, a: NDArray(Complex(T))) !NDArray(Complex(T)) {
    var result = try NDArray(Complex(T)).init(allocator, a.shape);
    for (a.data, 0..) |val, i| {
        result.data[i] = Complex(T).init(val.re, -val.im);
    }
    return result;
}

/// Return the angle of the complex argument.
pub fn angle(allocator: Allocator, comptime T: type, a: NDArray(Complex(T))) !NDArray(T) {
    var result = try NDArray(T).init(allocator, a.shape);
    for (a.data, 0..) |val, i| {
        result.data[i] = std.math.atan2(val.im, val.re);
    }
    return result;
}

/// Return the absolute value (magnitude) of the complex argument.
pub fn abs(allocator: Allocator, comptime T: type, a: NDArray(Complex(T))) !NDArray(T) {
    var result = try NDArray(T).init(allocator, a.shape);
    for (a.data, 0..) |val, i| {
        result.data[i] = val.magnitude();
    }
    return result;
}

test "complex operations" {
    const allocator = std.testing.allocator;
    var c_arr = try NDArray(Complex(f32)).init(allocator, &.{2});
    defer c_arr.deinit(allocator);
    c_arr.data[0] = Complex(f32).init(3.0, 4.0);
    c_arr.data[1] = Complex(f32).init(1.0, -1.0);

    var r = try real(allocator, f32, c_arr);
    defer r.deinit(allocator);
    try std.testing.expectEqual(r.data[0], 3.0);
    try std.testing.expectEqual(r.data[1], 1.0);

    var i = try imag(allocator, f32, c_arr);
    defer i.deinit(allocator);
    try std.testing.expectEqual(i.data[0], 4.0);
    try std.testing.expectEqual(i.data[1], -1.0);

    var c = try conj(allocator, f32, c_arr);
    defer c.deinit(allocator);
    try std.testing.expectEqual(c.data[0].im, -4.0);

    var a = try abs(allocator, f32, c_arr);
    defer a.deinit(allocator);
    try std.testing.expectEqual(a.data[0], 5.0);
}
