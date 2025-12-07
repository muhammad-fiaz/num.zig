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
