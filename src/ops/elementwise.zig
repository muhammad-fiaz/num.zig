//! SIMD-accelerated elementwise mathematical and arithmetic operations.
//!
//! Monomorphized per-dtype compute kernels using Zig's `@Vector` and `std.simd` primitives
//! with automatic broadcasting for incompatible shapes and full NaN/Inf conformance.

const std = @import("std");
const Array = @import("../core/array.zig").Array;
const zeros = @import("../core/array.zig").zeros;
const empty = @import("../core/array.zig").empty;
const DType = @import("../core/dtype.zig").DType;
const Shape = @import("../core/shape.zig").Shape;
const ShapeError = @import("../core/error.zig").ShapeError;
const DTypeError = @import("../core/error.zig").DTypeError;
const broadcast2 = @import("broadcast.zig").broadcast2;
const BroadcastPairIterator = @import("../core/iterator.zig").BroadcastPairIterator;
const NdIterator = @import("../core/iterator.zig").NdIterator;

const VEC_SIZE = 8;

pub const BinaryOp = enum {
    add,
    sub,
    mul,
    div,
    pow,
    remainder,
    minimum,
    maximum,
    atan2,
    hypot,
};

pub const UnaryOp = enum {
    negate,
    abs,
    sqrt,
    cbrt,
    square,
    exp,
    expm1,
    log,
    log1p,
    log2,
    log10,
    sin,
    cos,
    tan,
    asin,
    acos,
    atan,
    sinh,
    cosh,
    tanh,
    asinh,
    acosh,
    atanh,
    floor,
    ceil,
    round,
    trunc,
    sign,
    gamma,
    lgamma,
    erf,
    erfc,
    degreesToRadians,
    radiansToDegrees,
    reciprocal,
    exp2,
};

pub const BinaryOpOptions = struct {
    dtype: ?DType = null,
};

pub const UnaryOpOptions = struct {
    dtype: ?DType = null,
};

/// Core dispatcher for binary operations between arrays `a` and `b`.
pub fn binaryOp(
    a: Array,
    b: Array,
    comptime op: BinaryOp,
    options: BinaryOpOptions,
) (ShapeError || DTypeError || std.mem.Allocator.Error)!Array {
    const bc = try broadcast2(a, b);
    const target_shape = bc.target_shape;
    const target_dtype = options.dtype orelse DType.promote(a.dtype, b.dtype);

    const out = try empty(a.allocator, .{
        .shape = target_shape.slice(),
        .dtype = target_dtype,
    });

    inline for (std.meta.fields(DType)) |field| {
        const tag: DType = @enumFromInt(field.value);
        if (target_dtype == tag) {
            const T = tag.toType();
            try executeBinaryKernel(T, op, bc.a, bc.b, out);
            return out;
        }
    }

    return DTypeError.UnsupportedDType;
}

fn executeBinaryKernel(
    comptime T: type,
    comptime op: BinaryOp,
    a: Array,
    b: Array,
    out: Array,
) (ShapeError || DTypeError)!void {
    const total = out.elementCount();
    if (total == 0) return;

    // Fast path: both contiguous and same dtype
    if (a.isContiguous() and b.isContiguous() and a.dtype == out.dtype and b.dtype == out.dtype) {
        const sa = a.asConstSlice(T) catch unreachable;
        const sb = b.asConstSlice(T) catch unreachable;
        const so = out.asSlice(T) catch unreachable;

        var i: usize = 0;
        if (@typeInfo(T) == .float or @typeInfo(T) == .int) {
            while (i + VEC_SIZE <= total) : (i += VEC_SIZE) {
                const va: @Vector(VEC_SIZE, T) = sa[i..][0..VEC_SIZE].*;
                const vb: @Vector(VEC_SIZE, T) = sb[i..][0..VEC_SIZE].*;

                const vres: @Vector(VEC_SIZE, T) = switch (op) {
                    .add => va + vb,
                    .sub => va - vb,
                    .mul => va * vb,
                    .div => switch (@typeInfo(T)) {
                        .float => va / vb,
                        .int => @divTrunc(va, vb),
                        else => unreachable,
                    },
                    .pow => blk: {
                        var tmp: [VEC_SIZE]T = undefined;
                        inline for (0..VEC_SIZE) |k| {
                            tmp[k] = applyScalarPow(T, va[k], vb[k]);
                        }
                        break :blk tmp;
                    },
                    .remainder => blk: {
                        var tmp: [VEC_SIZE]T = undefined;
                        inline for (0..VEC_SIZE) |k| {
                            tmp[k] = applyScalarBinary(T, .remainder, va[k], vb[k]);
                        }
                        break :blk tmp;
                    },
                    .minimum => blk: {
                        var tmp: [VEC_SIZE]T = undefined;
                        inline for (0..VEC_SIZE) |k| {
                            tmp[k] = applyScalarBinary(T, .minimum, va[k], vb[k]);
                        }
                        break :blk tmp;
                    },
                    .maximum => blk: {
                        var tmp: [VEC_SIZE]T = undefined;
                        inline for (0..VEC_SIZE) |k| {
                            tmp[k] = applyScalarBinary(T, .maximum, va[k], vb[k]);
                        }
                        break :blk tmp;
                    },
                    .atan2 => blk: {
                        var tmp: [VEC_SIZE]T = undefined;
                        inline for (0..VEC_SIZE) |k| {
                            tmp[k] = applyScalarBinary(T, .atan2, va[k], vb[k]);
                        }
                        break :blk tmp;
                    },
                    .hypot => blk: {
                        var tmp: [VEC_SIZE]T = undefined;
                        inline for (0..VEC_SIZE) |k| {
                            tmp[k] = applyScalarBinary(T, .hypot, va[k], vb[k]);
                        }
                        break :blk tmp;
                    },
                };
                so[i..][0..VEC_SIZE].* = vres;
            }
        }

        while (i < total) : (i += 1) {
            so[i] = applyScalarBinary(T, op, sa[i], sb[i]);
        }
        return;
    }

    // Strided / Broadcasted general path
    var it = BroadcastPairIterator.init(out.shape(), a.strides(), b.strides());
    var out_it = NdIterator.init(out.shape(), out.strides());

    while (it.next()) |pair| {
        const out_item = out_it.next().?;
        const va = readScalarAt(T, a, pair.offset_a);
        const vb = readScalarAt(T, b, pair.offset_b);
        const res = applyScalarBinary(T, op, va, vb);
        writeScalarAt(T, out, out_item.offset, res);
    }
}

inline fn applyScalarPow(comptime T: type, base: T, exponent: T) T {
    switch (@typeInfo(T)) {
        .float => {
            if (T == f16) {
                return @floatCast(std.math.pow(f32, @as(f32, base), @as(f32, exponent)));
            }
            return std.math.pow(T, base, exponent);
        },
        .int => {
            if (exponent < 0) return 0;
            return std.math.pow(T, base, exponent);
        },
        .@"struct" => {
            // Complex pow fallback
            return base.mul(exponent);
        },
        else => return if (@typeInfo(T) == .@"struct") T.init(0.0, 0.0) else 0,
    }
}

inline fn applyScalarBinary(comptime T: type, comptime op: BinaryOp, a: T, b: T) T {
    if (T == bool) {
        return switch (op) {
            .add, .mul => a and b,
            else => a,
        };
    }
    if (@typeInfo(T) == .@"struct") {
        return switch (op) {
            .add => a.add(b),
            .sub => a.sub(b),
            .mul => a.mul(b),
            .div => a.div(b),
            .pow => a.mul(b),
            else => a,
        };
    }
    switch (op) {
        .add => return a + b,
        .sub => return a - b,
        .mul => return a * b,
        .div => switch (@typeInfo(T)) {
            .float => return a / b,
            .int => return if (b != 0) @divTrunc(a, b) else 0,
            else => return 0,
        },
        .pow => return applyScalarPow(T, a, b),
        .remainder => switch (@typeInfo(T)) {
            .float => return @mod(a, b),
            .int => return if (b != 0) @rem(a, b) else 0,
            else => return 0,
        },
        .minimum => switch (@typeInfo(T)) {
            .float => return if (std.math.isNan(a) or std.math.isNan(b)) std.math.nan(T) else @min(a, b),
            .int => return @min(a, b),
            else => return a,
        },
        .maximum => switch (@typeInfo(T)) {
            .float => return if (std.math.isNan(a) or std.math.isNan(b)) std.math.nan(T) else @max(a, b),
            .int => return @max(a, b),
            else => return a,
        },
        .atan2 => switch (@typeInfo(T)) {
            .float => return if (T == f16)
                @floatCast(std.math.atan2(@as(f32, @floatCast(a)), @as(f32, @floatCast(b))))
            else
                std.math.atan2(a, b),
            .int => return @intFromFloat(std.math.atan2(@as(f64, @floatFromInt(a)), @as(f64, @floatFromInt(b)))),
            else => return a,
        },
        .hypot => switch (@typeInfo(T)) {
            .float => return std.math.hypot(a, b),
            .int => return @intFromFloat(std.math.hypot(@as(f64, @floatFromInt(a)), @as(f64, @floatFromInt(b)))),
            else => return 0,
        },
    }
}

fn readScalarAt(comptime TargetT: type, arr: Array, offset: isize) TargetT {
    inline for (std.meta.fields(DType)) |field| {
        const tag: DType = @enumFromInt(field.value);
        if (arr.dtype == tag) {
            const SrcT = tag.toType();
            const ptr: [*]const SrcT = @ptrCast(@alignCast(arr.data_ptr));
            const elem = if (offset >= 0) ptr[@as(usize, @intCast(offset))] else ptr[0];
            return castValue(TargetT, SrcT, elem);
        }
    }
    return if (TargetT == bool) false else if (@typeInfo(TargetT) == .@"struct") TargetT.init(0.0, 0.0) else 0;
}

fn writeScalarAt(comptime T: type, arr: Array, offset: isize, val: T) void {
    const ptr: [*]T = @ptrCast(@alignCast(arr.data_ptr));
    if (offset >= 0) {
        ptr[@as(usize, @intCast(offset))] = val;
    }
}

inline fn castValue(comptime DstT: type, comptime SrcT: type, val: SrcT) DstT {
    if (DstT == SrcT) return val;
    return switch (@typeInfo(DstT)) {
        .float => switch (@typeInfo(SrcT)) {
            .int, .comptime_int => @floatFromInt(val),
            .float, .comptime_float => @floatCast(val),
            .bool => if (val) 1.0 else 0.0,
            else => 0.0,
        },
        .int => switch (@typeInfo(SrcT)) {
            .int, .comptime_int => @intCast(val),
            .float, .comptime_float => @intFromFloat(val),
            .bool => if (val) 1 else 0,
            else => 0,
        },
        .bool => switch (@typeInfo(SrcT)) {
            .bool => val,
            .int, .comptime_int => val != 0,
            .float, .comptime_float => val != 0.0,
            .@"struct" => val.re != 0.0 or val.im != 0.0,
            else => false,
        },
        .@"struct" => switch (@typeInfo(SrcT)) {
            .@"struct" => .{ .re = @floatCast(val.re), .im = @floatCast(val.im) },
            .float, .comptime_float => .{ .re = @floatCast(val), .im = 0.0 },
            .int, .comptime_int => .{ .re = @floatFromInt(val), .im = 0.0 },
            .bool => .{ .re = if (val) 1.0 else 0.0, .im = 0.0 },
            else => .{ .re = 0.0, .im = 0.0 },
        },
        else => 0,
    };
}

/// Elementwise addition: `a + b`.
pub fn add(a: Array, b: Array, options: BinaryOpOptions) !Array {
    return binaryOp(a, b, .add, options);
}

/// Elementwise subtraction: `a - b`.
pub fn subtract(a: Array, b: Array, options: BinaryOpOptions) !Array {
    return binaryOp(a, b, .sub, options);
}

/// Elementwise multiplication: `a * b`.
pub fn multiply(a: Array, b: Array, options: BinaryOpOptions) !Array {
    return binaryOp(a, b, .mul, options);
}

/// Elementwise division: `a / b`.
pub fn divide(a: Array, b: Array, options: BinaryOpOptions) !Array {
    return binaryOp(a, b, .div, options);
}

/// Elementwise power: `a ^ b`.
pub fn pow(a: Array, b: Array, options: BinaryOpOptions) !Array {
    return binaryOp(a, b, .pow, options);
}

/// Elementwise remainder: `a % b`.
pub fn remainder(a: Array, b: Array, options: BinaryOpOptions) !Array {
    return binaryOp(a, b, .remainder, options);
}

/// Elementwise minimum of two arrays: `min(a, b)`.
pub fn minimum(a: Array, b: Array, options: BinaryOpOptions) !Array {
    return binaryOp(a, b, .minimum, options);
}

/// Elementwise maximum of two arrays: `max(a, b)`.
pub fn maximum(a: Array, b: Array, options: BinaryOpOptions) !Array {
    return binaryOp(a, b, .maximum, options);
}

// Deliberate shorthand aliases mapping directly to canonical operations
pub const sub = subtract;
pub const mul = multiply;
pub const div = divide;
pub const rem = remainder;

/// Core dispatcher for unary mathematical functions.
pub fn unaryOp(
    a: Array,
    comptime op: UnaryOp,
    options: UnaryOpOptions,
) (ShapeError || DTypeError || std.mem.Allocator.Error)!Array {
    const target_dtype = options.dtype orelse if (op == .abs or op == .sign or op == .negate)
        a.dtype
    else if (a.dtype.isFloat())
        a.dtype
    else
        .f64;

    const out = try empty(a.allocator, .{
        .shape = a.shapeSlice(),
        .dtype = target_dtype,
    });

    inline for (std.meta.fields(DType)) |field| {
        const tag: DType = @enumFromInt(field.value);
        if (target_dtype == tag) {
            const T = tag.toType();
            try executeUnaryKernel(T, op, a, out);
            return out;
        }
    }

    return DTypeError.UnsupportedDType;
}

fn executeUnaryKernel(comptime T: type, comptime op: UnaryOp, a: Array, out: Array) (ShapeError || DTypeError)!void {
    const total = out.elementCount();
    if (total == 0) return;

    if (a.isContiguous() and a.dtype == out.dtype) {
        const sa = a.asConstSlice(T) catch unreachable;
        const so = out.asSlice(T) catch unreachable;

        for (sa, 0..) |elem, i| {
            so[i] = applyScalarUnary(T, op, elem);
        }
        return;
    }

    var it = NdIterator.init(a.shape(), a.strides());
    var out_it = NdIterator.init(out.shape(), out.strides());

    while (it.next()) |item| {
        const out_item = out_it.next().?;
        const va = readScalarAt(T, a, item.offset);
        const res = applyScalarUnary(T, op, va);
        writeScalarAt(T, out, out_item.offset, res);
    }
}

/// Numerical approximation of error function (Chebyshev/rational approximation).
fn scalarErf(x: f64) f64 {
    if (std.math.isNan(x)) return x;
    if (x == 0.0) return 0.0;
    const sign_val: f64 = if (x < 0.0) -1.0 else 1.0;
    const ax = @abs(x);
    if (ax > 6.0) return sign_val;

    // Abramowitz and Stegun formula 7.1.26
    const a1: f64 = 0.254829592;
    const a2: f64 = -0.284496736;
    const a3: f64 = 1.421413741;
    const a4: f64 = -1.453152027;
    const a5: f64 = 1.061405429;
    const p: f64 = 0.3275911;

    const t = 1.0 / (1.0 + p * ax);
    const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * @exp(-ax * ax);
    return sign_val * y;
}

inline fn applyScalarUnary(comptime T: type, comptime op: UnaryOp, val: T) T {
    switch (@typeInfo(T)) {
        .float => switch (op) {
            .negate => return -val,
            .abs => return @abs(val),
            .sqrt => return @sqrt(val),
            .cbrt => return std.math.cbrt(val),
            .square => return val * val,
            .exp => return @exp(val),
            .expm1 => return std.math.expm1(val),
            .log => return @log(val),
            .log1p => return std.math.log1p(val),
            .log2 => return @log2(val),
            .log10 => return @log10(val),
            .sin => return @sin(val),
            .cos => return @cos(val),
            .tan => return @tan(val),
            .asin => return std.math.asin(val),
            .acos => return std.math.acos(val),
            .atan => return std.math.atan(val),
            .sinh => return std.math.sinh(val),
            .cosh => return std.math.cosh(val),
            .tanh => return std.math.tanh(val),
            .asinh => return std.math.asinh(val),
            .acosh => return std.math.acosh(val),
            .atanh => return std.math.atanh(val),
            .floor => return @floor(val),
            .ceil => return @ceil(val),
            .round => return @round(val),
            .trunc => return @trunc(val),
            .sign => return if (std.math.isNan(val)) val else if (val > 0) 1.0 else if (val < 0) -1.0 else 0.0,
            .gamma => {
                if (T == f16) {
                    return @floatCast(std.math.gamma(f32, @floatCast(val)));
                }
                return std.math.gamma(T, val);
            },
            .lgamma => {
                if (T == f16) {
                    return @floatCast(std.math.lgamma(f32, @floatCast(val)));
                }
                return std.math.lgamma(T, val);
            },
            .erf => {
                const vf: f64 = @floatCast(val);
                return @floatCast(scalarErf(vf));
            },
            .erfc => {
                const vf: f64 = @floatCast(val);
                return @floatCast(1.0 - scalarErf(vf));
            },
            .degreesToRadians => {
                return @floatCast(std.math.degreesToRadians(@as(f64, @floatCast(val))));
            },
            .radiansToDegrees => {
                return @floatCast(std.math.radiansToDegrees(@as(f64, @floatCast(val))));
            },
            .reciprocal => {
                return 1.0 / val;
            },
            .exp2 => {
                return @exp2(val);
            },
        },
        .int => switch (op) {
            .negate => return -val,
            .abs => return if (val < 0) -val else val,
            .square => return val * val,
            .sign => return if (val > 0) 1 else if (val < 0) -1 else 0,
            .degreesToRadians => return @intFromFloat(std.math.degreesToRadians(@as(f64, @floatFromInt(val)))),
            .radiansToDegrees => return @intFromFloat(std.math.radiansToDegrees(@as(f64, @floatFromInt(val)))),
            .reciprocal => return if (val != 0) @divTrunc(1, val) else 0,
            .exp2 => return std.math.pow(T, 2, val),
            else => return val,
        },
        else => return val,
    }
}

pub fn negate(a: Array, options: UnaryOpOptions) !Array {
    return unaryOp(a, .negate, options);
}

pub fn abs(a: Array, options: UnaryOpOptions) !Array {
    return unaryOp(a, .abs, options);
}

pub fn sqrt(a: Array, options: UnaryOpOptions) !Array {
    return unaryOp(a, .sqrt, options);
}

pub fn exp(a: Array, options: UnaryOpOptions) !Array {
    return unaryOp(a, .exp, options);
}

pub fn log(a: Array, options: UnaryOpOptions) !Array {
    return unaryOp(a, .log, options);
}

pub fn log2(a: Array, options: UnaryOpOptions) !Array {
    return unaryOp(a, .log2, options);
}

pub fn log10(a: Array, options: UnaryOpOptions) !Array {
    return unaryOp(a, .log10, options);
}

pub fn sin(a: Array, options: UnaryOpOptions) !Array {
    return unaryOp(a, .sin, options);
}

pub fn cos(a: Array, options: UnaryOpOptions) !Array {
    return unaryOp(a, .cos, options);
}

pub fn tan(a: Array, options: UnaryOpOptions) !Array {
    return unaryOp(a, .tan, options);
}

pub fn sinh(a: Array, options: UnaryOpOptions) !Array {
    return unaryOp(a, .sinh, options);
}

pub fn cosh(a: Array, options: UnaryOpOptions) !Array {
    return unaryOp(a, .cosh, options);
}

pub fn tanh(a: Array, options: UnaryOpOptions) !Array {
    return unaryOp(a, .tanh, options);
}

pub fn floor(a: Array, options: UnaryOpOptions) !Array {
    return unaryOp(a, .floor, options);
}

pub fn ceil(a: Array, options: UnaryOpOptions) !Array {
    return unaryOp(a, .ceil, options);
}

pub fn round(a: Array, options: UnaryOpOptions) !Array {
    return unaryOp(a, .round, options);
}

pub fn trunc(a: Array, options: UnaryOpOptions) !Array {
    return unaryOp(a, .trunc, options);
}

pub fn sign(a: Array, options: UnaryOpOptions) !Array {
    return unaryOp(a, .sign, options);
}

pub fn cbrt(a: Array, options: UnaryOpOptions) !Array {
    return unaryOp(a, .cbrt, options);
}

pub fn square(a: Array, options: UnaryOpOptions) !Array {
    return unaryOp(a, .square, options);
}

pub fn expm1(a: Array, options: UnaryOpOptions) !Array {
    return unaryOp(a, .expm1, options);
}

pub fn log1p(a: Array, options: UnaryOpOptions) !Array {
    return unaryOp(a, .log1p, options);
}

pub fn asin(a: Array, options: UnaryOpOptions) !Array {
    return unaryOp(a, .asin, options);
}

pub fn acos(a: Array, options: UnaryOpOptions) !Array {
    return unaryOp(a, .acos, options);
}

pub fn atan(a: Array, options: UnaryOpOptions) !Array {
    return unaryOp(a, .atan, options);
}

pub fn asinh(a: Array, options: UnaryOpOptions) !Array {
    return unaryOp(a, .asinh, options);
}

pub fn acosh(a: Array, options: UnaryOpOptions) !Array {
    return unaryOp(a, .acosh, options);
}

pub fn atanh(a: Array, options: UnaryOpOptions) !Array {
    return unaryOp(a, .atanh, options);
}

pub fn gamma(a: Array, options: UnaryOpOptions) !Array {
    return unaryOp(a, .gamma, options);
}

pub fn lgamma(a: Array, options: UnaryOpOptions) !Array {
    return unaryOp(a, .lgamma, options);
}

pub fn erf(a: Array, options: UnaryOpOptions) !Array {
    return unaryOp(a, .erf, options);
}

pub fn erfc(a: Array, options: UnaryOpOptions) !Array {
    return unaryOp(a, .erfc, options);
}

pub fn degreesToRadians(a: Array, options: UnaryOpOptions) !Array {
    return unaryOp(a, .degreesToRadians, options);
}

pub fn radiansToDegrees(a: Array, options: UnaryOpOptions) !Array {
    return unaryOp(a, .radiansToDegrees, options);
}

pub fn reciprocal(a: Array, options: UnaryOpOptions) !Array {
    return unaryOp(a, .reciprocal, options);
}

pub fn exp2(a: Array, options: UnaryOpOptions) !Array {
    return unaryOp(a, .exp2, options);
}

pub fn atan2(y: Array, x: Array, options: BinaryOpOptions) !Array {
    return binaryOp(y, x, .atan2, options);
}

pub fn hypot(a: Array, b: Array, options: BinaryOpOptions) !Array {
    return binaryOp(a, b, .hypot, options);
}

/// Clips (limits) the values in an array within a specified [min, max] interval.
pub fn clip(
    a: Array,
    options: anytype,
) (ShapeError || DTypeError || std.mem.Allocator.Error)!Array {
    var out = try a.clone();
    inline for (std.meta.fields(DType)) |field| {
        const tag: DType = @enumFromInt(field.value);
        if (out.dtype == tag) {
            const T = tag.toType();
            if (T == bool) return out;
            const slice = out.asSlice(T) catch unreachable;

            const OptType = @TypeOf(options);
            const has_min = comptime blk: {
                if (@hasField(OptType, "min")) {
                    if (@typeInfo(@TypeOf(options.min)) == .optional) {
                        break :blk options.min != null;
                    }
                    break :blk true;
                }
                break :blk false;
            };
            const has_max = comptime blk: {
                if (@hasField(OptType, "max")) {
                    if (@typeInfo(@TypeOf(options.max)) == .optional) {
                        break :blk options.max != null;
                    }
                    break :blk true;
                }
                break :blk false;
            };

            const min_val: ?T = if (has_min) castValue(T, @TypeOf(options.min), options.min) else null;
            const max_val: ?T = if (has_max) castValue(T, @TypeOf(options.max), options.max) else null;

            if (@typeInfo(T) != .@"struct") {
                for (slice) |*elem| {
                    if (@typeInfo(T) == .float and std.math.isNan(elem.*)) continue;
                    if (min_val) |mn| {
                        if (elem.* < mn) elem.* = mn;
                    }
                    if (max_val) |mx| {
                        if (elem.* > mx) elem.* = mx;
                    }
                }
            }

            return out;
        }
    }
    return out;
}

/// Return elements chosen from `x` or `y` depending on boolean `condition`.
pub fn where(
    condition: Array,
    x: Array,
    y: Array,
    options: struct { dtype: ?DType = null },
) (ShapeError || DTypeError || std.mem.Allocator.Error)!Array {
    _ = options;
    const bc_xy = try broadcast2(x, y);
    const bc_all = try broadcast2(condition, bc_xy.a);
    const target_shape = bc_all.target_shape;
    const target_dtype = DType.promote(x.dtype, y.dtype);

    var out = try empty(x.allocator, .{
        .shape = target_shape.slice(),
        .dtype = target_dtype,
    });

    inline for (std.meta.fields(DType)) |field| {
        const tag: DType = @enumFromInt(field.value);
        if (target_dtype == tag) {
            const T = tag.toType();

            var it_c = NdIterator.init(condition.shape(), condition.strides());
            var it_x = NdIterator.init(x.shape(), x.strides());
            var it_y = NdIterator.init(y.shape(), y.strides());
            var it_o = NdIterator.init(out.shape(), out.strides());

            while (it_o.next()) |o_item| {
                const c_item = it_c.next().?;
                const x_item = it_x.next().?;
                const y_item = it_y.next().?;

                const cond_bool = readScalarAt(bool, condition, c_item.offset);
                const selected = if (cond_bool)
                    readScalarAt(T, x, x_item.offset)
                else
                    readScalarAt(T, y, y_item.offset);

                writeScalarAt(T, out, o_item.offset, selected);
            }

            return out;
        }
    }

    return DTypeError.UnsupportedDType;
}

test "binary arithmetic and broadcasting" {
    const allocator = std.testing.allocator;
    const fromSlice = @import("../core/array.zig").fromSlice;

    const data_a = [_]f32{ 1.0, 2.0, 3.0 };
    var a = try fromSlice(allocator, f32, .{ .data = &data_a, .shape = &.{ 1, 3 } });
    defer a.deinit();

    const data_b = [_]f32{ 10.0, 20.0 };
    var b = try fromSlice(allocator, f32, .{ .data = &data_b, .shape = &.{ 2, 1 } });
    defer b.deinit();

    var c = try add(a, b, .{});
    defer c.deinit();

    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, c.shapeSlice());
    const c_slice = try c.asSlice(f32);
    // [ [11, 12, 13], [21, 22, 23] ]
    try std.testing.expectEqualSlices(f32, &.{ 11.0, 12.0, 13.0, 21.0, 22.0, 23.0 }, c_slice);

    // Test aliases sub, mul, div, rem
    var s_canon = try subtract(b, a, .{});
    defer s_canon.deinit();
    var s_alias = try sub(b, a, .{});
    defer s_alias.deinit();
    try std.testing.expectEqualSlices(f32, try s_canon.asSlice(f32), try s_alias.asSlice(f32));

    var m_canon = try multiply(a, b, .{});
    defer m_canon.deinit();
    var m_alias = try mul(a, b, .{});
    defer m_alias.deinit();
    try std.testing.expectEqualSlices(f32, try m_canon.asSlice(f32), try m_alias.asSlice(f32));

    var d_canon = try divide(b, a, .{});
    defer d_canon.deinit();
    var d_alias = try div(b, a, .{});
    defer d_alias.deinit();
    try std.testing.expectEqualSlices(f32, try d_canon.asSlice(f32), try d_alias.asSlice(f32));

    var r_canon = try remainder(b, a, .{});
    defer r_canon.deinit();
    var r_alias = try rem(b, a, .{});
    defer r_alias.deinit();
    try std.testing.expectEqualSlices(f32, try r_canon.asSlice(f32), try r_alias.asSlice(f32));
}

test "unary math and clip" {
    const allocator = std.testing.allocator;
    const fromSlice = @import("../core/array.zig").fromSlice;

    const data = [_]f64{ 1.0, 4.0, 9.0, 16.0 };
    var a = try fromSlice(allocator, f64, .{ .data = &data, .shape = &.{4} });
    defer a.deinit();

    var s = try sqrt(a, .{});
    defer s.deinit();

    const s_slice = try s.asSlice(f64);
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 2.0, 3.0, 4.0 }, s_slice);

    var cl = try clip(a, .{ .min = 3.0, .max = 10.0 });
    defer cl.deinit();

    const cl_slice = try cl.asSlice(f64);
    try std.testing.expectEqualSlices(f64, &.{ 3.0, 4.0, 9.0, 10.0 }, cl_slice);

    // atan2 and hypot
    const data_y = [_]f64{ 3.0, 0.0 };
    const data_x = [_]f64{ 4.0, 1.0 };
    var arr_y = try fromSlice(allocator, f64, .{ .data = &data_y, .shape = &.{2} });
    defer arr_y.deinit();
    var arr_x = try fromSlice(allocator, f64, .{ .data = &data_x, .shape = &.{2} });
    defer arr_x.deinit();

    var h = try hypot(arr_y, arr_x, .{});
    defer h.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), try h.get(f64, &.{0}), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), try h.get(f64, &.{1}), 1e-5);

    var at = try atan2(arr_y, arr_x, .{});
    defer at.deinit();
    try std.testing.expectApproxEqAbs(std.math.atan2(@as(f64, 3.0), @as(f64, 4.0)), try at.get(f64, &.{0}), 1e-5);

    // erf and erfc
    const erf_in = [_]f64{ 0.0, 1.0 };
    var arr_erf = try fromSlice(allocator, f64, .{ .data = &erf_in, .shape = &.{2} });
    defer arr_erf.deinit();

    var erf_out = try erf(arr_erf, .{});
    defer erf_out.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), try erf_out.get(f64, &.{0}), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f64, 0.8427), try erf_out.get(f64, &.{1}), 1e-3);

    var erfc_out = try erfc(arr_erf, .{});
    defer erfc_out.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), try erfc_out.get(f64, &.{0}), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 - 0.8427), try erfc_out.get(f64, &.{1}), 1e-3);

    // degreesToRadians
    const deg_in = [_]f64{ 180.0, 90.0 };
    var arr_deg = try fromSlice(allocator, f64, .{ .data = &deg_in, .shape = &.{2} });
    defer arr_deg.deinit();
    var rad_out = try degreesToRadians(arr_deg, .{});
    defer rad_out.deinit();
    try std.testing.expectApproxEqAbs(std.math.pi, try rad_out.get(f64, &.{0}), 1e-5);
    try std.testing.expectApproxEqAbs(std.math.pi / 2.0, try rad_out.get(f64, &.{1}), 1e-5);

    // reciprocal and exp2
    var recip = try reciprocal(a, .{});
    defer recip.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 4.0), try recip.get(f64, &.{1}), 1e-5);

    const pow2_data = [_]f64{ 0.0, 1.0, 2.0, 3.0 };
    var p2_arr = try fromSlice(allocator, f64, .{ .data = &pow2_data, .shape = &.{4} });
    defer p2_arr.deinit();
    var e2 = try exp2(p2_arr, .{});
    defer e2.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), try e2.get(f64, &.{0}), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), try e2.get(f64, &.{1}), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), try e2.get(f64, &.{2}), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f64, 8.0), try e2.get(f64, &.{3}), 1e-5);
}
