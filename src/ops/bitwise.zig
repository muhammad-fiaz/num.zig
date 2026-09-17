//! Integer bitwise operations with broadcast semantics.
//!
//! Canonical integer-only operations: `bitwiseAnd`, `bitwiseOr`, `bitwiseXor`,
//! `bitwiseNot`, `leftShift`, `rightShift`, plus population utilities
//! `bitCount`, `clz` (leading-zero count) and `ctz` (trailing-zero count).
//! Floating-point, boolean and complex dtypes are rejected with
//! `DTypeError.UnsupportedDType`. Shift amounts are taken from the second
//! operand with standard masking semantics.

const std = @import("std");
const Array = @import("../core/array.zig").Array;
const empty = @import("../core/array.zig").empty;
const DType = @import("../core/dtype.zig").DType;
const ShapeError = @import("../core/error.zig").ShapeError;
const DTypeError = @import("../core/error.zig").DTypeError;
const broadcast2 = @import("broadcast.zig").broadcast2;
const BroadcastPairIterator = @import("../core/iterator.zig").BroadcastPairIterator;
const NdIterator = @import("../core/iterator.zig").NdIterator;

pub const BitwiseOp = enum {
    and_,
    or_,
    xor_,
    lshift,
    rshift,
};

fn requireInteger(dtype: DType) DTypeError!void {
    if (!dtype.isInteger()) return DTypeError.UnsupportedDType;
}

fn bitWidth(comptime T: type) u6 {
    return @intCast(@bitSizeOf(T));
}

inline fn applyBitwise(comptime T: type, comptime op: BitwiseOp, a: T, b: T) T {
    const UT = std.meta.Int(.unsigned, @bitSizeOf(T));
    return switch (op) {
        .and_ => a & b,
        .or_ => a | b,
        .xor_ => a ^ b,
        .lshift => blk: {
            const shift: u6 = @intCast(@as(UT, @bitCast(b)) % @as(UT, @bitSizeOf(T)));
            break :blk std.math.shl(T, a, shift);
        },
        .rshift => blk: {
            const shift: u6 = @intCast(@as(UT, @bitCast(b)) % @as(UT, @bitSizeOf(T)));
            break :blk std.math.shr(T, a, shift);
        },
    };
}

fn executeBitwiseKernel(comptime T: type, comptime op: BitwiseOp, a: Array, b: Array, out: Array) !void {
    const total = out.elementCount();
    if (total == 0) return;
    if (a.isContiguous() and b.isContiguous()) {
        const sa = a.asConstSlice(T) catch unreachable;
        const sb = b.asConstSlice(T) catch unreachable;
        const so = out.asSlice(T) catch unreachable;
        var i: usize = 0;
        const VEC = 8;
        // Integer SIMD fast path for and/or/xor.
        if (op == .and_ or op == .or_ or op == .xor_) {
            while (i + VEC <= total) : (i += VEC) {
                const va: @Vector(VEC, T) = sa[i..][0..VEC].*;
                const vb: @Vector(VEC, T) = sb[i..][0..VEC].*;
                const vr: @Vector(VEC, T) = switch (op) {
                    .and_ => va & vb,
                    .or_ => va | vb,
                    .xor_ => va ^ vb,
                    else => unreachable,
                };
                so[i..][0..VEC].* = vr;
            }
        }
        while (i < total) : (i += 1) {
            so[i] = applyBitwise(T, op, sa[i], sb[i]);
        }
        return;
    }
    var it = BroadcastPairIterator.init(out.shape(), a.strides(), b.strides());
    var out_it = NdIterator.init(out.shape(), out.strides());
    while (it.next()) |pair| {
        const out_item = out_it.next().?;
        const va = readIntAt(T, a, pair.offset_a);
        const vb = readIntAt(T, b, pair.offset_b);
        writeIntAt(T, out, out_item.offset, applyBitwise(T, op, va, vb));
    }
}

fn readIntAt(comptime T: type, arr: Array, offset: isize) T {
    const ptr: [*]const T = @ptrCast(@alignCast(arr.data_ptr));
    return ptr[@as(usize, @intCast(offset))];
}

fn writeIntAt(comptime T: type, arr: Array, offset: isize, val: T) void {
    const ptr: [*]T = @ptrCast(@alignCast(arr.data_ptr));
    ptr[@as(usize, @intCast(offset))] = val;
}

pub fn binaryBitwise(a: Array, b: Array, comptime op: BitwiseOp) (ShapeError || DTypeError || std.mem.Allocator.Error)!Array {
    try requireInteger(a.dtype);
    try requireInteger(b.dtype);
    const target_dtype = DType.promote(a.dtype, b.dtype);
    try requireInteger(target_dtype);
    const bc = try broadcast2(a, b);
    var out = try empty(a.allocator, .{ .shape = bc.target_shape.slice(), .dtype = target_dtype });
    errdefer out.deinit();
    inline for (.{ DType.i8, DType.i16, DType.i32, DType.i64, DType.u8, DType.u16, DType.u32, DType.u64 }) |tag| {
        if (target_dtype == tag) {
            const T = tag.toType();
            // Cast operands to target dtype views via strided reads.
            // Fast path requires same dtype; otherwise use generic strided loop with conversion.
            if (a.dtype == target_dtype and b.dtype == target_dtype) {
                try executeBitwiseKernel(T, op, bc.a, bc.b, out);
            } else {
                var it = BroadcastPairIterator.init(out.shape(), a.strides(), b.strides());
                var out_it = NdIterator.init(out.shape(), out.strides());
                const out_ptr: [*]T = @ptrCast(@alignCast(out.data_ptr));
                while (it.next()) |pair| {
                    const out_item = out_it.next().?;
                    const va = readAs(T, a, pair.offset_a);
                    const vb = readAs(T, b, pair.offset_b);
                    out_ptr[@as(usize, @intCast(out_item.offset))] = applyBitwise(T, op, va, vb);
                }
            }
            return out;
        }
    }
    return DTypeError.UnsupportedDType;
}

fn readAs(comptime T: type, arr: Array, offset: isize) T {
    inline for (.{ DType.i8, DType.i16, DType.i32, DType.i64, DType.u8, DType.u16, DType.u32, DType.u64 }) |tag| {
        if (arr.dtype == tag) {
            const S = tag.toType();
            const ptr: [*]const S = @ptrCast(@alignCast(arr.data_ptr));
            const v = ptr[@as(usize, @intCast(offset))];
            return DType.castValue(T, S, v);
        }
    }
    return 0;
}

/// Bitwise AND: `a & b` (integer dtypes only, broadcast).
pub fn bitwiseAnd(a: Array, b: Array) (ShapeError || DTypeError || std.mem.Allocator.Error)!Array {
    return binaryBitwise(a, b, .and_);
}

/// Bitwise OR: `a | b` (integer dtypes only, broadcast).
pub fn bitwiseOr(a: Array, b: Array) (ShapeError || DTypeError || std.mem.Allocator.Error)!Array {
    return binaryBitwise(a, b, .or_);
}

/// Bitwise XOR: `a ^ b` (integer dtypes only, broadcast).
pub fn bitwiseXor(a: Array, b: Array) (ShapeError || DTypeError || std.mem.Allocator.Error)!Array {
    return binaryBitwise(a, b, .xor_);
}

/// Left shift: `a << b` (integer dtypes only, broadcast).
pub fn leftShift(a: Array, b: Array) (ShapeError || DTypeError || std.mem.Allocator.Error)!Array {
    return binaryBitwise(a, b, .lshift);
}

/// Right shift: `a >> b` (integer dtypes only, broadcast; arithmetic for signed, logical for unsigned).
pub fn rightShift(a: Array, b: Array) (ShapeError || DTypeError || std.mem.Allocator.Error)!Array {
    return binaryBitwise(a, b, .rshift);
}

/// Bitwise NOT: `~a` (integer dtypes only). Returns an owned contiguous copy.
pub fn bitwiseNot(a: Array) (ShapeError || DTypeError || std.mem.Allocator.Error)!Array {
    try requireInteger(a.dtype);
    var out = try empty(a.allocator, .{ .shape = a.shapeSlice(), .dtype = a.dtype });
    errdefer out.deinit();
    inline for (.{ DType.i8, DType.i16, DType.i32, DType.i64, DType.u8, DType.u16, DType.u32, DType.u64 }) |tag| {
        if (a.dtype == tag) {
            const T = tag.toType();
            if (a.isContiguous()) {
                const sa = a.asConstSlice(T) catch unreachable;
                const so = out.asSlice(T) catch unreachable;
                for (sa, 0..) |v, i| so[i] = ~v;
            } else {
                var it = NdIterator.init(a.shape(), a.strides());
                var out_it = NdIterator.init(out.shape(), out.strides());
                const in_ptr: [*]const T = @ptrCast(@alignCast(a.data_ptr));
                const out_ptr: [*]T = @ptrCast(@alignCast(out.data_ptr));
                while (it.next()) |item| {
                    const o = out_it.next().?;
                    out_ptr[@as(usize, @intCast(o.offset))] = ~in_ptr[@as(usize, @intCast(item.offset))];
                }
            }
            return out;
        }
    }
    return DTypeError.UnsupportedDType;
}

const UnaryCountOp = enum { popcount, clz, ctz };

fn executeCountKernel(comptime T: type, comptime op: UnaryCountOp, a: Array, out: Array) !void {
    const UT = std.meta.Int(.unsigned, @bitSizeOf(T));
    if (a.isContiguous()) {
        const sa = a.asConstSlice(T) catch unreachable;
        // out dtype matches input width unsigned? We keep same dtype family width: use same T for counts.
        const so = out.asSlice(T) catch unreachable;
        for (sa, 0..) |v, i| {
            const u: UT = @bitCast(v);
            so[i] = switch (op) {
                .popcount => @intCast(@popCount(u)),
                .clz => @intCast(@clz(u)),
                .ctz => @intCast(@ctz(u)),
            };
        }
        return;
    }
    var it = NdIterator.init(a.shape(), a.strides());
    var out_it = NdIterator.init(out.shape(), out.strides());
    const in_ptr: [*]const T = @ptrCast(@alignCast(a.data_ptr));
    const out_ptr: [*]T = @ptrCast(@alignCast(out.data_ptr));
    while (it.next()) |item| {
        const o = out_it.next().?;
        const u: UT = @bitCast(in_ptr[@as(usize, @intCast(item.offset))]);
        out_ptr[@as(usize, @intCast(o.offset))] = switch (op) {
            .popcount => @intCast(@popCount(u)),
            .clz => @intCast(@clz(u)),
            .ctz => @intCast(@ctz(u)),
        };
    }
}

/// Population count (number of set bits) per element. Integer dtypes only; preserves input dtype.
pub fn bitCount(a: Array) (ShapeError || DTypeError || std.mem.Allocator.Error)!Array {
    try requireInteger(a.dtype);
    var out = try empty(a.allocator, .{ .shape = a.shapeSlice(), .dtype = a.dtype });
    errdefer out.deinit();
    inline for (.{ DType.i8, DType.i16, DType.i32, DType.i64, DType.u8, DType.u16, DType.u32, DType.u64 }) |tag| {
        if (a.dtype == tag) {
            try executeCountKernel(tag.toType(), .popcount, a, out);
            return out;
        }
    }
    return DTypeError.UnsupportedDType;
}

/// Leading-zero count per element. Integer dtypes only; preserves input dtype.
pub fn clz(a: Array) (ShapeError || DTypeError || std.mem.Allocator.Error)!Array {
    try requireInteger(a.dtype);
    var out = try empty(a.allocator, .{ .shape = a.shapeSlice(), .dtype = a.dtype });
    errdefer out.deinit();
    inline for (.{ DType.i8, DType.i16, DType.i32, DType.i64, DType.u8, DType.u16, DType.u32, DType.u64 }) |tag| {
        if (a.dtype == tag) {
            try executeCountKernel(tag.toType(), .clz, a, out);
            return out;
        }
    }
    return DTypeError.UnsupportedDType;
}

/// Trailing-zero count per element. Integer dtypes only; preserves input dtype.
pub fn ctz(a: Array) (ShapeError || DTypeError || std.mem.Allocator.Error)!Array {
    try requireInteger(a.dtype);
    var out = try empty(a.allocator, .{ .shape = a.shapeSlice(), .dtype = a.dtype });
    errdefer out.deinit();
    inline for (.{ DType.i8, DType.i16, DType.i32, DType.i64, DType.u8, DType.u16, DType.u32, DType.u64 }) |tag| {
        if (a.dtype == tag) {
            try executeCountKernel(tag.toType(), .ctz, a, out);
            return out;
        }
    }
    return DTypeError.UnsupportedDType;
}

// Descriptive aliases for leading/trailing zero counts.
pub const leadingZeros = clz;
pub const trailingZeros = ctz;
pub const popcount = bitCount;

test "bitwise logic and shifts" {
    const fromSlice = @import("../core/array.zig").fromSlice;
    const allocator = std.testing.allocator;
    const a_vals = [_]i32{ 0b1100, 0b1010, 1, 8 };
    const b_vals = [_]i32{ 0b1010, 0b1100, 2, 1 };
    var a = try fromSlice(allocator, i32, .{ .data = &a_vals, .shape = &.{4} });
    defer a.deinit();
    var b = try fromSlice(allocator, i32, .{ .data = &b_vals, .shape = &.{4} });
    defer b.deinit();

    var and_res = try bitwiseAnd(a, b);
    defer and_res.deinit();
    try std.testing.expectEqualSlices(i32, &.{ 0b1000, 0b1000, 0, 0 }, try and_res.asSlice(i32));

    var or_res = try bitwiseOr(a, b);
    defer or_res.deinit();
    try std.testing.expectEqualSlices(i32, &.{ 0b1110, 0b1110, 3, 9 }, try or_res.asSlice(i32));

    var xor_res = try bitwiseXor(a, b);
    defer xor_res.deinit();
    try std.testing.expectEqualSlices(i32, &.{ 0b0110, 0b0110, 3, 9 }, try xor_res.asSlice(i32));

    var shl = try leftShift(a, b);
    defer shl.deinit();
    try std.testing.expectEqual(@as(i32, 12 << 10), try shl.get(i32, &.{0}));

    var shr = try rightShift(a, b);
    defer shr.deinit();
    try std.testing.expectEqual(@as(i32, 12 >> 10), try shr.get(i32, &.{0}));

    var not_a = try bitwiseNot(a);
    defer not_a.deinit();
    try std.testing.expectEqual(~@as(i32, 12), try not_a.get(i32, &.{0}));
}

test "bitwise broadcast scalar and dtype promotion" {
    const fromSlice = @import("../core/array.zig").fromSlice;
    const allocator = std.testing.allocator;
    const a_vals = [_]u8{ 0xF0, 0x0F };
    var a = try fromSlice(allocator, u8, .{ .data = &a_vals, .shape = &.{2} });
    defer a.deinit();
    const s_vals = [_]u8{0xFF};
    var s = try fromSlice(allocator, u8, .{ .data = &s_vals, .shape = &.{} });
    defer s.deinit();
    var r = try bitwiseAnd(a, s);
    defer r.deinit();
    try std.testing.expectEqualSlices(u8, &.{ 0xF0, 0x0F }, try r.asSlice(u8));
}

test "bitwise rejects float and counts bits" {
    const fromSlice = @import("../core/array.zig").fromSlice;
    const allocator = std.testing.allocator;
    const f_vals = [_]f64{ 1.0, 2.0 };
    var f = try fromSlice(allocator, f64, .{ .data = &f_vals, .shape = &.{2} });
    defer f.deinit();
    var g = try fromSlice(allocator, f64, .{ .data = &f_vals, .shape = &.{2} });
    defer g.deinit();
    try std.testing.expectError(DTypeError.UnsupportedDType, bitwiseAnd(f, g));

    const u_vals = [_]u8{ 0b10110000, 0xFF };
    var u = try fromSlice(allocator, u8, .{ .data = &u_vals, .shape = &.{2} });
    defer u.deinit();
    var pc = try bitCount(u);
    defer pc.deinit();
    try std.testing.expectEqual(@as(u8, 3), try pc.get(u8, &.{0}));
    try std.testing.expectEqual(@as(u8, 8), try pc.get(u8, &.{1}));
    var lz = try clz(u);
    defer lz.deinit();
    try std.testing.expectEqual(@as(u8, 0), try lz.get(u8, &.{0}));
    var tz = try ctz(u);
    defer tz.deinit();
    try std.testing.expectEqual(@as(u8, 4), try tz.get(u8, &.{0}));
}
