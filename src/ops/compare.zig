//! Vectorized relational and logical comparison operations.
//!
//! Evaluates elementwise equality, inequality, ordered comparisons, and boolean logic
//! producing boolean mask arrays with automatic broadcasting.

const std = @import("std");
const Array = @import("../core/array.zig").Array;
const empty = @import("../core/array.zig").empty;
const DType = @import("../core/dtype.zig").DType;
const Shape = @import("../core/shape.zig").Shape;
const ShapeError = @import("../core/error.zig").ShapeError;
const DTypeError = @import("../core/error.zig").DTypeError;
const broadcast2 = @import("broadcast.zig").broadcast2;
const BroadcastPairIterator = @import("../core/iterator.zig").BroadcastPairIterator;
const NdIterator = @import("../core/iterator.zig").NdIterator;

const VEC_SIZE = 8;

pub const CompareOp = enum {
    eq,
    ne,
    lt,
    le,
    gt,
    ge,
    logical_and,
    logical_or,
    logical_xor,
};

/// Core comparison dispatcher returning an Array of dtype `.bool`.
pub fn compareOp(
    a: Array,
    b: Array,
    comptime op: CompareOp,
) (ShapeError || DTypeError || std.mem.Allocator.Error)!Array {
    const bc = try broadcast2(a, b);
    const target_shape = bc.target_shape;
    const common_dtype = DType.promote(a.dtype, b.dtype);

    const out = try empty(a.allocator, .{
        .shape = target_shape.slice(),
        .dtype = .bool,
    });

    inline for (std.meta.fields(DType)) |field| {
        const tag: DType = @enumFromInt(field.value);
        if (common_dtype == tag) {
            const T = tag.toType();
            try executeCompareKernel(T, op, bc.a, bc.b, out);
            return out;
        }
    }

    return DTypeError.UnsupportedDType;
}

fn executeCompareKernel(
    comptime T: type,
    comptime op: CompareOp,
    a: Array,
    b: Array,
    out: Array,
) (ShapeError || DTypeError)!void {
    const total = out.elementCount();
    if (total == 0) return;

    if (a.isContiguous() and b.isContiguous() and a.dtype == b.dtype) {
        const sa = a.asConstSlice(T) catch unreachable;
        const sb = b.asConstSlice(T) catch unreachable;
        const so = out.asSlice(bool) catch unreachable;

        var i: usize = 0;
        if ((@typeInfo(T) == .float or @typeInfo(T) == .int) and @sizeOf(T) >= 1) {
            while (i + VEC_SIZE <= total) : (i += VEC_SIZE) {
                const va: @Vector(VEC_SIZE, T) = sa[i..][0..VEC_SIZE].*;
                const vb: @Vector(VEC_SIZE, T) = sb[i..][0..VEC_SIZE].*;

                const vres: @Vector(VEC_SIZE, bool) = switch (op) {
                    .eq => va == vb,
                    .ne => va != vb,
                    .lt => va < vb,
                    .le => va <= vb,
                    .gt => va > vb,
                    .ge => va >= vb,
                    .logical_and => (va != @as(@Vector(VEC_SIZE, T), @splat(0))) and (vb != @as(@Vector(VEC_SIZE, T), @splat(0))),
                    .logical_or => (va != @as(@Vector(VEC_SIZE, T), @splat(0))) or (vb != @as(@Vector(VEC_SIZE, T), @splat(0))),
                    .logical_xor => (va != @as(@Vector(VEC_SIZE, T), @splat(0))) != (vb != @as(@Vector(VEC_SIZE, T), @splat(0))),
                };
                so[i..][0..VEC_SIZE].* = vres;
            }
        }

        while (i < total) : (i += 1) {
            so[i] = applyScalarCompare(T, op, sa[i], sb[i]);
        }
        return;
    }

    var it = BroadcastPairIterator.init(out.shape(), a.strides(), b.strides());
    var out_it = NdIterator.init(out.shape(), out.strides());

    while (it.next()) |pair| {
        const out_item = out_it.next().?;
        const va = readScalar(T, a, pair.offset_a);
        const vb = readScalar(T, b, pair.offset_b);
        const res = applyScalarCompare(T, op, va, vb);
        writeScalar(out, out_item.offset, res);
    }
}

inline fn applyScalarCompare(comptime T: type, comptime op: CompareOp, a: T, b: T) bool {
    if (T == bool) {
        return switch (op) {
            .eq => a == b,
            .ne => a != b,
            .lt => !a and b,
            .le => !a or b,
            .gt => a and !b,
            .ge => a or !b,
            .logical_and => a and b,
            .logical_or => a or b,
            .logical_xor => a != b,
        };
    }
    if (@typeInfo(T) == .@"struct") {
        const eq = (a.re == b.re and a.im == b.im);
        return switch (op) {
            .eq => eq,
            .ne => !eq,
            .logical_and => (a.re != 0.0 or a.im != 0.0) and (b.re != 0.0 or b.im != 0.0),
            .logical_or => (a.re != 0.0 or a.im != 0.0) or (b.re != 0.0 or b.im != 0.0),
            .logical_xor => (a.re != 0.0 or a.im != 0.0) != (b.re != 0.0 or b.im != 0.0),
            else => false,
        };
    }
    return switch (op) {
        .eq => a == b,
        .ne => a != b,
        .lt => a < b,
        .le => a <= b,
        .gt => a > b,
        .ge => a >= b,
        .logical_and => (a != 0) and (b != 0),
        .logical_or => (a != 0) or (b != 0),
        .logical_xor => (a != 0) != (b != 0),
    };
}

fn readScalar(comptime TargetT: type, arr: Array, offset: isize) TargetT {
    inline for (std.meta.fields(DType)) |field| {
        const tag: DType = @enumFromInt(field.value);
        if (arr.dtype == tag) {
            const SrcT = tag.toType();
            const ptr: [*]const SrcT = @ptrCast(@alignCast(arr.data_ptr));
            const elem = if (offset >= 0) ptr[@as(usize, @intCast(offset))] else ptr[0];
            if (TargetT == SrcT) return elem;
            return switch (@typeInfo(TargetT)) {
                .float => switch (@typeInfo(SrcT)) {
                    .int, .comptime_int => @floatFromInt(elem),
                    .float, .comptime_float => @floatCast(elem),
                    .bool => if (elem) 1.0 else 0.0,
                    .@"struct" => @floatCast(elem.re),
                    else => 0.0,
                },
                .int => switch (@typeInfo(SrcT)) {
                    .int, .comptime_int => @intCast(elem),
                    .float, .comptime_float => @intFromFloat(elem),
                    .bool => if (elem) 1 else 0,
                    .@"struct" => @intFromFloat(elem.re),
                    else => 0,
                },
                .bool => switch (@typeInfo(SrcT)) {
                    .bool => elem,
                    .int, .comptime_int => elem != 0,
                    .float, .comptime_float => elem != 0.0,
                    .@"struct" => elem.re != 0.0 or elem.im != 0.0,
                    else => false,
                },
                .@"struct" => switch (@typeInfo(SrcT)) {
                    .@"struct" => .{ .re = @floatCast(elem.re), .im = @floatCast(elem.im) },
                    .float, .comptime_float => .{ .re = @floatCast(elem), .im = 0.0 },
                    .int, .comptime_int => .{ .re = @floatFromInt(elem), .im = 0.0 },
                    .bool => .{ .re = if (elem) 1.0 else 0.0, .im = 0.0 },
                    else => .{ .re = 0.0, .im = 0.0 },
                },
                else => 0,
            };
        }
    }
    return if (TargetT == bool) false else if (@typeInfo(TargetT) == .@"struct") TargetT.init(0.0, 0.0) else 0;
}

fn writeScalar(arr: Array, offset: isize, val: bool) void {
    const ptr: [*]bool = @ptrCast(@alignCast(arr.data_ptr));
    if (offset >= 0) {
        ptr[@as(usize, @intCast(offset))] = val;
    }
}

pub fn equal(a: Array, b: Array) !Array {
    return compareOp(a, b, .eq);
}

pub fn notEqual(a: Array, b: Array) !Array {
    return compareOp(a, b, .ne);
}

pub fn less(a: Array, b: Array) !Array {
    return compareOp(a, b, .lt);
}

pub fn lessEqual(a: Array, b: Array) !Array {
    return compareOp(a, b, .le);
}

pub fn greater(a: Array, b: Array) !Array {
    return compareOp(a, b, .gt);
}

pub fn greaterEqual(a: Array, b: Array) !Array {
    return compareOp(a, b, .ge);
}

pub fn logicalAnd(a: Array, b: Array) !Array {
    return compareOp(a, b, .logical_and);
}

pub fn logicalOr(a: Array, b: Array) !Array {
    return compareOp(a, b, .logical_or);
}

pub fn logicalXor(a: Array, b: Array) !Array {
    return compareOp(a, b, .logical_xor);
}

/// Computes truth value of NOT x elementwise on an array.
pub fn logicalNot(a: Array) (ShapeError || DTypeError || std.mem.Allocator.Error)!Array {
    const out = try empty(a.allocator, .{
        .shape = a.shapeSlice(),
        .dtype = .bool,
    });

    var it = NdIterator.init(a.shape(), a.strides());
    var out_it = NdIterator.init(out.shape(), out.strides());

    while (it.next()) |item| {
        const out_item = out_it.next().?;
        const v = readScalar(bool, a, item.offset);
        writeScalar(out, out_item.offset, !v);
    }

    return out;
}

/// Test elementwise for NaN.
pub fn isNaN(a: Array) (ShapeError || std.mem.Allocator.Error)!Array {
    const out = try empty(a.allocator, .{
        .shape = a.shapeSlice(),
        .dtype = .bool,
    });

    var it = NdIterator.init(a.shape(), a.strides());
    var out_it = NdIterator.init(out.shape(), out.strides());

    while (it.next()) |item| {
        const out_item = out_it.next().?;
        const v = a.getAsFloat(item.indices[0..a.ndim]) catch 0.0;
        writeScalar(out, out_item.offset, std.math.isNan(v));
    }

    return out;
}

/// Test elementwise for positive or negative infinity.
pub fn isInf(a: Array) (ShapeError || std.mem.Allocator.Error)!Array {
    const out = try empty(a.allocator, .{
        .shape = a.shapeSlice(),
        .dtype = .bool,
    });

    var it = NdIterator.init(a.shape(), a.strides());
    var out_it = NdIterator.init(out.shape(), out.strides());

    while (it.next()) |item| {
        const out_item = out_it.next().?;
        const v = a.getAsFloat(item.indices[0..a.ndim]) catch 0.0;
        writeScalar(out, out_item.offset, std.math.isInf(v));
    }

    return out;
}

/// Test elementwise for finiteness (not NaN and not Inf).
pub fn isFinite(a: Array) (ShapeError || std.mem.Allocator.Error)!Array {
    const out = try empty(a.allocator, .{
        .shape = a.shapeSlice(),
        .dtype = .bool,
    });

    var it = NdIterator.init(a.shape(), a.strides());
    var out_it = NdIterator.init(out.shape(), out.strides());

    while (it.next()) |item| {
        const out_item = out_it.next().?;
        const v = a.getAsFloat(item.indices[0..a.ndim]) catch 0.0;
        writeScalar(out, out_item.offset, std.math.isFinite(v));
    }

    return out;
}

pub const IsCloseOptions = struct {
    rtol: f64 = 1e-5,
    atol: f64 = 1e-8,
    equalNan: bool = false,
};

/// Returns a boolean array where two arrays are elementwise equal within a tolerance.
pub fn isClose(
    a: Array,
    b: Array,
    options: IsCloseOptions,
) (ShapeError || DTypeError || std.mem.Allocator.Error)!Array {
    const bc = try broadcast2(a, b);
    const target_shape = bc.target_shape;

    const out = try empty(a.allocator, .{
        .shape = target_shape.slice(),
        .dtype = .bool,
    });

    var it = BroadcastPairIterator.init(target_shape, bc.a.strides(), bc.b.strides());
    var out_it = NdIterator.init(out.shape(), out.strides());

    while (it.next()) |item| {
        const out_item = out_it.next().?;
        const va = readScalar(f64, bc.a, item.offset_a);
        const vb = readScalar(f64, bc.b, item.offset_b);

        var close = false;
        if (std.math.isNan(va) or std.math.isNan(vb)) {
            close = options.equalNan and std.math.isNan(va) and std.math.isNan(vb);
        } else if (std.math.isInf(va) or std.math.isInf(vb)) {
            close = (va == vb);
        } else {
            const diff = @abs(va - vb);
            close = diff <= (options.atol + options.rtol * @abs(vb));
        }

        writeScalar(out, out_item.offset, close);
    }

    return out;
}

/// Returns true if two arrays are elementwise equal within a tolerance.
pub fn allClose(
    a: Array,
    b: Array,
    options: IsCloseOptions,
) (ShapeError || DTypeError || std.mem.Allocator.Error)!bool {
    var close_mask = try isClose(a, b, options);
    defer close_mask.deinit();

    var it = NdIterator.init(close_mask.shape(), close_mask.strides());
    while (it.next()) |item| {
        const val = readScalar(bool, close_mask, item.offset);
        if (!val) return false;
    }

    return true;
}

test "comparisons and broadcasting" {
    const allocator = std.testing.allocator;
    const fromSlice = @import("../core/array.zig").fromSlice;

    const data_a = [_]f32{ 1.0, 5.0, 10.0 };
    var a = try fromSlice(allocator, f32, .{ .data = &data_a, .shape = &.{3} });
    defer a.deinit();

    const data_b = [_]f32{5.0};
    var b = try fromSlice(allocator, f32, .{ .data = &data_b, .shape = &.{1} });
    defer b.deinit();

    var eq_res = try equal(a, b);
    defer eq_res.deinit();

    try std.testing.expectEqualSlices(bool, &.{ false, true, false }, try eq_res.asSlice(bool));

    var lt_res = try less(a, b);
    defer lt_res.deinit();

    try std.testing.expectEqualSlices(bool, &.{ true, false, false }, try lt_res.asSlice(bool));

    // isNaN, isInf, isFinite
    const nan_data = [_]f64{ 1.0, std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64) };
    var f_arr = try fromSlice(allocator, f64, .{ .data = &nan_data, .shape = &.{4} });
    defer f_arr.deinit();

    var nan_mask = try isNaN(f_arr);
    defer nan_mask.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false }, try nan_mask.asSlice(bool));

    var inf_mask = try isInf(f_arr);
    defer inf_mask.deinit();
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, true }, try inf_mask.asSlice(bool));

    var fin_mask = try isFinite(f_arr);
    defer fin_mask.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, false }, try fin_mask.asSlice(bool));

    // isClose and allClose
    const near_data1 = [_]f64{ 1.0, 2.0, 3.0 };
    const near_data2 = [_]f64{ 1.000001, 1.999999, 3.000005 };
    var n1 = try fromSlice(allocator, f64, .{ .data = &near_data1, .shape = &.{3} });
    defer n1.deinit();
    var n2 = try fromSlice(allocator, f64, .{ .data = &near_data2, .shape = &.{3} });
    defer n2.deinit();

    try std.testing.expect(try allClose(n1, n2, .{ .atol = 1e-4 }));
}
