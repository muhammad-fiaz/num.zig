//! Axis-aware multidimensional reduction operations.
//!
//! Provides global and per-axis reductions (sum, prod, min, max, mean, argmin, argmax,
//! all, any, cumsum, cumprod) with SIMD `@reduce` vector hardware acceleration and keepDims support.

const std = @import("std");
const Array = @import("../core/array.zig").Array;
const zeros = @import("../core/array.zig").zeros;
const empty = @import("../core/array.zig").empty;
const DType = @import("../core/dtype.zig").DType;
const Shape = @import("../core/shape.zig").Shape;
const Strides = @import("../core/shape.zig").Strides;
const MAX_RANK = @import("../core/shape.zig").MAX_RANK;
const ShapeError = @import("../core/error.zig").ShapeError;
const DTypeError = @import("../core/error.zig").DTypeError;
const NdIterator = @import("../core/iterator.zig").NdIterator;

const VEC_SIZE = 8;

const ReduceOp = enum {
    sum,
    prod,
    min,
    max,
    all,
    any,
};

const ReduceOptions = struct {
    axis: ?isize = null,
    keepDims: bool = false,
    dtype: ?DType = null,
};

const ArgOptions = struct {
    axis: ?isize = null,
    keepDims: bool = false,
};

const CumOptions = struct {
    axis: ?isize = null,
    dtype: ?DType = null,
};

const CumMinMaxOptions = struct {
    axis: ?isize = null,
};

const DiffOptions = struct {
    n: usize = 1,
    axis: ?isize = null,
};

inline fn optAxis(options: anytype) ?isize {
    const T = @TypeOf(options);
    if (@typeInfo(T) != .@"struct") return null;
    if (!@hasField(T, "axis")) return null;
    const ax = options.axis;
    if (ax == null) return null;
    return @intCast(ax.?);
}

inline fn optKeepDims(options: anytype) bool {
    const T = @TypeOf(options);
    if (@typeInfo(T) != .@"struct") return false;
    if (!@hasField(T, "keepDims")) return false;
    return options.keepDims;
}

inline fn optDtype(options: anytype) ?DType {
    const T = @TypeOf(options);
    if (@typeInfo(T) != .@"struct") return null;
    if (!@hasField(T, "dtype")) return null;
    return options.dtype;
}

/// Computes the sum of array elements over a given axis or globally.
/// Optional inline config: `.{ .axis = 0, .keepDims = false, .dtype = .f64 }`; may be omitted.
pub fn sum(arr: Array, options: ReduceOptions) (ShapeError || DTypeError || std.mem.Allocator.Error)!Array {
    const out_dtype = optDtype(options) orelse switch (arr.dtype) {
        .bool, .i8, .i16, .i32 => .i64,
        .u8, .u16, .u32 => .u64,
        else => arr.dtype,
    };
    return executeReduction(arr, .sum, optAxis(options), optKeepDims(options), out_dtype);
}

/// Computes the product of array elements over a given axis or globally.
/// Optional inline config may be omitted.
pub fn prod(arr: Array, options: ReduceOptions) (ShapeError || DTypeError || std.mem.Allocator.Error)!Array {
    const out_dtype = optDtype(options) orelse switch (arr.dtype) {
        .bool, .i8, .i16, .i32 => .i64,
        .u8, .u16, .u32 => .u64,
        else => arr.dtype,
    };
    return executeReduction(arr, .prod, optAxis(options), optKeepDims(options), out_dtype);
}

/// Finds the minimum value over a given axis or globally.
pub fn min(arr: Array, options: ReduceOptions) (ShapeError || DTypeError || std.mem.Allocator.Error)!Array {
    if (arr.elementCount() == 0) return ShapeError.EmptyArray;
    const out_dtype = optDtype(options) orelse arr.dtype;
    return executeReduction(arr, .min, optAxis(options), optKeepDims(options), out_dtype);
}

/// Finds the maximum value over a given axis or globally.
pub fn max(arr: Array, options: ReduceOptions) (ShapeError || DTypeError || std.mem.Allocator.Error)!Array {
    if (arr.elementCount() == 0) return ShapeError.EmptyArray;
    const out_dtype = optDtype(options) orelse arr.dtype;
    return executeReduction(arr, .max, optAxis(options), optKeepDims(options), out_dtype);
}

/// Computes the arithmetic mean over a given axis or globally.
pub fn mean(arr: Array, options: ReduceOptions) (ShapeError || DTypeError || std.mem.Allocator.Error)!Array {
    if (arr.elementCount() == 0) return ShapeError.EmptyArray;
    const float_dtype: DType = optDtype(options) orelse (if (arr.dtype == .f32) DType.f32 else DType.f64);

    var s = try sum(arr, .{
        .axis = optAxis(options),
        .keepDims = optKeepDims(options),
        .dtype = float_dtype,
    });
    errdefer s.deinit();

    const count_dim: usize = if (optAxis(options)) |ax| blk: {
        const norm_ax = try arr.shape().normalizeAxis(ax);
        break :blk arr.shape_dims[norm_ax];
    } else arr.elementCount();

    const count_f: f64 = @floatFromInt(count_dim);

    inline for (std.meta.fields(DType)) |field| {
        const tag: DType = @enumFromInt(field.value);
        if (s.dtype == tag) {
            const T = tag.toType();
            const slice = s.asSlice(T) catch unreachable;
            for (slice) |*val| {
                const cur_f: f64 = switch (@typeInfo(T)) {
                    .float => @floatCast(val.*),
                    .int => @floatFromInt(val.*),
                    else => 0.0,
                };
                val.* = switch (@typeInfo(T)) {
                    .float => @floatCast(cur_f / count_f),
                    .int => @intFromFloat(cur_f / count_f),
                    else => unreachable,
                };
            }
            return s;
        }
    }

    return s;
}

/// Computes the median over a given axis or globally.
/// Returns f32 for f32 inputs, otherwise f64. Supports keepDims and negative axes.
pub fn median(arr: Array, options: ReduceOptions) (ShapeError || DTypeError || std.mem.Allocator.Error)!Array {
    if (arr.elementCount() == 0) return ShapeError.EmptyArray;
    if (arr.dtype.isComplex() or arr.dtype == .bool) return DTypeError.UnsupportedDType;
    const float_dtype: DType = optDtype(options) orelse (if (arr.dtype == .f32) DType.f32 else DType.f64);
    return executeStatReduction(arr, .median, optAxis(options), optKeepDims(options), float_dtype);
}

/// Computes the population variance over a given axis or globally.
/// Returns f32 for f32 inputs, otherwise f64.
pub fn variance(arr: Array, options: ReduceOptions) (ShapeError || DTypeError || std.mem.Allocator.Error)!Array {
    if (arr.elementCount() == 0) return ShapeError.EmptyArray;
    if (arr.dtype.isComplex() or arr.dtype == .bool) return DTypeError.UnsupportedDType;
    const float_dtype: DType = optDtype(options) orelse (if (arr.dtype == .f32) DType.f32 else DType.f64);
    return executeStatReduction(arr, .variance, optAxis(options), optKeepDims(options), float_dtype);
}

/// Computes the population standard deviation over a given axis or globally.
pub fn stdDev(arr: Array, options: ReduceOptions) (ShapeError || DTypeError || std.mem.Allocator.Error)!Array {
    var v = try variance(arr, .{ .axis = optAxis(options), .keepDims = optKeepDims(options), .dtype = optDtype(options) });
    errdefer v.deinit();
    const n = v.elementCount();
    if (v.dtype == .f32) {
        const s = try v.asSlice(f32);
        for (s) |*x| x.* = @sqrt(x.*);
    } else {
        const s = try v.asSlice(f64);
        for (s) |*x| x.* = @sqrt(x.*);
    }
    _ = n;
    return v;
}

/// Tests whether all elements evaluate to true over a given axis or globally.
pub fn all(arr: Array, options: ArgOptions) (ShapeError || DTypeError || std.mem.Allocator.Error)!Array {
    return executeReduction(arr, .all, optAxis(options), optKeepDims(options), .bool);
}

/// Tests whether any element evaluates to true over a given axis or globally.
pub fn any(arr: Array, options: ArgOptions) (ShapeError || DTypeError || std.mem.Allocator.Error)!Array {
    return executeReduction(arr, .any, optAxis(options), optKeepDims(options), .bool);
}

/// Returns the indices of the minimum values along an axis.
pub fn argmin(arr: Array, options: ArgOptions) (ShapeError || DTypeError || std.mem.Allocator.Error)!Array {
    return executeArgReduction(arr, false, optAxis(options), optKeepDims(options));
}

/// Returns the indices of the maximum values along an axis.
pub fn argmax(arr: Array, options: ArgOptions) (ShapeError || DTypeError || std.mem.Allocator.Error)!Array {
    return executeArgReduction(arr, true, optAxis(options), optKeepDims(options));
}

/// Cumulative sum of elements along a given axis.
pub fn cumsum(arr: Array, options: CumOptions) (ShapeError || DTypeError || std.mem.Allocator.Error)!Array {
    const out_dtype = optDtype(options) orelse switch (arr.dtype) {
        .bool, .i8, .i16, .i32 => .i64,
        .u8, .u16, .u32 => .u64,
        else => arr.dtype,
    };
    return executeCumulative(arr, .sum, optAxis(options), out_dtype);
}

/// Cumulative product of elements along a given axis.
pub fn cumprod(arr: Array, options: CumOptions) (ShapeError || DTypeError || std.mem.Allocator.Error)!Array {
    const out_dtype = optDtype(options) orelse switch (arr.dtype) {
        .bool, .i8, .i16, .i32 => .i64,
        .u8, .u16, .u32 => .u64,
        else => arr.dtype,
    };
    return executeCumulative(arr, .prod, optAxis(options), out_dtype);
}

/// Cumulative minimum of elements along a given axis.
pub fn cummin(arr: Array, options: CumMinMaxOptions) (ShapeError || DTypeError || std.mem.Allocator.Error)!Array {
    return executeCumulative(arr, .min, optAxis(options), arr.dtype);
}

/// Cumulative maximum of elements along a given axis.
pub fn cummax(arr: Array, options: CumMinMaxOptions) (ShapeError || DTypeError || std.mem.Allocator.Error)!Array {
    return executeCumulative(arr, .max, optAxis(options), arr.dtype);
}

/// Counts the number of non-zero elements in the array.
pub fn countNonzero(arr: Array, options: ArgOptions) (ShapeError || DTypeError || std.mem.Allocator.Error)!Array {
    var fnz_bool = try empty(arr.allocator, .{ .shape = arr.shapeSlice(), .dtype = .i64 });
    defer fnz_bool.deinit();

    var it = NdIterator.init(arr.shape(), arr.strides());
    var out_it = NdIterator.init(fnz_bool.shape(), fnz_bool.strides());
    while (it.next()) |item| {
        const out_item = out_it.next().?;
        const v = arr.getAsFloat(item.indices[0..arr.ndim]) catch 0.0;
        fnz_bool.set(i64, out_item.indices[0..fnz_bool.ndim], if (v != 0.0) 1 else 0) catch unreachable;
    }

    return sum(fnz_bool, .{ .axis = optAxis(options), .keepDims = optKeepDims(options), .dtype = .i64 });
}

/// Calculate the n-th discrete difference along the given axis.
/// Optional inline config: `.{ .n = 1, .axis = null }`; may be omitted.
pub fn diff(arr: Array, options: DiffOptions) (ShapeError || DTypeError || std.mem.Allocator.Error)!Array {
    const n: usize = options.n;
    const axis_opt: ?isize = options.axis;
    if (n == 0) return arr.clone();
    const s = arr.shape();
    const ax = if (axis_opt) |a| try s.normalizeAxis(a) else s.ndim - 1;
    if (s.dims[ax] <= n) return ShapeError.InvalidDimension;

    var cur = try arr.clone();
    errdefer cur.deinit();

    for (0..n) |_| {
        const cur_s = cur.shape();
        const dim_len = cur_s.dims[ax];

        var out_dims: [MAX_RANK]usize = undefined;
        for (0..cur_s.ndim) |i| {
            out_dims[i] = if (i == ax) dim_len - 1 else cur_s.dims[i];
        }
        const out_shape = Shape{ .dims = out_dims, .ndim = cur_s.ndim };

        var next_arr = try empty(arr.allocator, .{ .shape = out_shape.slice(), .dtype = cur.dtype });
        errdefer next_arr.deinit();

        var out_it = NdIterator.init(out_shape, next_arr.strides());
        while (out_it.next()) |item| {
            var idx0: [MAX_RANK]usize = undefined;
            var idx1: [MAX_RANK]usize = undefined;
            @memcpy(idx0[0..cur_s.ndim], item.indices[0..cur_s.ndim]);
            @memcpy(idx1[0..cur_s.ndim], item.indices[0..cur_s.ndim]);
            idx1[ax] += 1;

            const v0 = cur.getAsFloat(idx0[0..cur_s.ndim]) catch 0.0;
            const v1 = cur.getAsFloat(idx1[0..cur_s.ndim]) catch 0.0;
            next_arr.setFromFloat(item.indices[0..cur_s.ndim], v1 - v0) catch unreachable;
        }

        cur.deinit();
        cur = next_arr;
    }

    return cur;
}

fn executeReduction(
    arr: Array,
    comptime op: ReduceOp,
    axis_opt: ?isize,
    keepDims: bool,
    out_dtype: DType,
) (ShapeError || DTypeError || std.mem.Allocator.Error)!Array {
    const s = arr.shape();

    if (axis_opt == null) {
        // Global reduction
        const out_shape = if (keepDims) blk: {
            const kd_dims = [_]usize{1} ** MAX_RANK;
            break :blk Shape{ .dims = kd_dims, .ndim = s.ndim };
        } else Shape.scalar();

        const out = try empty(arr.allocator, .{
            .shape = out_shape.slice(),
            .dtype = out_dtype,
        });

        inline for (std.meta.fields(DType)) |field| {
            const tag: DType = @enumFromInt(field.value);
            if (out_dtype == tag) {
                const T = tag.toType();
                const res = reduceAll(T, op, arr);
                const ptr: [*]T = @ptrCast(@alignCast(out.data_ptr));
                ptr[0] = res;
                return out;
            }
        }
        return out;
    }

    // Per-axis reduction
    const axis = try s.normalizeAxis(axis_opt.?);

    var out_shape = Shape{ .ndim = if (keepDims) s.ndim else s.ndim - 1 };
    var out_idx: usize = 0;
    for (0..s.ndim) |i| {
        if (i == axis) {
            if (keepDims) {
                out_shape.dims[out_idx] = 1;
                out_idx += 1;
            }
        } else {
            out_shape.dims[out_idx] = s.dims[i];
            out_idx += 1;
        }
    }

    const out = try empty(arr.allocator, .{
        .shape = out_shape.slice(),
        .dtype = out_dtype,
    });

    inline for (std.meta.fields(DType)) |field| {
        const tag: DType = @enumFromInt(field.value);
        if (out_dtype == tag) {
            const T = tag.toType();
            reduceAxis(T, op, arr, axis, out, keepDims);
            return out;
        }
    }

    return out;
}

fn reduceAll(comptime T: type, comptime op: ReduceOp, arr: Array) T {
    var acc: T = initialValue(T, op);
    const total = arr.elementCount();
    if (total == 0) return acc;

    // SIMD fast path when contiguous
    if (arr.isContiguous() and arr.dtype == DType.fromType(T) and (T != bool)) {
        const slice = arr.asConstSlice(T) catch unreachable;
        var i: usize = 0;
        if ((@typeInfo(T) == .float or @typeInfo(T) == .int) and total >= VEC_SIZE) {
            var vec_acc: @Vector(VEC_SIZE, T) = @splat(acc);
            while (i + VEC_SIZE <= total) : (i += VEC_SIZE) {
                const v: @Vector(VEC_SIZE, T) = slice[i..][0..VEC_SIZE].*;
                vec_acc = switch (op) {
                    .sum => vec_acc + v,
                    .prod => vec_acc * v,
                    .min => @min(vec_acc, v),
                    .max => @max(vec_acc, v),
                    else => unreachable,
                };
            }
            acc = switch (op) {
                .sum => @reduce(.Add, vec_acc),
                .prod => @reduce(.Mul, vec_acc),
                .min => @reduce(.Min, vec_acc),
                .max => @reduce(.Max, vec_acc),
                else => unreachable,
            };
        }
        while (i < total) : (i += 1) {
            acc = combineValues(T, op, acc, slice[i]);
        }
        return acc;
    }

    var it = NdIterator.init(arr.shape(), arr.strides());
    while (it.next()) |item| {
        const val = readAndCast(T, arr, item.offset);
        acc = combineValues(T, op, acc, val);
    }
    return acc;
}

fn reduceAxis(comptime T: type, comptime op: ReduceOp, arr: Array, axis: usize, out: Array, keepdims: bool) void {
    const s = arr.shape();
    const axis_len = s.dims[axis];
    if (axis_len == 0) return;

    var out_it = NdIterator.init(out.shape(), out.strides());
    const out_ptr: [*]T = @ptrCast(@alignCast(out.data_ptr));

    while (out_it.next()) |out_item| {
        var acc: T = initialValue(T, op);

        // Map out indices to input coordinates
        var in_indices: [MAX_RANK]usize = undefined;
        var in_dim: usize = 0;
        for (0..s.ndim) |dim| {
            if (dim == axis) {
                in_indices[dim] = 0;
            } else {
                in_indices[dim] = if (keepdims) out_item.indices[dim] else out_item.indices[in_dim];
                in_dim += 1;
            }
        }

        for (0..axis_len) |k| {
            in_indices[axis] = k;
            const offset = arr.elementOffset(in_indices[0..s.ndim]) catch unreachable;
            const val = readAndCast(T, arr, offset);
            acc = combineValues(T, op, acc, val);
        }

        if (out_item.offset >= 0) {
            out_ptr[@as(usize, @intCast(out_item.offset))] = acc;
        }
    }
}

const StatOp = enum { median, variance };

fn executeStatReduction(
    arr: Array,
    comptime op: StatOp,
    axis_opt: ?isize,
    keepDims: bool,
    out_dtype: DType,
) (ShapeError || DTypeError || std.mem.Allocator.Error)!Array {
    const s = arr.shape();
    // Determine output shape matching executeReduction conventions.
    var out_shape: Shape = undefined;
    var axis_usize: ?usize = null;
    if (axis_opt) |ax| {
        axis_usize = try s.normalizeAxis(ax);
        out_shape = Shape{ .ndim = if (keepDims) s.ndim else if (s.ndim == 0) 0 else s.ndim - 1 };
        var out_idx: usize = 0;
        for (0..s.ndim) |i| {
            if (i == axis_usize.?) {
                if (keepDims) {
                    out_shape.dims[out_idx] = 1;
                    out_idx += 1;
                }
            } else {
                out_shape.dims[out_idx] = s.dims[i];
                out_idx += 1;
            }
        }
    } else {
        out_shape = if (keepDims) blk: {
            const kd_dims = [_]usize{1} ** MAX_RANK;
            break :blk Shape{ .dims = kd_dims, .ndim = s.ndim };
        } else Shape.scalar();
    }

    var out = try empty(arr.allocator, .{ .shape = out_shape.slice(), .dtype = out_dtype });
    errdefer out.deinit();

    if (axis_usize == null) {
        const stat = try statGlobal(arr, op);
        if (out_dtype == .f32) {
            (try out.asSlice(f32))[0] = @floatCast(stat);
        } else {
            (try out.asSlice(f64))[0] = stat;
        }
        return out;
    }

    const axis = axis_usize.?;
    const axis_len = s.dims[axis];
    if (axis_len == 0) return ShapeError.EmptyArray;
    const buf = try arr.allocator.alloc(f64, axis_len);
    defer arr.allocator.free(buf);

    var out_it = NdIterator.init(out.shape(), out.strides());
    while (out_it.next()) |out_item| {
        var in_indices: [MAX_RANK]usize = undefined;
        var in_dim: usize = 0;
        for (0..s.ndim) |dim| {
            if (dim == axis) {
                in_indices[dim] = 0;
            } else {
                in_indices[dim] = if (keepDims) out_item.indices[dim] else out_item.indices[in_dim];
                in_dim += 1;
            }
        }
        for (0..axis_len) |k| {
            in_indices[axis] = k;
            buf[k] = arr.getAsFloat(in_indices[0..s.ndim]) catch 0.0;
        }
        const stat = statOfSlice(buf, op);
        if (out_dtype == .f32) {
            const ptr: [*]f32 = @ptrCast(@alignCast(out.data_ptr));
            ptr[@as(usize, @intCast(out_item.offset))] = @floatCast(stat);
        } else {
            const ptr: [*]f64 = @ptrCast(@alignCast(out.data_ptr));
            ptr[@as(usize, @intCast(out_item.offset))] = stat;
        }
    }
    return out;
}

fn statGlobal(arr: Array, comptime op: StatOp) (ShapeError || std.mem.Allocator.Error)!f64 {
    const n = arr.elementCount();
    const buf = try arr.allocator.alloc(f64, n);
    defer arr.allocator.free(buf);
    var it = NdIterator.init(arr.shape(), arr.strides());
    var i: usize = 0;
    while (it.next()) |item| {
        buf[i] = arr.getAsFloat(item.indices[0..arr.ndim]) catch 0.0;
        i += 1;
    }
    return statOfSlice(buf, op);
}

fn statOfSlice(values: []f64, comptime op: StatOp) f64 {
    switch (op) {
        .median => {
            std.mem.sort(f64, values, {}, std.sort.asc(f64));
            const n = values.len;
            if (n % 2 == 1) return values[n / 2];
            return (values[n / 2 - 1] + values[n / 2]) / 2.0;
        },
        .variance => {
            var mean_val: f64 = 0;
            for (values) |v| mean_val += v;
            mean_val /= @as(f64, @floatFromInt(values.len));
            var acc: f64 = 0;
            for (values) |v| {
                const d = v - mean_val;
                acc += d * d;
            }
            return acc / @as(f64, @floatFromInt(values.len));
        },
    }
}

fn executeArgReduction(
    arr: Array,
    is_max: bool,
    axis_opt: ?isize,
    keepDims: bool,
) (ShapeError || DTypeError || std.mem.Allocator.Error)!Array {
    if (arr.elementCount() == 0) return ShapeError.EmptyArray;
    const s = arr.shape();

    if (axis_opt == null) {
        // Global argmin / argmax (flat index)
        const out_shape = if (keepDims) blk: {
            const kd_dims = [_]usize{1} ** MAX_RANK;
            break :blk Shape{ .dims = kd_dims, .ndim = s.ndim };
        } else Shape.scalar();

        const out = try empty(arr.allocator, .{
            .shape = out_shape.slice(),
            .dtype = .i64,
        });

        var it = NdIterator.init(s, arr.strides());
        var best_idx: usize = 0;
        var flat_idx: usize = 0;
        var best_val: f64 = if (is_max) -std.math.inf(f64) else std.math.inf(f64);

        while (it.next()) |item| : (flat_idx += 1) {
            const v = readAndCast(f64, arr, item.offset);
            const improves = if (is_max) v > best_val else v < best_val;
            if (improves or flat_idx == 0) {
                best_val = v;
                best_idx = flat_idx;
            }
        }

        const out_ptr: [*]i64 = @ptrCast(@alignCast(out.data_ptr));
        out_ptr[0] = @intCast(best_idx);
        return out;
    }

    const axis = try s.normalizeAxis(axis_opt.?);
    const axis_len = s.dims[axis];

    var out_shape = Shape{ .ndim = if (keepDims) s.ndim else s.ndim - 1 };
    var out_idx: usize = 0;
    for (0..s.ndim) |i| {
        if (i == axis) {
            if (keepDims) {
                out_shape.dims[out_idx] = 1;
                out_idx += 1;
            }
        } else {
            out_shape.dims[out_idx] = s.dims[i];
            out_idx += 1;
        }
    }

    var out = try empty(arr.allocator, .{
        .shape = out_shape.slice(),
        .dtype = .i64,
    });

    var out_it = NdIterator.init(out.shape(), out.strides());
    const out_ptr: [*]i64 = @ptrCast(@alignCast(out.data_ptr));

    while (out_it.next()) |out_item| {
        var in_indices: [MAX_RANK]usize = undefined;
        var in_dim: usize = 0;
        for (0..s.ndim) |dim| {
            if (dim == axis) {
                in_indices[dim] = 0;
            } else {
                in_indices[dim] = if (keepDims) out_item.indices[dim] else out_item.indices[in_dim];
                in_dim += 1;
            }
        }

        var best_idx: usize = 0;
        var best_val: f64 = if (is_max) -std.math.inf(f64) else std.math.inf(f64);

        for (0..axis_len) |k| {
            in_indices[axis] = k;
            const offset = arr.elementOffset(in_indices[0..s.ndim]) catch unreachable;
            const v = readAndCast(f64, arr, offset);
            const improves = if (is_max) v > best_val else v < best_val;
            if (improves or k == 0) {
                best_val = v;
                best_idx = k;
            }
        }

        if (out_item.offset >= 0) {
            out_ptr[@as(usize, @intCast(out_item.offset))] = @intCast(best_idx);
        }
    }

    return out;
}

fn executeCumulative(
    arr: Array,
    comptime op: enum { sum, prod, min, max },
    axis_opt: ?isize,
    out_dtype: DType,
) (ShapeError || DTypeError || std.mem.Allocator.Error)!Array {
    const s = arr.shape();
    if (s.elementCount() == 0) return arr.clone();

    const target_axis = if (axis_opt) |ax| try s.normalizeAxis(ax) else 0;
    const axis_len = s.dims[target_axis];

    var out = try empty(arr.allocator, .{
        .shape = s.slice(),
        .dtype = out_dtype,
    });

    inline for (std.meta.fields(DType)) |field| {
        const tag: DType = @enumFromInt(field.value);
        if (out_dtype == tag) {
            const T = tag.toType();
            const out_ptr: [*]T = @ptrCast(@alignCast(out.data_ptr));

            var out_it = NdIterator.init(s, out.strides());
            while (out_it.next()) |out_item| {
                if (out_item.indices[target_axis] == 0) {
                    var in_indices: [MAX_RANK]usize = undefined;
                    @memcpy(in_indices[0..s.ndim], out_item.indices[0..s.ndim]);
                    const off = arr.elementOffset(in_indices[0..s.ndim]) catch unreachable;
                    var acc: T = readAndCast(T, arr, off);
                    out_ptr[@as(usize, @intCast(out_item.offset))] = acc;

                    for (1..axis_len) |k| {
                        in_indices[target_axis] = k;
                        const next_in_off = arr.elementOffset(in_indices[0..s.ndim]) catch unreachable;
                        const next_out_off = out.elementOffset(in_indices[0..s.ndim]) catch unreachable;
                        const next_val = readAndCast(T, arr, next_in_off);
                        if (T == bool) {
                            acc = switch (op) {
                                .sum => acc or next_val,
                                .prod => acc and next_val,
                                .min => acc and next_val,
                                .max => acc or next_val,
                            };
                        } else if (@typeInfo(T) == .@"struct") {
                            acc = switch (op) {
                                .sum => acc.add(next_val),
                                .prod => acc.mul(next_val),
                                else => acc,
                            };
                        } else {
                            acc = switch (op) {
                                .sum => acc + next_val,
                                .prod => acc * next_val,
                                .min => if (next_val < acc) next_val else acc,
                                .max => if (next_val > acc) next_val else acc,
                            };
                        }
                        out_ptr[@as(usize, @intCast(next_out_off))] = acc;
                    }
                }
            }

            return out;
        }
    }

    return out;
}

inline fn initialValue(comptime T: type, comptime op: ReduceOp) T {
    if (T == bool) {
        return switch (op) {
            .sum => false,
            .prod => true,
            .min => true,
            .max => false,
            .all => true,
            .any => false,
        };
    }
    if (@typeInfo(T) == .@"struct") {
        return switch (op) {
            .sum => T.init(0.0, 0.0),
            .prod => T.init(1.0, 0.0),
            else => T.init(0.0, 0.0),
        };
    }
    return switch (op) {
        .sum => 0,
        .prod => 1,
        .all => 1,
        .any => 0,
        .min => switch (@typeInfo(T)) {
            .float => std.math.inf(T),
            .int => std.math.maxInt(T),
            else => 0,
        },
        .max => switch (@typeInfo(T)) {
            .float => -std.math.inf(T),
            .int => std.math.minInt(T),
            else => 0,
        },
    };
}

inline fn combineValues(comptime T: type, comptime op: ReduceOp, a: T, b: T) T {
    if (T == bool) {
        return switch (op) {
            .sum, .any => a or b,
            .prod, .all => a and b,
            .min => a and b,
            .max => a or b,
        };
    }
    if (@typeInfo(T) == .@"struct") {
        return switch (op) {
            .sum => a.add(b),
            .prod => a.mul(b),
            else => a,
        };
    }
    return switch (op) {
        .sum => a + b,
        .prod => a * b,
        .min => @min(a, b),
        .max => @max(a, b),
        .all => if (a != 0 and b != 0) 1 else 0,
        .any => if (a != 0 or b != 0) 1 else 0,
    };
}

fn readAndCast(comptime TargetT: type, arr: Array, offset: isize) TargetT {
    inline for (std.meta.fields(DType)) |field| {
        const tag: DType = @enumFromInt(field.value);
        if (arr.dtype == tag) {
            const SrcT = tag.toType();
            const ptr: [*]const SrcT = @ptrCast(@alignCast(arr.data_ptr));
            const elem = if (offset >= 0) ptr[@as(usize, @intCast(offset))] else ptr[0];
            return DType.castValue(TargetT, SrcT, elem);
        }
    }
    return if (TargetT == bool) false else if (@typeInfo(TargetT) == .@"struct") TargetT.init(0.0, 0.0) else 0;
}

test "global and axis reductions" {
    const allocator = std.testing.allocator;
    const fromSlice = @import("../core/array.zig").fromSlice;

    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    var a = try fromSlice(allocator, f64, .{ .data = &data, .shape = &.{ 2, 3 } });
    defer a.deinit();

    // Global sum: 21.0
    var s_all = try sum(a, .{});
    defer s_all.deinit();
    try std.testing.expectEqual(@as(f64, 21.0), try s_all.get(f64, &.{}));

    // Sum along axis 0: [5, 7, 9]
    var s_ax0 = try sum(a, .{ .axis = 0 });
    defer s_ax0.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 5.0, 7.0, 9.0 }, try s_ax0.asSlice(f64));

    // Sum along axis 1: [6, 15]
    var s_ax1 = try sum(a, .{ .axis = 1 });
    defer s_ax1.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 6.0, 15.0 }, try s_ax1.asSlice(f64));

    // Mean along axis 1: [2.0, 5.0]
    var m_ax1 = try mean(a, .{ .axis = 1 });
    defer m_ax1.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 5.0 }, try m_ax1.asSlice(f64));

    // Argmax along axis 1: [2, 2]
    var am_ax1 = try argmax(a, .{ .axis = 1 });
    defer am_ax1.deinit();
    try std.testing.expectEqualSlices(i64, &.{ 2, 2 }, try am_ax1.asSlice(i64));

    // Cumsum along axis 1
    var cs = try cumsum(a, .{ .axis = 1 });
    defer cs.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 3.0, 6.0, 4.0, 9.0, 15.0 }, try cs.asSlice(f64));

    // Cummin and cummax
    var c_min = try cummin(a, .{ .axis = 1 });
    defer c_min.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 1.0, 1.0, 4.0, 4.0, 4.0 }, try c_min.asSlice(f64));

    // CountNonzero
    const mask_data = [_]f64{ 0.0, 2.0, 0.0, 4.0, 5.0, 0.0 };
    var m = try fromSlice(allocator, f64, .{ .data = &mask_data, .shape = &.{ 2, 3 } });
    defer m.deinit();
    var c_nz = try countNonzero(m, .{});
    defer c_nz.deinit();
    try std.testing.expectEqual(@as(i64, 3), try c_nz.get(i64, &.{}));

    // Diff
    const diff_data = [_]f64{ 1.0, 2.0, 4.0, 7.0, 11.0 };
    var d_arr = try fromSlice(allocator, f64, .{ .data = &diff_data, .shape = &.{5} });
    defer d_arr.deinit();
    var d_res = try diff(d_arr, .{});
    defer d_res.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1.0, 2.0, 3.0, 4.0 }, try d_res.asSlice(f64));
}

test "median variance stdDev global and axis" {
    const allocator = std.testing.allocator;
    const fromSlice = @import("../core/array.zig").fromSlice;
    const vals = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    var a = try fromSlice(allocator, f64, .{ .data = &vals, .shape = &.{4} });
    defer a.deinit();
    var med = try median(a, .{});
    defer med.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 2.5), try med.get(f64, &.{}), 1e-12);
    var vari = try variance(a, .{});
    defer vari.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 1.25), try vari.get(f64, &.{}), 1e-12);
    var sd = try stdDev(a, .{});
    defer sd.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 1.1180339887), try sd.get(f64, &.{}), 1e-9);

    // f32 preserves dtype
    const f32vals = [_]f32{ 1.0, 3.0 };
    var b = try fromSlice(allocator, f32, .{ .data = &f32vals, .shape = &.{2} });
    defer b.deinit();
    var med32 = try median(b, .{});
    defer med32.deinit();
    try std.testing.expect(med32.dtype == .f32);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), try med32.get(f32, &.{}), 1e-5);

    // axis reduction with keepDims and negative axis
    const m2 = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    var c = try fromSlice(allocator, f64, .{ .data = &m2, .shape = &.{ 2, 3 } });
    defer c.deinit();
    var med_ax = try median(c, .{ .axis = 1 });
    defer med_ax.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 5.0 }, try med_ax.asSlice(f64));
    var med_neg = try median(c, .{ .axis = -1 });
    defer med_neg.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 2.0, 5.0 }, try med_neg.asSlice(f64));
    var var_keep = try variance(c, .{ .axis = 0, .keepDims = true });
    defer var_keep.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 3 }, var_keep.shapeSlice());
}
