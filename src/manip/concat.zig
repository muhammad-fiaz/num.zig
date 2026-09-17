//! Array concatenation, stacking, splitting, tiling, and repeating.
//!
//! Provides joining and splitting primitives across arbitrary dimensions with
//! promotion of mixed data types.

const std = @import("std");
const Array = @import("../core/array.zig").Array;
const zeros = @import("../core/array.zig").zeros;
const empty = @import("../core/array.zig").empty;
const Shape = @import("../core/shape.zig").Shape;
const Strides = @import("../core/shape.zig").Strides;
const DType = @import("../core/dtype.zig").DType;
const MAX_RANK = @import("../core/shape.zig").MAX_RANK;
const ShapeError = @import("../core/error.zig").ShapeError;
const DTypeError = @import("../core/error.zig").DTypeError;
const NdIterator = @import("../core/iterator.zig").NdIterator;

/// Joins a sequence of arrays along an existing axis.
pub fn concat(
    arrays: []const Array,
    options: struct {
        axis: isize = 0,
    },
) (ShapeError || DTypeError || std.mem.Allocator.Error)!Array {
    if (arrays.len == 0) return ShapeError.EmptyArray;
    const first = arrays[0];
    const s0 = first.shape();
    const axis = try s0.normalizeAxis(options.axis);

    var promoted_dtype = first.dtype;
    var total_axis_len: usize = 0;

    for (arrays) |arr| {
        if (arr.ndim != first.ndim) return ShapeError.IncompatibleShapes;
        for (0..first.ndim) |i| {
            if (i == axis) {
                total_axis_len += arr.shape_dims[i];
            } else if (arr.shape_dims[i] != first.shape_dims[i]) {
                return ShapeError.IncompatibleShapes;
            }
        }
        promoted_dtype = DType.promote(promoted_dtype, arr.dtype);
    }

    var out_dims: [MAX_RANK]usize = undefined;
    for (0..first.ndim) |i| {
        out_dims[i] = if (i == axis) total_axis_len else first.shape_dims[i];
    }

    const out_shape = Shape{ .dims = out_dims, .ndim = first.ndim };
    var out = try empty(first.allocator, .{
        .shape = out_shape.slice(),
        .dtype = promoted_dtype,
    });

    inline for (std.meta.fields(DType)) |field| {
        const tag: DType = @enumFromInt(field.value);
        if (promoted_dtype == tag) {
            const T = tag.toType();
            var axis_offset: usize = 0;

            for (arrays) |arr| {
                var it = NdIterator.init(arr.shape(), arr.strides());
                while (it.next()) |item| {
                    var out_indices: [MAX_RANK]usize = undefined;
                    @memcpy(out_indices[0..first.ndim], item.indices[0..first.ndim]);
                    out_indices[axis] += axis_offset;

                    const val = readAndCast(T, arr, item.offset);
                    out.set(T, out_indices[0..first.ndim], val) catch unreachable;
                }
                axis_offset += arr.shape_dims[axis];
            }

            return out;
        }
    }

    return out;
}

/// Joins a sequence of arrays along a new axis.
pub fn stack(
    arrays: []const Array,
    options: struct {
        axis: isize = 0,
    },
) (ShapeError || DTypeError || std.mem.Allocator.Error)!Array {
    if (arrays.len == 0) return ShapeError.EmptyArray;
    const first = arrays[0];
    const expandDims = @import("reshape.zig").expandDims;

    const new_rank = first.ndim + 1;
    if (new_rank > MAX_RANK) return ShapeError.RankExceeded;

    var expanded = try first.allocator.alloc(Array, arrays.len);
    defer first.allocator.free(expanded);

    for (arrays, 0..) |arr, i| {
        if (!arr.shape().equal(first.shape())) return ShapeError.IncompatibleShapes;
        expanded[i] = try expandDims(arr, .{ .axis = options.axis });
    }

    return concat(expanded, .{ .axis = options.axis });
}

/// Splits an array into multiple sub-arrays along an axis.
pub fn split(
    allocator: std.mem.Allocator,
    arr: Array,
    options: struct {
        parts: usize,
        axis: isize = 0,
    },
) (ShapeError || std.mem.Allocator.Error)![]Array {
    if (options.parts == 0) return ShapeError.InvalidDimension;
    const s = arr.shape();
    const axis = try s.normalizeAxis(options.axis);
    const axis_len = s.dims[axis];

    if (axis_len % options.parts != 0) return ShapeError.InvalidDimension;
    const part_size = axis_len / options.parts;

    const slice_fn = @import("slice.zig").slice;
    const Slice = @import("../core/shape.zig").Slice;

    const result = try allocator.alloc(Array, options.parts);
    errdefer allocator.free(result);

    var current_start: usize = 0;
    for (0..options.parts) |p| {
        var slices: [MAX_RANK]Slice = undefined;
        for (0..s.ndim) |i| {
            if (i == axis) {
                slices[i] = Slice{
                    .start = @intCast(current_start),
                    .stop = @intCast(current_start + part_size),
                    .step = 1,
                };
            } else {
                slices[i] = Slice{ .start = null, .stop = null, .step = 1 };
            }
        }
        result[p] = slice_fn(arr, slices[0..s.ndim]) catch unreachable;
        current_start += part_size;
    }

    return result;
}

/// Constructs an array by repeating `arr` the number of times given by `reps`.
pub fn tile(
    arr: Array,
    options: struct {
        reps: []const usize,
    },
) (ShapeError || DTypeError || std.mem.Allocator.Error)!Array {
    if (options.reps.len > MAX_RANK) return ShapeError.RankExceeded;

    const s = arr.shape();
    const max_ndim = @max(s.ndim, @as(u8, @intCast(options.reps.len)));
    if (max_ndim > MAX_RANK) return ShapeError.RankExceeded;

    var in_dims = [_]usize{1} ** MAX_RANK;
    const lead = max_ndim - s.ndim;
    for (0..s.ndim) |i| {
        in_dims[lead + i] = s.dims[i];
    }

    var rep_dims = [_]usize{1} ** MAX_RANK;
    const rep_lead = max_ndim - @as(u8, @intCast(options.reps.len));
    for (0..options.reps.len) |i| {
        rep_dims[rep_lead + i] = options.reps[i];
    }

    var out_dims: [MAX_RANK]usize = undefined;
    for (0..max_ndim) |i| {
        out_dims[i] = in_dims[i] * rep_dims[i];
    }

    const out_shape = Shape{ .dims = out_dims, .ndim = max_ndim };
    var out = try empty(arr.allocator, .{
        .shape = out_shape.slice(),
        .dtype = arr.dtype,
    });

    inline for (std.meta.fields(DType)) |field| {
        const tag: DType = @enumFromInt(field.value);
        if (arr.dtype == tag) {
            const T = tag.toType();

            var out_it = NdIterator.init(out_shape, out.strides());
            while (out_it.next()) |out_item| {
                var src_indices: [MAX_RANK]usize = undefined;
                for (0..s.ndim) |dim| {
                    src_indices[dim] = out_item.indices[lead + dim] % s.dims[dim];
                }
                const offset = arr.elementOffset(src_indices[0..s.ndim]) catch unreachable;
                const val = readAndCast(T, arr, offset);
                out.set(T, out_item.indices[0..max_ndim], val) catch unreachable;
            }

            return out;
        }
    }

    return out;
}

/// Repeats elements of an array.
pub fn repeat(
    arr: Array,
    options: struct {
        repeats: usize,
        axis: ?isize = null,
    },
) (ShapeError || DTypeError || std.mem.Allocator.Error)!Array {
    if (options.repeats == 0) return ShapeError.InvalidDimension;

    if (options.axis == null) {
        // Flatten and repeat elements
        const ravel = @import("reshape.zig").ravel;
        var flat = try ravel(arr);
        defer flat.deinit();

        const count = flat.elementCount() * options.repeats;
        var out = try empty(arr.allocator, .{
            .shape = &.{count},
            .dtype = arr.dtype,
        });

        inline for (std.meta.fields(DType)) |field| {
            const tag: DType = @enumFromInt(field.value);
            if (arr.dtype == tag) {
                const T = tag.toType();
                var out_idx: usize = 0;
                for (0..flat.elementCount()) |i| {
                    const val = flat.get(T, &.{i}) catch unreachable;
                    for (0..options.repeats) |_| {
                        out.set(T, &.{out_idx}, val) catch unreachable;
                        out_idx += 1;
                    }
                }
                return out;
            }
        }
        return out;
    }

    const s = arr.shape();
    const axis = try s.normalizeAxis(options.axis.?);

    var out_dims: [MAX_RANK]usize = undefined;
    for (0..s.ndim) |i| {
        out_dims[i] = if (i == axis) s.dims[i] * options.repeats else s.dims[i];
    }

    const out_shape = Shape{ .dims = out_dims, .ndim = s.ndim };
    var out = try empty(arr.allocator, .{
        .shape = out_shape.slice(),
        .dtype = arr.dtype,
    });

    inline for (std.meta.fields(DType)) |field| {
        const tag: DType = @enumFromInt(field.value);
        if (arr.dtype == tag) {
            const T = tag.toType();

            var out_it = NdIterator.init(out_shape, out.strides());
            while (out_it.next()) |out_item| {
                var src_indices: [MAX_RANK]usize = undefined;
                for (0..s.ndim) |dim| {
                    src_indices[dim] = if (dim == axis)
                        out_item.indices[dim] / options.repeats
                    else
                        out_item.indices[dim];
                }
                const offset = arr.elementOffset(src_indices[0..s.ndim]) catch unreachable;
                const val = readAndCast(T, arr, offset);
                out.set(T, out_item.indices[0..s.ndim], val) catch unreachable;
            }

            return out;
        }
    }

    return out;
}

fn readAndCast(comptime TargetT: type, arr: Array, offset: isize) TargetT {
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

test "concat, stack, and split" {
    const allocator = std.testing.allocator;
    const fromSlice = @import("../core/array.zig").fromSlice;

    const data1 = [_]f32{ 1, 2, 3 };
    var a = try fromSlice(allocator, f32, .{ .data = &data1, .shape = &.{ 1, 3 } });
    defer a.deinit();

    const data2 = [_]f32{ 4, 5, 6 };
    var b = try fromSlice(allocator, f32, .{ .data = &data2, .shape = &.{ 1, 3 } });
    defer b.deinit();

    // Concat along axis 0 -> [2, 3]
    var c0 = try concat(&.{ a, b }, .{ .axis = 0 });
    defer c0.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, c0.shapeSlice());
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4, 5, 6 }, try c0.asSlice(f32));

    // Concat along axis 1 -> [1, 6]
    var c1 = try concat(&.{ a, b }, .{ .axis = 1 });
    defer c1.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 6 }, c1.shapeSlice());
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4, 5, 6 }, try c1.asSlice(f32));

    // Stack along axis 0 -> [2, 1, 3]
    var st = try stack(&.{ a, b }, .{ .axis = 0 });
    defer st.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 1, 3 }, st.shapeSlice());

    // Split c0 into 2 parts along axis 0
    const parts = try split(allocator, c0, .{ .parts = 2, .axis = 0 });
    defer allocator.free(parts);
    defer for (parts) |*p| p.deinit();

    try std.testing.expectEqual(@as(usize, 2), parts.len);
    try std.testing.expectEqualSlices(usize, &.{ 1, 3 }, parts[0].shapeSlice());
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3 }, try parts[0].asSlice(f32));
    try std.testing.expectEqualSlices(f32, &.{ 4, 5, 6 }, try parts[1].asSlice(f32));
}

test "tile and repeat" {
    const allocator = std.testing.allocator;
    const fromSlice = @import("../core/array.zig").fromSlice;

    const data = [_]f64{ 1, 2 };
    var a = try fromSlice(allocator, f64, .{ .data = &data, .shape = &.{2} });
    defer a.deinit();

    // Tile (2,) with reps (2,) -> (4,)
    var tl = try tile(a, .{ .reps = &.{2} });
    defer tl.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 2, 1, 2 }, try tl.asSlice(f64));

    // Repeat elements 3 times -> [1, 1, 1, 2, 2, 2]
    var rep = try repeat(a, .{ .repeats = 3 });
    defer rep.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 1, 1, 2, 2, 2 }, try rep.asSlice(f64));
}
