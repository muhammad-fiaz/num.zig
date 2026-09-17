//! Array sorting and argsort algorithms.
//!
//! Provides axis-aware in-place sorting, sorted copies, and index sorting (argsort).

const std = @import("std");
const Array = @import("../core/array.zig").Array;
const zeros = @import("../core/array.zig").zeros;
const empty = @import("../core/array.zig").empty;
const fromSlice = @import("../core/array.zig").fromSlice;
const DType = @import("../core/dtype.zig").DType;
const Shape = @import("../core/shape.zig").Shape;
const Strides = @import("../core/shape.zig").Strides;
const MAX_RANK = @import("../core/shape.zig").MAX_RANK;
const ShapeError = @import("../core/error.zig").ShapeError;
const DTypeError = @import("../core/error.zig").DTypeError;
const IndexError = @import("../core/error.zig").IndexError;

pub const SortOrder = enum {
    asc,
    desc,
};

pub const SortOptions = struct {
    axis: ?isize = -1,
    order: SortOrder = .asc,
    stable: bool = false,
};

/// In-place sort of an array along the specified axis.
pub fn sort(
    arr: *Array,
    options: SortOptions,
) (ShapeError || DTypeError || IndexError || std.mem.Allocator.Error)!void {
    if (arr.elementCount() <= 1) return;

    // Case 1: Global sort over entire array if axis is null
    if (options.axis == null) {
        if (arr.isContiguous()) {
            inline for (std.meta.fields(DType)) |field| {
                const tag: DType = @enumFromInt(field.value);
                if (arr.dtype == tag) {
                    const T = tag.toType();
                    const slice = arr.asSlice(T) catch unreachable;
                    sortSlice(T, slice, options.order, options.stable);
                    return;
                }
            }
        } else {
            const N = arr.elementCount();
            const temp = try arr.allocator.alloc(f64, N);
            defer arr.allocator.free(temp);

            const ravel = @import("../manip/reshape.zig").ravel;
            var flat = try ravel(arr.*);
            defer flat.deinit();

            for (0..N) |i| {
                temp[i] = try flat.get(f64, &.{i});
            }

            sortSlice(f64, temp, options.order, options.stable);

            for (0..N) |i| {
                try flat.set(f64, &.{i}, temp[i]);
            }
            return;
        }
    }

    // Case 2: Axis-reduced sort
    const ax = options.axis.?;
    const r: isize = @intCast(arr.ndim);
    const resolved_axis: usize = if (ax < 0) @intCast(ax + r) else @intCast(ax);
    if (resolved_axis >= arr.ndim) return ShapeError.AxisOutOfBounds;

    const line_len = arr.shape_dims[resolved_axis];
    if (line_len <= 1) return;

    if (arr.ndim == 1 and arr.isContiguous()) {
        inline for (std.meta.fields(DType)) |field| {
            const tag: DType = @enumFromInt(field.value);
            if (arr.dtype == tag) {
                const T = tag.toType();
                const slice = arr.asSlice(T) catch unreachable;
                sortSlice(T, slice, options.order, options.stable);
                return;
            }
        }
    }

    // General N-D array sorting along axis
    var outer_dims: [MAX_RANK]usize = undefined;
    var outer_ndim: u8 = 0;

    for (0..arr.ndim) |d| {
        if (d != resolved_axis) {
            outer_dims[outer_ndim] = arr.shape_dims[d];
            outer_ndim += 1;
        }
    }

    const outer_shape = Shape{ .dims = outer_dims, .ndim = outer_ndim };
    const NdIterator = @import("../core/iterator.zig").NdIterator;
    var it = NdIterator.init(outer_shape, Strides.fromShape(outer_shape, .c));

    const line_buf = try arr.allocator.alloc(f64, line_len);
    defer arr.allocator.free(line_buf);

    var coords: [MAX_RANK]usize = undefined;

    while (it.next()) |item| {
        var out_d: usize = 0;
        for (0..arr.ndim) |d| {
            if (d == resolved_axis) {
                coords[d] = 0;
            } else {
                coords[d] = item.indices[out_d];
                out_d += 1;
            }
        }

        for (0..line_len) |k| {
            coords[resolved_axis] = k;
            line_buf[k] = try arr.get(f64, coords[0..arr.ndim]);
        }

        sortSlice(f64, line_buf, options.order, options.stable);

        for (0..line_len) |k| {
            coords[resolved_axis] = k;
            try arr.set(f64, coords[0..arr.ndim], line_buf[k]);
        }
    }
}

/// Returns a sorted copy of the array.
pub fn sorted(
    arr: Array,
    options: SortOptions,
) (ShapeError || DTypeError || IndexError || std.mem.Allocator.Error)!Array {
    var out = try arr.clone();
    errdefer out.deinit();

    try sort(&out, options);
    return out;
}

pub const ArgsortOptions = struct {
    axis: isize = -1,
    order: SortOrder = .asc,
    stable: bool = false,
};

/// Returns the indices that would sort an array along the specified axis.
pub fn argsort(
    arr: Array,
    options: ArgsortOptions,
) (ShapeError || DTypeError || IndexError || std.mem.Allocator.Error)!Array {
    if (arr.elementCount() == 0) return ShapeError.EmptyArray;

    const r: isize = @intCast(arr.ndim);
    const resolved_axis: usize = if (options.axis < 0) @intCast(options.axis + r) else @intCast(options.axis);
    if (resolved_axis >= arr.ndim) return ShapeError.AxisOutOfBounds;

    const line_len = arr.shape_dims[resolved_axis];

    var out = try zeros(arr.allocator, .{
        .shape = arr.shapeSlice(),
        .dtype = .i64,
    });
    errdefer out.deinit();

    var outer_dims: [MAX_RANK]usize = undefined;
    var outer_ndim: u8 = 0;

    for (0..arr.ndim) |d| {
        if (d != resolved_axis) {
            outer_dims[outer_ndim] = arr.shape_dims[d];
            outer_ndim += 1;
        }
    }

    const outer_shape = Shape{ .dims = outer_dims, .ndim = outer_ndim };
    const NdIterator = @import("../core/iterator.zig").NdIterator;
    var it = NdIterator.init(outer_shape, Strides.fromShape(outer_shape, .c));

    const val_buf = try arr.allocator.alloc(f64, line_len);
    defer arr.allocator.free(val_buf);

    const idx_buf = try arr.allocator.alloc(usize, line_len);
    defer arr.allocator.free(idx_buf);

    var coords: [MAX_RANK]usize = undefined;

    while (it.next()) |item| {
        var out_d: usize = 0;
        for (0..arr.ndim) |d| {
            if (d == resolved_axis) {
                coords[d] = 0;
            } else {
                coords[d] = item.indices[out_d];
                out_d += 1;
            }
        }

        for (0..line_len) |k| {
            coords[resolved_axis] = k;
            val_buf[k] = try arr.get(f64, coords[0..arr.ndim]);
            idx_buf[k] = k;
        }

        const Context = struct {
            vals: []const f64,
            order: SortOrder,

            pub fn lessThan(ctx: @This(), a: usize, b: usize) bool {
                if (ctx.order == .asc) {
                    return ctx.vals[a] < ctx.vals[b];
                } else {
                    return ctx.vals[a] > ctx.vals[b];
                }
            }
        };

        const ctx = Context{ .vals = val_buf, .order = options.order };

        if (options.stable) {
            std.sort.block(usize, idx_buf, ctx, Context.lessThan);
        } else {
            std.sort.pdq(usize, idx_buf, ctx, Context.lessThan);
        }

        for (0..line_len) |k| {
            coords[resolved_axis] = k;
            try out.set(i64, coords[0..arr.ndim], @intCast(idx_buf[k]));
        }
    }

    return out;
}

fn sortSlice(comptime T: type, slice: []T, order: SortOrder, stable: bool) void {
    const Context = struct {
        order: SortOrder,
        pub fn lessThan(ctx: @This(), a: T, b: T) bool {
            if (T == bool) {
                const a_int: u1 = @intFromBool(a);
                const b_int: u1 = @intFromBool(b);
                if (ctx.order == .asc) {
                    return a_int < b_int;
                } else {
                    return a_int > b_int;
                }
            } else if (@typeInfo(T) == .@"struct") {
                // Lexicographical ordering by real part, then imaginary part
                if (a.re != b.re) {
                    return if (ctx.order == .asc) a.re < b.re else a.re > b.re;
                }
                return if (ctx.order == .asc) a.im < b.im else a.im > b.im;
            } else {
                if (ctx.order == .asc) {
                    return a < b;
                } else {
                    return a > b;
                }
            }
        }
    };
    const ctx = Context{ .order = order };
    if (stable) {
        std.sort.block(T, slice, ctx, Context.lessThan);
    } else {
        std.sort.pdq(T, slice, ctx, Context.lessThan);
    }
}

test "sort and sorted" {
    const allocator = std.testing.allocator;
    const items = [_]f64{ 5.0, 2.0, 9.0, 1.0, 7.0 };
    var arr = try fromSlice(allocator, f64, .{ .data = &items, .shape = &.{5} });
    defer arr.deinit();

    var s_asc = try sorted(arr, .{ .order = .asc });
    defer s_asc.deinit();

    const expected_asc = [_]f64{ 1.0, 2.0, 5.0, 7.0, 9.0 };
    for (0..5) |i| {
        try std.testing.expectEqual(expected_asc[i], try s_asc.get(f64, &.{i}));
    }

    var s_desc = try sorted(arr, .{ .order = .desc });
    defer s_desc.deinit();

    const expected_desc = [_]f64{ 9.0, 7.0, 5.0, 2.0, 1.0 };
    for (0..5) |i| {
        try std.testing.expectEqual(expected_desc[i], try s_desc.get(f64, &.{i}));
    }
}

test "argsort" {
    const allocator = std.testing.allocator;
    const items = [_]f64{ 30.0, 10.0, 20.0 };
    var arr = try fromSlice(allocator, f64, .{ .data = &items, .shape = &.{3} });
    defer arr.deinit();

    var idxs = try argsort(arr, .{});
    defer idxs.deinit();

    // 10.0 is at 1, 20.0 is at 2, 30.0 is at 0 -> [1, 2, 0]
    try std.testing.expectEqual(@as(i64, 1), try idxs.get(i64, &.{0}));
    try std.testing.expectEqual(@as(i64, 2), try idxs.get(i64, &.{1}));
    try std.testing.expectEqual(@as(i64, 0), try idxs.get(i64, &.{2}));
}
