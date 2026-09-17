//! Transposition and axis permutation operations.
//!
//! Provides zero-allocation views for matrix transpose, arbitrary dimension permutations,
//! and axis swapping by reordering shape dimensions and stride values.

const std = @import("std");
const Array = @import("../core/array.zig").Array;
const Shape = @import("../core/shape.zig").Shape;
const Strides = @import("../core/shape.zig").Strides;
const MAX_RANK = @import("../core/shape.zig").MAX_RANK;
const ShapeError = @import("../core/error.zig").ShapeError;

/// Reverse or permute the axes of an array. Returns a zero-allocation view.
pub fn transpose(
    arr: Array,
    options: struct {
        axes: ?[]const usize = null,
    },
) ShapeError!Array {
    const s = arr.shape();
    if (s.ndim <= 1 and options.axes == null) return arr.view();

    var new_dims: [MAX_RANK]usize = undefined;
    var new_strides: [MAX_RANK]isize = undefined;

    if (options.axes) |perm| {
        if (perm.len != s.ndim) return ShapeError.AxisOutOfBounds;
        var seen = [_]bool{false} ** MAX_RANK;

        for (perm, 0..) |ax, i| {
            if (ax >= s.ndim or seen[ax]) return ShapeError.AxisOutOfBounds;
            seen[ax] = true;
            new_dims[i] = arr.shape_dims[ax];
            new_strides[i] = arr.stride_vals[ax];
        }
    } else {
        // Reverse all axes
        for (0..s.ndim) |i| {
            const rev_idx = s.ndim - 1 - i;
            new_dims[i] = arr.shape_dims[rev_idx];
            new_strides[i] = arr.stride_vals[rev_idx];
        }
    }

    var v = arr.view();
    for (0..s.ndim) |i| {
        v.shape_dims[i] = new_dims[i];
        v.stride_vals[i] = new_strides[i];
    }

    const new_shape = v.shape();
    const new_st = v.strides();
    v.flags.isCContiguous = Strides.isCContiguous(new_shape, new_st);
    v.flags.isFContiguous = Strides.isFContiguous(new_shape, new_st);

    return v;
}

/// Interchange two axes of an array. Returns a zero-allocation view.
pub fn swapAxes(arr: Array, axis1: isize, axis2: isize) ShapeError!Array {
    const s = arr.shape();
    const ax1 = try s.normalizeAxis(axis1);
    const ax2 = try s.normalizeAxis(axis2);

    if (ax1 == ax2) return arr.view();

    var v = arr.view();
    std.mem.swap(usize, &v.shape_dims[ax1], &v.shape_dims[ax2]);
    std.mem.swap(isize, &v.stride_vals[ax1], &v.stride_vals[ax2]);

    const new_shape = v.shape();
    const new_st = v.strides();
    v.flags.isCContiguous = Strides.isCContiguous(new_shape, new_st);
    v.flags.isFContiguous = Strides.isFContiguous(new_shape, new_st);

    return v;
}

/// Move axes of an array to new positions. Returns a zero-allocation view.
pub fn moveAxis(arr: Array, source: isize, destination: isize) ShapeError!Array {
    const s = arr.shape();
    const src = try s.normalizeAxis(source);
    const dst = try s.normalizeAxis(destination);

    if (src == dst) return arr.view();

    var perm: [MAX_RANK]usize = undefined;
    var remaining: [MAX_RANK]usize = undefined;
    var rem_count: usize = 0;

    for (0..s.ndim) |i| {
        if (i != src) {
            remaining[rem_count] = i;
            rem_count += 1;
        }
    }

    var rem_idx: usize = 0;
    for (0..s.ndim) |i| {
        if (i == dst) {
            perm[i] = src;
        } else {
            perm[i] = remaining[rem_idx];
            rem_idx += 1;
        }
    }

    return transpose(arr, .{ .axes = perm[0..s.ndim] });
}

/// Reverse the order of elements along the specified axis. Returns a zero-allocation view.
pub fn flip(arr: Array, options: struct { axis: ?isize = null }) ShapeError!Array {
    const s = arr.shape();
    if (s.ndim == 0) return arr.view();

    var v = arr.view();
    if (options.axis) |ax| {
        const norm_ax = try s.normalizeAxis(ax);
        v.stride_vals[norm_ax] = -v.stride_vals[norm_ax];
        // Shift data pointer to point to the last element of that dimension
        const dim_len: isize = @intCast(v.shape_dims[norm_ax]);
        const orig_stride: isize = arr.stride_vals[norm_ax];
        const offset_bytes = (dim_len - 1) * orig_stride * @as(isize, @intCast(v.dtype.sizeOf()));
        const byte_ptr: [*]u8 = @ptrCast(v.data_ptr);
        v.data_ptr = @ptrCast(if (offset_bytes >= 0) byte_ptr + @as(usize, @intCast(offset_bytes)) else byte_ptr - @as(usize, @intCast(-offset_bytes)));
    } else {
        // Reverse all axes
        for (0..s.ndim) |i| {
            v.stride_vals[i] = -v.stride_vals[i];
            const dim_len: isize = @intCast(v.shape_dims[i]);
            const orig_stride: isize = arr.stride_vals[i];
            const offset_bytes = (dim_len - 1) * orig_stride * @as(isize, @intCast(v.dtype.sizeOf()));
            const byte_ptr: [*]u8 = @ptrCast(v.data_ptr);
            v.data_ptr = @ptrCast(if (offset_bytes >= 0) byte_ptr + @as(usize, @intCast(offset_bytes)) else byte_ptr - @as(usize, @intCast(-offset_bytes)));
        }
    }
    const new_shape = v.shape();
    const new_st = v.strides();
    v.flags.isCContiguous = Strides.isCContiguous(new_shape, new_st);
    v.flags.isFContiguous = Strides.isFContiguous(new_shape, new_st);
    return v;
}

/// Roll array elements along a specified axis. Allocates a new array with elements shifted.
pub fn roll(
    arr: Array,
    options: struct {
        shift: isize,
        axis: ?isize = null,
    },
) (ShapeError || std.mem.Allocator.Error)!Array {
    var out = try arr.clone();
    errdefer out.deinit();

    const s = arr.shape();
    if (s.elementCount() == 0 or options.shift == 0) return out;

    if (options.axis) |ax| {
        const norm_ax = try s.normalizeAxis(ax);
        const dim_len = s.dims[norm_ax];
        const eff_shift = @mod(options.shift, @as(isize, @intCast(dim_len)));
        if (eff_shift == 0) return out;

        const NdIterator = @import("../core/iterator.zig").NdIterator;
        var it = NdIterator.init(s, arr.strides());
        var out_it = NdIterator.init(s, out.strides());

        while (it.next()) |item| {
            _ = out_it.next();
            var src_coords: [MAX_RANK]usize = undefined;
            var dst_coords: [MAX_RANK]usize = undefined;
            for (0..s.ndim) |dim_idx| {
                src_coords[dim_idx] = item.indices[dim_idx];
                dst_coords[dim_idx] = item.indices[dim_idx];
            }
            const shifted: usize = @intCast(@mod(@as(isize, @intCast(src_coords[norm_ax])) + eff_shift, @as(isize, @intCast(dim_len))));
            dst_coords[norm_ax] = shifted;

            const val = arr.getAsFloat(src_coords[0..s.ndim]) catch unreachable;
            out.setFromFloat(dst_coords[0..s.ndim], val) catch unreachable;
        }
    } else {
        // Roll over flattened elements
        const total = s.elementCount();
        const eff_shift = @mod(options.shift, @as(isize, @intCast(total)));
        if (eff_shift == 0) return out;

        var flat_src = try @import("reshape.zig").ravel(arr);
        defer flat_src.deinit();
        var flat_dst = try @import("reshape.zig").ravel(out);
        defer flat_dst.deinit();

        for (0..total) |i| {
            const dst_idx: usize = @intCast(@mod(@as(isize, @intCast(i)) + eff_shift, @as(isize, @intCast(total))));
            const val = flat_src.getAsFloat(&.{i}) catch unreachable;
            flat_dst.setFromFloat(&.{dst_idx}, val) catch unreachable;
        }
    }

    return out;
}

test "transpose and swapAxes" {
    const allocator = std.testing.allocator;
    const fromSlice = @import("../core/array.zig").fromSlice;

    const data = [_]f64{ 1, 2, 3, 4, 5, 6 };
    var a = try fromSlice(allocator, f64, .{ .data = &data, .shape = &.{ 2, 3 } });
    defer a.deinit();

    // Transpose 2x3 -> 3x2
    var at = try transpose(a, .{});
    defer at.deinit();

    try std.testing.expectEqualSlices(usize, &.{ 3, 2 }, at.shapeSlice());
    // a was [[1, 2, 3], [4, 5, 6]] -> at[0, 1] is 4.0
    try std.testing.expectEqual(@as(f64, 4.0), try at.get(f64, &.{ 0, 1 }));
    try std.testing.expectEqual(@as(f64, 2.0), try at.get(f64, &.{ 1, 0 }));

    // SwapAxes 0 and 1
    var sw = try swapAxes(a, 0, 1);
    defer sw.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 3, 2 }, sw.shapeSlice());
    try std.testing.expectEqual(@as(f64, 4.0), try sw.get(f64, &.{ 0, 1 }));
}
