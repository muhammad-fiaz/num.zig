//! Array reshaping, flattening, squeezing, and dimension expansion.
//!
//! Provides zero-allocation views for compatible contiguous layouts and safe copies
//! when memory rearrangement is required.

const std = @import("std");
const Array = @import("../core/array.zig").Array;
const empty = @import("../core/array.zig").empty;
const Shape = @import("../core/shape.zig").Shape;
const Strides = @import("../core/shape.zig").Strides;
const Order = @import("../core/shape.zig").Order;
const MAX_RANK = @import("../core/shape.zig").MAX_RANK;
const ShapeError = @import("../core/error.zig").ShapeError;

/// Reshapes an array to a new shape. Returns a zero-allocation view if contiguous,
/// or creates a contiguous copy if the layout cannot be viewed directly.
pub fn reshape(
    arr: Array,
    options: struct {
        shape: []const isize,
        order: Order = .c,
    },
) (ShapeError || std.mem.Allocator.Error)!Array {
    if (options.shape.len > MAX_RANK) return ShapeError.RankExceeded;

    const total = arr.elementCount();
    var inferred_idx: ?usize = null;
    var product: usize = 1;
    var resolved_dims: [MAX_RANK]usize = undefined;

    for (options.shape, 0..) |d, i| {
        if (d == -1) {
            if (inferred_idx != null) return ShapeError.InvalidDimension;
            inferred_idx = i;
        } else if (d < -1) {
            return ShapeError.InvalidDimension;
        } else {
            const ud: usize = @intCast(d);
            resolved_dims[i] = ud;
            product *= ud;
        }
    }

    if (inferred_idx) |idx| {
        if (product == 0 or total % product != 0) return ShapeError.ReshapeMismatch;
        resolved_dims[idx] = total / product;
    } else {
        if (product != total) return ShapeError.ReshapeMismatch;
    }

    const new_shape = Shape{
        .dims = resolved_dims,
        .ndim = @intCast(options.shape.len),
    };

    // Zero-allocation view path if array is contiguous
    if (arr.isContiguous()) {
        const new_strides = Strides.fromShape(new_shape, options.order);
        var v = arr.view();
        v.ndim = new_shape.ndim;
        for (0..new_shape.ndim) |i| {
            v.shape_dims[i] = new_shape.dims[i];
            v.stride_vals[i] = new_strides.values[i];
        }
        v.flags.isCContiguous = Strides.isCContiguous(new_shape, new_strides);
        v.flags.isFContiguous = Strides.isFContiguous(new_shape, new_strides);
        return v;
    }

    // Materialize a contiguous copy if non-contiguous
    var copy = try arr.clone();
    errdefer copy.deinit();
    return reshape(copy, options);
}

/// Returns a flattened 1D view if array is contiguous, or a flattened copy if not.
pub fn ravel(arr: Array) (ShapeError || std.mem.Allocator.Error)!Array {
    return reshape(arr, .{ .shape = &.{-1} });
}

/// Returns a copy of the array collapsed into one dimension. Caller owns returned array.
pub fn flatten(arr: Array) (ShapeError || std.mem.Allocator.Error)!Array {
    var copy = try arr.clone();
    errdefer copy.deinit();
    return reshape(copy, .{ .shape = &.{-1} });
}

/// Removes single-dimensional entries from the shape of an array. Zero-allocation view.
pub fn squeeze(
    arr: Array,
    options: struct {
        axis: ?isize = null,
    },
) ShapeError!Array {
    const s = arr.shape();
    const st = arr.strides();

    var new_dims: [MAX_RANK]usize = undefined;
    var new_strides: [MAX_RANK]isize = undefined;
    var out_ndim: u8 = 0;

    if (options.axis) |ax| {
        const target_ax = try s.normalizeAxis(ax);
        for (0..s.ndim) |i| {
            if (i == target_ax) {
                if (s.dims[i] != 1) return ShapeError.InvalidDimension;
            } else {
                new_dims[out_ndim] = s.dims[i];
                new_strides[out_ndim] = st.values[i];
                out_ndim += 1;
            }
        }
    } else {
        for (0..s.ndim) |i| {
            if (s.dims[i] != 1) {
                new_dims[out_ndim] = s.dims[i];
                new_strides[out_ndim] = st.values[i];
                out_ndim += 1;
            }
        }
    }

    var v = arr.view();
    v.ndim = out_ndim;
    for (0..out_ndim) |i| {
        v.shape_dims[i] = new_dims[i];
        v.stride_vals[i] = new_strides[i];
    }
    return v;
}

/// Expands the shape of an array by inserting a new axis of size 1 at the specified position.
/// Zero-allocation view.
pub fn expandDims(
    arr: Array,
    options: struct {
        axis: isize,
    },
) ShapeError!Array {
    const s = arr.shape();
    if (s.ndim >= MAX_RANK) return ShapeError.RankExceeded;

    const rank_i: isize = @intCast(s.ndim);
    if (options.axis < -(rank_i + 1) or options.axis > rank_i) {
        return ShapeError.AxisOutOfBounds;
    }

    const insert_ax: usize = if (options.axis < 0)
        @intCast(options.axis + rank_i + 1)
    else
        @intCast(options.axis);

    const st = arr.strides();
    var new_dims: [MAX_RANK]usize = undefined;
    var new_strides: [MAX_RANK]isize = undefined;
    var src_idx: usize = 0;

    for (0..s.ndim + 1) |i| {
        if (i == insert_ax) {
            new_dims[i] = 1;
            new_strides[i] = if (src_idx < s.ndim) st.values[src_idx] else 1;
        } else {
            new_dims[i] = s.dims[src_idx];
            new_strides[i] = st.values[src_idx];
            src_idx += 1;
        }
    }

    var v = arr.view();
    v.ndim = s.ndim + 1;
    for (0..v.ndim) |i| {
        v.shape_dims[i] = new_dims[i];
        v.stride_vals[i] = new_strides[i];
    }
    return v;
}

/// Convert input to an array with at least one dimension.
pub fn atleast1d(arr: Array) ShapeError!Array {
    if (arr.ndim == 0) {
        return expandDims(arr, .{ .axis = 0 });
    }
    return arr.view();
}

/// Convert input to an array with at least two dimensions.
pub fn atleast2d(arr: Array) ShapeError!Array {
    if (arr.ndim == 0) {
        const a1 = try expandDims(arr, .{ .axis = 0 });
        return expandDims(a1, .{ .axis = 0 });
    } else if (arr.ndim == 1) {
        return expandDims(arr, .{ .axis = 0 });
    }
    return arr.view();
}

/// Convert input to an array with at least three dimensions.
pub fn atleast3d(arr: Array) ShapeError!Array {
    if (arr.ndim == 0) {
        const a1 = try expandDims(arr, .{ .axis = 0 });
        const a2 = try expandDims(a1, .{ .axis = 0 });
        return expandDims(a2, .{ .axis = 0 });
    } else if (arr.ndim == 1) {
        const a1 = try expandDims(arr, .{ .axis = 0 });
        return expandDims(a1, .{ .axis = -1 });
    } else if (arr.ndim == 2) {
        return expandDims(arr, .{ .axis = -1 });
    }
    return arr.view();
}

/// Converts a flat index or array of flat indices into a 2D coordinate array of shape [indices.len, shape.len].
pub fn unravelIndex(
    allocator: std.mem.Allocator,
    flat_indices: []const usize,
    dims: []const usize,
) (ShapeError || std.mem.Allocator.Error)!Array {
    if (dims.len == 0 or dims.len > MAX_RANK) return ShapeError.InvalidDimension;

    const n_indices = flat_indices.len;
    const ndim = dims.len;

    var out = try empty(allocator, .{ .shape = &.{ n_indices, ndim }, .dtype = .i64 });
    errdefer out.deinit();

    for (flat_indices, 0..) |flat_idx, row| {
        var rem = flat_idx;
        var d = ndim;
        while (d > 0) {
            d -= 1;
            const coord = rem % dims[d];
            rem /= dims[d];
            out.set(i64, &.{ row, d }, @intCast(coord)) catch unreachable;
        }
    }

    return out;
}

/// Converts a 2D coordinate array of shape [n, ndim] into a 1D array of flat indices.
pub fn ravelIndex(
    allocator: std.mem.Allocator,
    coords: Array,
    dims: []const usize,
) (ShapeError || std.mem.Allocator.Error)!Array {
    if (coords.ndim != 2 or coords.shape_dims[1] != dims.len) return ShapeError.InvalidDimension;
    const n = coords.shape_dims[0];
    const ndim = dims.len;

    var out = try empty(allocator, .{ .shape = &.{n}, .dtype = .i64 });
    errdefer out.deinit();

    for (0..n) |row| {
        var flat_idx: usize = 0;
        var multiplier: usize = 1;
        var d = ndim;
        while (d > 0) {
            d -= 1;
            const coord: usize = @intCast(coords.getAsInt(&.{ row, d }) catch 0);
            flat_idx += coord * multiplier;
            multiplier *= dims[d];
        }
        out.set(i64, &.{row}, @intCast(flat_idx)) catch unreachable;
    }

    return out;
}

/// Return an array of grid coordinates for the given shape.
pub fn indices(
    allocator: std.mem.Allocator,
    dims: []const usize,
) (ShapeError || std.mem.Allocator.Error)!Array {
    if (dims.len == 0 or dims.len > MAX_RANK) return ShapeError.InvalidDimension;
    const ndim = dims.len;

    var out_dims: [MAX_RANK + 1]usize = undefined;
    out_dims[0] = ndim;
    for (0..ndim) |i| {
        out_dims[i + 1] = dims[i];
    }
    const out_shape = Shape{ .dims = out_dims[0..MAX_RANK].*, .ndim = @intCast(ndim + 1) };

    var out = try empty(allocator, .{ .shape = out_dims[0 .. ndim + 1], .dtype = .i64 });
    errdefer out.deinit();

    var it = @import("../core/iterator.zig").NdIterator.init(out_shape, out.strides());
    while (it.next()) |item| {
        const d = item.indices[0];
        const val = item.indices[d + 1];
        out.set(i64, item.indices[0 .. ndim + 1], @intCast(val)) catch unreachable;
    }

    return out;
}

test "reshape, ravel, squeeze, and expandDims" {
    const allocator = std.testing.allocator;
    const fromSlice = @import("../core/array.zig").fromSlice;

    const data = [_]f64{ 1, 2, 3, 4, 5, 6 };
    var a = try fromSlice(allocator, f64, .{ .data = &data, .shape = &.{6} });
    defer a.deinit();

    // Reshape to 2x3 view with -1 inference
    var r = try reshape(a, .{ .shape = &.{ 2, -1 } });
    defer r.deinit();

    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, r.shapeSlice());
    try std.testing.expectEqual(@as(f64, 4.0), try r.get(f64, &.{ 1, 0 }));

    // Expand dims at axis 0: [1, 2, 3]
    var exp = try expandDims(r, .{ .axis = 0 });
    defer exp.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 2, 3 }, exp.shapeSlice());

    // Squeeze back to [2, 3]
    var sq = try squeeze(exp, .{ .axis = 0 });
    defer sq.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, sq.shapeSlice());

    // Ravel to 1D
    var rav = try ravel(r);
    defer rav.deinit();
    try std.testing.expectEqualSlices(usize, &.{6}, rav.shapeSlice());

    // Coordinate utilities
    // Flat index 5 in shape (2, 3) is (1, 2)
    const flat_idx = [_]usize{ 0, 5 };
    var unrav = try unravelIndex(allocator, &flat_idx, &.{ 2, 3 });
    defer unrav.deinit();
    try std.testing.expectEqual(@as(i64, 0), try unrav.get(i64, &.{ 0, 0 }));
    try std.testing.expectEqual(@as(i64, 0), try unrav.get(i64, &.{ 0, 1 }));
    try std.testing.expectEqual(@as(i64, 1), try unrav.get(i64, &.{ 1, 0 }));
    try std.testing.expectEqual(@as(i64, 2), try unrav.get(i64, &.{ 1, 1 }));

    var rav_back = try ravelIndex(allocator, unrav, &.{ 2, 3 });
    defer rav_back.deinit();
    try std.testing.expectEqual(@as(i64, 0), try rav_back.get(i64, &.{0}));
    try std.testing.expectEqual(@as(i64, 5), try rav_back.get(i64, &.{1}));

    var grid = try indices(allocator, &.{ 2, 3 });
    defer grid.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2, 3 }, grid.shapeSlice());
    try std.testing.expectEqual(@as(i64, 0), try grid.get(i64, &.{ 0, 0, 2 }));
    try std.testing.expectEqual(@as(i64, 1), try grid.get(i64, &.{ 0, 1, 2 }));
    try std.testing.expectEqual(@as(i64, 2), try grid.get(i64, &.{ 1, 0, 2 }));
}
