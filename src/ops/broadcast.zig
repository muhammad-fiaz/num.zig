//! Broadcasting utilities for multidimensional arrays.
//!
//! Provides zero-allocation view broadcasting for single arrays and pairs of arrays
//! following standard numerical array broadcasting semantics.

const std = @import("std");
const Array = @import("../core/array.zig").Array;
const Shape = @import("../core/shape.zig").Shape;
const Strides = @import("../core/shape.zig").Strides;
const broadcastShapes = @import("../core/shape.zig").broadcastShapes;
const broadcastStrides = @import("../core/shape.zig").broadcastStrides;
const ShapeError = @import("../core/error.zig").ShapeError;

/// Broadcasts an array to a target shape, returning a zero-allocation view with adapted strides.
pub fn broadcastTo(arr: Array, target_shape_slice: []const usize) ShapeError!Array {
    const target_shape = try Shape.init(target_shape_slice);
    const src_shape = arr.shape();
    const src_strides = arr.strides();

    const new_strides = try broadcastStrides(src_shape, src_strides, target_shape);

    var out = arr.view();
    out.ndim = target_shape.ndim;
    for (0..target_shape.ndim) |i| {
        out.shape_dims[i] = target_shape.dims[i];
        out.stride_vals[i] = new_strides.values[i];
    }

    out.flags.isCContiguous = Strides.isCContiguous(target_shape, new_strides);
    out.flags.isFContiguous = Strides.isFContiguous(target_shape, new_strides);

    return out;
}

/// Broadcasts two arrays to their common broadcasted shape, returning two zero-allocation views.
pub fn broadcast2(a: Array, b: Array) ShapeError!struct { a: Array, b: Array, target_shape: Shape } {
    const shape_a = a.shape();
    const shape_b = b.shape();
    const target_shape = try broadcastShapes(shape_a, shape_b);

    const strides_a = try broadcastStrides(shape_a, a.strides(), target_shape);
    const strides_b = try broadcastStrides(shape_b, b.strides(), target_shape);

    var view_a = a.view();
    view_a.ndim = target_shape.ndim;
    for (0..target_shape.ndim) |i| {
        view_a.shape_dims[i] = target_shape.dims[i];
        view_a.stride_vals[i] = strides_a.values[i];
    }
    view_a.flags.isCContiguous = Strides.isCContiguous(target_shape, strides_a);
    view_a.flags.isFContiguous = Strides.isFContiguous(target_shape, strides_a);

    var view_b = b.view();
    view_b.ndim = target_shape.ndim;
    for (0..target_shape.ndim) |i| {
        view_b.shape_dims[i] = target_shape.dims[i];
        view_b.stride_vals[i] = strides_b.values[i];
    }
    view_b.flags.isCContiguous = Strides.isCContiguous(target_shape, strides_b);
    view_b.flags.isFContiguous = Strides.isFContiguous(target_shape, strides_b);

    return .{
        .a = view_a,
        .b = view_b,
        .target_shape = target_shape,
    };
}

test "broadcastTo and broadcast2" {
    const allocator = std.testing.allocator;
    const zeros = @import("../core/array.zig").zeros;

    var a = try zeros(allocator, .{ .shape = &.{ 1, 3 }, .dtype = .f32 });
    defer a.deinit();

    var b = try zeros(allocator, .{ .shape = &.{ 2, 1 }, .dtype = .f32 });
    defer b.deinit();

    const bc = try broadcast2(a, b);
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, bc.target_shape.slice());
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, bc.a.shapeSlice());
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, bc.b.shapeSlice());

    // a had shape [1, 3] -> broadcast to [2, 3] -> stride 0 is 0
    try std.testing.expectEqual(@as(isize, 0), bc.a.stridesSlice()[0]);
    try std.testing.expectEqual(@as(isize, 1), bc.a.stridesSlice()[1]);

    // b had shape [2, 1] -> broadcast to [2, 3] -> stride 1 is 0
    try std.testing.expectEqual(@as(isize, 1), bc.b.stridesSlice()[0]);
    try std.testing.expectEqual(@as(isize, 0), bc.b.stridesSlice()[1]);
}
