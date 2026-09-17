//! Multidimensional strided slicing and sub-array viewing.
//!
//! Provides zero-allocation views for arbitrary stepped and strided slices
//! across n-dimensional arrays without copying underlying memory.

const std = @import("std");
const Array = @import("../core/array.zig").Array;
const Shape = @import("../core/shape.zig").Shape;
const Strides = @import("../core/shape.zig").Strides;
const Slice = @import("../core/shape.zig").Slice;
const MAX_RANK = @import("../core/shape.zig").MAX_RANK;
const IndexError = @import("../core/error.zig").IndexError;

/// Slices an array across its dimensions using a list of Slice descriptors.
/// Returns a zero-allocation view with adjusted base data pointer and strides.
pub fn slice(
    arr: Array,
    slices: []const Slice,
) IndexError!Array {
    if (slices.len > arr.ndim) return IndexError.RankMismatch;

    var new_dims: [MAX_RANK]usize = undefined;
    var new_strides: [MAX_RANK]isize = undefined;
    var offset_elements: isize = 0;

    for (0..arr.ndim) |dim| {
        if (dim < slices.len) {
            const sl = slices[dim];
            const resolved = try sl.resolve(arr.shape_dims[dim]);
            new_dims[dim] = resolved.len;
            new_strides[dim] = arr.stride_vals[dim] * resolved.step;
            offset_elements += @as(isize, @intCast(resolved.start)) * arr.stride_vals[dim];
        } else {
            // Keep unchanged
            new_dims[dim] = arr.shape_dims[dim];
            new_strides[dim] = arr.stride_vals[dim];
        }
    }

    const elem_sz = arr.dtype.sizeOf();
    const byte_offset = offset_elements * @as(isize, @intCast(elem_sz));

    var v = arr.view();
    if (byte_offset >= 0) {
        v.data_ptr = v.data_ptr + @as(usize, @intCast(byte_offset));
    } else {
        v.data_ptr = v.data_ptr - @as(usize, @intCast(-byte_offset));
    }

    for (0..arr.ndim) |i| {
        v.shape_dims[i] = new_dims[i];
        v.stride_vals[i] = new_strides[i];
    }

    const new_shape = v.shape();
    const new_st = v.strides();
    v.flags.isCContiguous = Strides.isCContiguous(new_shape, new_st);
    v.flags.isFContiguous = Strides.isFContiguous(new_shape, new_st);

    return v;
}

test "multidimensional strided slicing" {
    const allocator = std.testing.allocator;
    const fromSlice = @import("../core/array.zig").fromSlice;

    // 4x4 matrix: 0..15
    const data = [_]f64{
        0,  1,  2,  3,
        4,  5,  6,  7,
        8,  9,  10, 11,
        12, 13, 14, 15,
    };
    var a = try fromSlice(allocator, f64, .{ .data = &data, .shape = &.{ 4, 4 } });
    defer a.deinit();

    // Slice rows 1..3 with step 1, cols 1..4 with step 2 -> 2x2: [[5, 7], [9, 11]]
    var sub = try slice(a, &.{
        Slice{ .start = 1, .stop = 3, .step = 1 },
        Slice{ .start = 1, .stop = 4, .step = 2 },
    });
    defer sub.deinit();

    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, sub.shapeSlice());
    try std.testing.expectEqual(@as(f64, 5.0), try sub.get(f64, &.{ 0, 0 }));
    try std.testing.expectEqual(@as(f64, 7.0), try sub.get(f64, &.{ 0, 1 }));
    try std.testing.expectEqual(@as(f64, 9.0), try sub.get(f64, &.{ 1, 0 }));
    try std.testing.expectEqual(@as(f64, 11.0), try sub.get(f64, &.{ 1, 1 }));
}
