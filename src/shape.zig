const std = @import("std");
const Allocator = std.mem.Allocator;
const core = @import("core.zig");
const NDArray = core.NDArray;

/// Reshapes an array to a new shape.
pub fn reshape(allocator: Allocator, comptime T: type, arr: NDArray(T), new_shape: []const usize) !NDArray(T) {
    return arr.reshape(allocator, new_shape);
}

/// Calculates the strides for a given shape assuming row-major (C-style) order.
pub fn calculateStrides(allocator: Allocator, shape: []const usize) ![]usize {
    const strides = try allocator.alloc(usize, shape.len);
    if (shape.len == 0) return strides;

    var stride: usize = 1;
    var i: usize = shape.len;
    while (i > 0) {
        i -= 1;
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

/// Broadcasts two shapes to a common shape.
/// Returns the new shape or an error if incompatible.
pub fn broadcastShapes(allocator: Allocator, shape1: []const usize, shape2: []const usize) ![]usize {
    const len1 = shape1.len;
    const len2 = shape2.len;
    const max_len = @max(len1, len2);

    var result = try allocator.alloc(usize, max_len);
    errdefer allocator.free(result);

    var i: usize = 0;
    while (i < max_len) : (i += 1) {
        const dim1 = if (i < len1) shape1[len1 - 1 - i] else 1;
        const dim2 = if (i < len2) shape2[len2 - 1 - i] else 1;

        if (dim1 == dim2) {
            result[max_len - 1 - i] = dim1;
        } else if (dim1 == 1) {
            result[max_len - 1 - i] = dim2;
        } else if (dim2 == 1) {
            result[max_len - 1 - i] = dim1;
        } else {
            return core.Error.ShapeMismatch;
        }
    }
    return result;
}

/// Checks if two shapes are compatible for broadcasting.
pub fn areBroadcastable(shape1: []const usize, shape2: []const usize) bool {
    const len1 = shape1.len;
    const len2 = shape2.len;
    const max_len = @max(len1, len2);

    var i: usize = 0;
    while (i < max_len) : (i += 1) {
        const dim1 = if (i < len1) shape1[len1 - 1 - i] else 1;
        const dim2 = if (i < len2) shape2[len2 - 1 - i] else 1;

        if (dim1 != dim2 and dim1 != 1 and dim2 != 1) {
            return false;
        }
    }
    return true;
}

test "shape broadcast" {
    const allocator = std.testing.allocator;

    const s1 = [_]usize{ 3, 1 };
    const s2 = [_]usize{2};

    const res = try broadcastShapes(allocator, &s1, &s2);
    defer allocator.free(res);

    try std.testing.expectEqual(res.len, 2);
    try std.testing.expectEqual(res[0], 3);
    try std.testing.expectEqual(res[1], 2);

    try std.testing.expect(areBroadcastable(&s1, &s2));

    const s3 = [_]usize{3};
    const s4 = [_]usize{2};
    try std.testing.expect(!areBroadcastable(&s3, &s4));
}
