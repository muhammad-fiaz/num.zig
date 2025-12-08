const std = @import("std");
const core = @import("core.zig");
const NDArray = core.NDArray;
const Allocator = std.mem.Allocator;

/// Calculate the n-th discrete difference along the given axis.
///
/// Arguments:
///     allocator: Allocator.
///     T: Element type.
///     a: Input array.
///     n: The number of times values are differenced.
///     axis: The axis along which the difference is taken.
///
/// Returns:
///     The n-th differences. The shape of the output is the same as a except along axis where the dimension is smaller by n.
pub fn diff(allocator: Allocator, comptime T: type, a: NDArray(T), n: usize, axis: usize) !NDArray(T) {
    if (n == 0) return a.copy(allocator);
    if (axis >= a.rank()) return core.Error.IndexOutOfBounds;
    if (a.shape[axis] < n) return core.Error.DimensionMismatch;

    var current = try a.copy(allocator);

    for (0..n) |_| {
        var prev = current;
        defer prev.deinit(allocator);

        // Calculate difference along axis
        // New shape has dim - 1 along axis
        var new_shape = try allocator.alloc(usize, prev.rank());
        errdefer allocator.free(new_shape);
        @memcpy(new_shape, prev.shape);
        new_shape[axis] -= 1;

        current = try NDArray(T).init(allocator, new_shape);
        errdefer current.deinit(allocator);
        allocator.free(new_shape);

        // Iterate over result
        var iter = try core.NdIterator.init(allocator, current.shape);
        defer iter.deinit(allocator);

        const src_coords_1 = try allocator.alloc(usize, prev.rank());
        defer allocator.free(src_coords_1);
        const src_coords_2 = try allocator.alloc(usize, prev.rank());
        defer allocator.free(src_coords_2);

        while (iter.next()) |coords| {
            @memcpy(src_coords_1, coords);
            @memcpy(src_coords_2, coords);

            // src_coords_1 points to i
            // src_coords_2 points to i+1
            src_coords_2[axis] += 1;

            const val1 = try prev.get(src_coords_1);
            const val2 = try prev.get(src_coords_2);

            try current.set(coords, val2 - val1);
        }
    }

    return current;
}

/// Return the gradient of an N-dimensional array.
/// The gradient is computed using second order accurate central differences in the interior points
/// and either first or second order accurate one-sides (forward or backward) differences at the boundaries.
///
/// Uses central difference: $(f(x+h) - f(x-h)) / 2h$
pub fn gradient(allocator: Allocator, comptime T: type, a: NDArray(T), axis: usize) !NDArray(T) {
    if (axis >= a.rank()) return core.Error.IndexOutOfBounds;

    var result = try NDArray(T).init(allocator, a.shape);
    errdefer result.deinit(allocator);

    const dim_size = a.shape[axis];
    if (dim_size < 2) return core.Error.DimensionMismatch;

    var iter = try core.NdIterator.init(allocator, a.shape);
    defer iter.deinit(allocator);

    var coords_prev = try allocator.alloc(usize, a.rank());
    defer allocator.free(coords_prev);
    var coords_next = try allocator.alloc(usize, a.rank());
    defer allocator.free(coords_next);

    while (iter.next()) |coords| {
        const idx = coords[axis];

        if (idx == 0) {
            // Forward difference
            @memcpy(coords_next, coords);
            coords_next[axis] = 1;
            const val0 = try a.get(coords);
            const val1 = try a.get(coords_next);
            try result.set(coords, val1 - val0);
        } else if (idx == dim_size - 1) {
            // Backward difference
            @memcpy(coords_prev, coords);
            coords_prev[axis] = dim_size - 2;
            const val_last = try a.get(coords);
            const val_prev = try a.get(coords_prev);
            try result.set(coords, val_last - val_prev);
        } else {
            // Central difference
            @memcpy(coords_prev, coords);
            coords_prev[axis] = idx - 1;
            @memcpy(coords_next, coords);
            coords_next[axis] = idx + 1;

            const val_prev = try a.get(coords_prev);
            const val_next = try a.get(coords_next);
            try result.set(coords, (val_next - val_prev) / 2.0);
        }
    }

    return result;
}

test "diff" {
    const allocator = std.testing.allocator;
    var arr = try NDArray(f32).init(allocator, &.{5});
    defer arr.deinit(allocator);
    // 1, 2, 4, 7, 0
    try arr.set(&.{0}, 1.0);
    try arr.set(&.{1}, 2.0);
    try arr.set(&.{2}, 4.0);
    try arr.set(&.{3}, 7.0);
    try arr.set(&.{4}, 0.0);

    var d1 = try diff(allocator, f32, arr, 1, 0);
    defer d1.deinit(allocator);
    // 1, 2, 3, -7
    try std.testing.expectEqual(d1.shape[0], 4);
    try std.testing.expectEqual(try d1.get(&.{0}), 1.0);
    try std.testing.expectEqual(try d1.get(&.{1}), 2.0);
    try std.testing.expectEqual(try d1.get(&.{2}), 3.0);
    try std.testing.expectEqual(try d1.get(&.{3}), -7.0);

    var d2 = try diff(allocator, f32, arr, 2, 0);
    defer d2.deinit(allocator);
    // 1, 1, -10
    try std.testing.expectEqual(d2.shape[0], 3);
    try std.testing.expectEqual(try d2.get(&.{0}), 1.0);
    try std.testing.expectEqual(try d2.get(&.{1}), 1.0);
    try std.testing.expectEqual(try d2.get(&.{2}), -10.0);
}

test "gradient" {
    const allocator = std.testing.allocator;
    var arr = try NDArray(f32).init(allocator, &.{3});
    defer arr.deinit(allocator);
    // 1, 2, 4
    try arr.set(&.{0}, 1.0);
    try arr.set(&.{1}, 2.0);
    try arr.set(&.{2}, 4.0);

    var g = try gradient(allocator, f32, arr, 0);
    defer g.deinit(allocator);

    // 0: (2-1) = 1
    // 1: (4-1)/2 = 1.5
    // 2: (4-2) = 2
    try std.testing.expectEqual(try g.get(&.{0}), 1.0);
    try std.testing.expectEqual(try g.get(&.{1}), 1.5);
    try std.testing.expectEqual(try g.get(&.{2}), 2.0);
}
