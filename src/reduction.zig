const std = @import("std");
const Allocator = std.mem.Allocator;
const core = @import("core.zig");
const NDArray = core.NDArray;

/// Sum of array elements over a given axis.
pub fn sum(allocator: Allocator, comptime T: type, arr: NDArray(T), axis: ?usize) !NDArray(T) {
    if (axis) |ax| {
        if (ax >= arr.rank()) return core.Error.IndexOutOfBounds;
        // Reduce along axis
        // New shape: remove axis
        var new_shape = try allocator.alloc(usize, arr.rank() - 1);
        defer allocator.free(new_shape);

        var j: usize = 0;
        for (arr.shape, 0..) |dim, i| {
            if (i != ax) {
                new_shape[j] = dim;
                j += 1;
            }
        }

        var result = try NDArray(T).zeros(allocator, new_shape);
        errdefer result.deinit(allocator);

        // Iterate and sum
        // This is a naive implementation. Optimized version would use strides directly.
        var iter = try core.NdIterator.init(allocator, arr.shape);
        defer iter.deinit(allocator);

        while (iter.next()) |coords| {
            const val = try arr.get(coords);

            // Map coords to result coords
            var res_coords = try allocator.alloc(usize, new_shape.len);
            defer allocator.free(res_coords);

            var k: usize = 0;
            for (coords, 0..) |c, i| {
                if (i != ax) {
                    res_coords[k] = c;
                    k += 1;
                }
            }

            const current = try result.get(res_coords);
            try result.set(res_coords, current + val);
        }
        return result;
    } else {
        // Sum all
        var total: T = 0;
        for (arr.data) |val| {
            total += val;
        }
        var res = try NDArray(T).init(allocator, &.{1});
        res.data[0] = total;
        return res;
    }
}

/// Mean of array elements over a given axis.
pub fn mean(allocator: Allocator, comptime T: type, arr: NDArray(T), axis: ?usize) !NDArray(T) {
    const s = try sum(allocator, T, arr, axis);
    // Divide by count
    const count: T = if (axis) |ax| @floatFromInt(arr.shape[ax]) else @floatFromInt(arr.size());

    // In-place division if possible, or create new
    // Since sum returns a new array, we can modify it in place
    for (s.data) |*val| {
        val.* /= count;
    }
    return s;
}

/// Max of array elements.
pub fn max(allocator: Allocator, comptime T: type, arr: NDArray(T)) !T {
    _ = allocator;
    if (arr.size() == 0) return core.Error.DimensionMismatch; // Or empty
    var m = arr.data[0];
    for (arr.data[1..]) |val| {
        if (val > m) m = val;
    }
    return m;
}

/// Min of array elements.
pub fn min(allocator: Allocator, comptime T: type, arr: NDArray(T)) !T {
    _ = allocator;
    if (arr.size() == 0) return core.Error.DimensionMismatch;
    var m = arr.data[0];
    for (arr.data[1..]) |val| {
        if (val < m) m = val;
    }
    return m;
}

test "reduction sum mean" {
    const allocator = std.testing.allocator;
    var arr = try NDArray(f32).init(allocator, &.{ 2, 2 });
    defer arr.deinit(allocator);
    arr.fill(1.0);
    try arr.set(&.{ 0, 0 }, 2.0);

    // [[2, 1], [1, 1]]
    // Sum axis 0: [3, 2]
    var s0 = try sum(allocator, f32, arr, 0);
    defer s0.deinit(allocator);
    try std.testing.expectEqual(try s0.get(&.{0}), 3.0);
    try std.testing.expectEqual(try s0.get(&.{1}), 2.0);

    // Mean axis 1: [1.5, 1.0]
    var m1 = try mean(allocator, f32, arr, 1);
    defer m1.deinit(allocator);
    try std.testing.expectEqual(try m1.get(&.{0}), 1.5);
    try std.testing.expectEqual(try m1.get(&.{1}), 1.0);
}

test "reduction min max" {
    const allocator = std.testing.allocator;
    var arr = try NDArray(f32).init(allocator, &.{3});
    defer arr.deinit(allocator);
    arr.data[0] = 1.0;
    arr.data[1] = 5.0;
    arr.data[2] = -2.0;

    try std.testing.expectEqual(try max(allocator, f32, arr), 5.0);
    try std.testing.expectEqual(try min(allocator, f32, arr), -2.0);
}
