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
        errdefer result.deinit();

        // Iterate and sum
        // This is a naive implementation. Optimized version would use strides directly.
        var iter = try core.NdIterator.init(allocator, arr.shape);
        defer iter.deinit();

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
