const std = @import("std");
const Allocator = std.mem.Allocator;
const core = @import("core.zig");
const NDArray = core.NDArray;

/// Compute the broadcasted shape of two arrays.
/// Caller owns the returned slice.
pub fn broadcastShape(allocator: Allocator, shape_a: []const usize, shape_b: []const usize) ![]usize {
    return core.broadcastShape(allocator, shape_a, shape_b);
}

/// Compute the broadcasted shape of multiple arrays.
/// Caller owns the returned slice.
pub fn broadcastShapes(allocator: Allocator, shapes: []const []const usize) ![]usize {
    if (shapes.len == 0) return try allocator.alloc(usize, 0);
    if (shapes.len == 1) {
        const res = try allocator.alloc(usize, shapes[0].len);
        @memcpy(res, shapes[0]);
        return res;
    }

    var current_shape = try allocator.alloc(usize, shapes[0].len);
    errdefer allocator.free(current_shape);
    @memcpy(current_shape, shapes[0]);

    for (shapes[1..]) |shape| {
        const next_shape = try core.broadcastShape(allocator, current_shape, shape);
        allocator.free(current_shape);
        current_shape = next_shape;
    }
    return current_shape;
}

/// Compute the strides for an array broadcasted to a target shape.
/// Caller owns the returned slice.
pub fn broadcastStrides(allocator: Allocator, source_shape: []const usize, source_strides: []const usize, target_shape: []const usize) ![]usize {
    const rank = target_shape.len;
    const strides = try allocator.alloc(usize, rank);
    errdefer allocator.free(strides);

    var i: usize = 0;
    while (i < rank) : (i += 1) {
        // i is the dimension index from 0 to rank-1 (left to right)
        // We need to match from the right (end)

        const result_dim_idx = rank - 1 - i;

        if (i < source_shape.len) {
            const source_dim_idx = source_shape.len - 1 - i;
            if (source_shape[source_dim_idx] == 1) {
                // Broadcast dimension
                strides[result_dim_idx] = 0;
            } else if (source_shape[source_dim_idx] == target_shape[result_dim_idx]) {
                // Matching dimension
                strides[result_dim_idx] = source_strides[source_dim_idx];
            } else {
                return error.ShapeMismatch;
            }
        } else {
            // New dimension (prepended)
            strides[result_dim_idx] = 0;
        }
    }
    return strides;
}

/// Iterator for iterating over a broadcasted shape.
pub const BroadcastIterator = struct {
    allocator: Allocator,
    shape: []const usize,
    coords: []usize,
    index: usize,
    size: usize,

    pub fn init(allocator: Allocator, shape: []const usize) !BroadcastIterator {
        const coords = try allocator.alloc(usize, shape.len);
        @memset(coords, 0);

        var size: usize = 1;
        for (shape) |d| size *= d;

        return BroadcastIterator{
            .allocator = allocator,
            .shape = shape,
            .coords = coords,
            .index = 0,
            .size = size,
        };
    }

    pub fn deinit(self: *BroadcastIterator) void {
        self.allocator.free(self.coords);
    }

    /// Advance the iterator and return the current coordinates.
    /// Returns null if finished.
    pub fn next(self: *BroadcastIterator) ?[]const usize {
        if (self.index >= self.size) return null;

        // If this is not the first element, advance coords
        if (self.index > 0) {
            var dim = self.shape.len - 1;
            while (true) {
                self.coords[dim] += 1;
                if (self.coords[dim] < self.shape[dim]) break;
                self.coords[dim] = 0;
                if (dim == 0) break; // Should not happen if index < size
                dim -= 1;
            }
        }

        self.index += 1;
        return self.coords;
    }
};

test "broadcast strides" {
    const allocator = std.testing.allocator;

    // Source: (3, 1)
    // Target: (2, 3, 4)
    // Result strides should handle the broadcasting

    const source_shape = [_]usize{ 3, 1 };
    const source_strides = [_]usize{ 1, 1 }; // Dummy strides
    const target_shape = [_]usize{ 2, 3, 4 };

    const strides = try broadcastStrides(allocator, &source_shape, &source_strides, &target_shape);
    defer allocator.free(strides);

    // Target dims: 0 (2), 1 (3), 2 (4)
    // Source dims map to: Target 1 (3), Target 2 (1->4)

    // Target 2 (size 4): Source dim 1 (size 1) -> Broadcast -> Stride 0
    // Target 1 (size 3): Source dim 0 (size 3) -> Match -> Stride 1
    // Target 0 (size 2): New dim -> Broadcast -> Stride 0

    try std.testing.expectEqual(strides[0], 0);
    try std.testing.expectEqual(strides[1], 1);
    try std.testing.expectEqual(strides[2], 0);
}

test "broadcast shapes" {
    const allocator = std.testing.allocator;
    const s1 = [_]usize{ 3, 1 };
    const s2 = [_]usize{ 1, 4 };
    const shapes = [_][]const usize{ &s1, &s2 };

    const res = try broadcastShapes(allocator, &shapes);
    defer allocator.free(res);

    try std.testing.expectEqual(res.len, 2);
    try std.testing.expectEqual(res[0], 3);
    try std.testing.expectEqual(res[1], 4);
}

test "broadcast iterator" {
    const allocator = std.testing.allocator;
    const shape = [_]usize{ 2, 2 };
    var iter = try BroadcastIterator.init(allocator, &shape);
    defer iter.deinit();

    var count: usize = 0;
    while (iter.next()) |_| {
        count += 1;
    }
    try std.testing.expectEqual(count, 4);
}
