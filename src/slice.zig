const std = @import("std");
const Allocator = std.mem.Allocator;
const core = @import("core.zig");

pub const Slice = union(enum) {
    all: void,
    index: usize,
    range: struct { start: usize, end: usize, step: usize },
};

/// Helper to create a range slice.
pub fn range(start: usize, end: usize) Slice {
    return Slice{ .range = .{ .start = start, .end = end, .step = 1 } };
}

/// Helper to create a stepped range slice.
pub fn rangeStep(start: usize, end: usize, step: usize) Slice {
    return Slice{ .range = .{ .start = start, .end = end, .step = step } };
}

/// Helper to create an 'all' slice (:).
pub fn all() Slice {
    return Slice{ .all = {} };
}

/// Helper to create an index slice.
pub fn index(idx: usize) Slice {
    return Slice{ .index = idx };
}

/// Calculates the output shape and offsets for a slicing operation.
/// This is a simplified version; full slicing logic is complex.
pub fn calculateSliceShape(allocator: Allocator, input_shape: []const usize, slices: []const Slice) ![]usize {
    var out_dims = std.ArrayListUnmanaged(usize){};
    defer out_dims.deinit(allocator);

    if (slices.len > input_shape.len) return core.Error.DimensionMismatch;

    for (slices, 0..) |s, i| {
        const dim_size = input_shape[i];
        switch (s) {
            .all => try out_dims.append(allocator, dim_size),
            .index => {
                // Index reduces dimension, so we don't append to out_dims
                // But we should check bounds
                if (s.index >= dim_size) return core.Error.IndexOutOfBounds;
            },
            .range => |r| {
                if (r.start >= dim_size or r.end > dim_size or r.start > r.end) return core.Error.IndexOutOfBounds;
                const len = (r.end - r.start + r.step - 1) / r.step;
                try out_dims.append(allocator, len);
            },
        }
    }

    // Append remaining dimensions if slices < rank
    for (slices.len..input_shape.len) |i| {
        try out_dims.append(allocator, input_shape[i]);
    }

    return out_dims.toOwnedSlice(allocator);
}

test "slice helpers" {
    const allocator = std.testing.allocator;

    // Test range helpers
    const r = range(0, 10);
    try std.testing.expectEqual(r.range.start, 0);
    try std.testing.expectEqual(r.range.end, 10);
    try std.testing.expectEqual(r.range.step, 1);

    const rs = rangeStep(0, 10, 2);
    try std.testing.expectEqual(rs.range.step, 2);

    // Test calculateSliceShape
    const shape = [_]usize{ 10, 20 };
    const slices = [_]Slice{ range(0, 5), all() };

    const new_shape = try calculateSliceShape(allocator, &shape, &slices);
    defer allocator.free(new_shape);

    try std.testing.expectEqual(new_shape.len, 2);
    try std.testing.expectEqual(new_shape[0], 5);
    try std.testing.expectEqual(new_shape[1], 20);
}
