//! Multidimensional strided array iterators.
//!
//! Provides zero-allocation coordinate generators and offset calculators
//! for iterating through non-contiguous and broadcasted arrays.

const std = @import("std");
const Shape = @import("shape.zig").Shape;
const Strides = @import("shape.zig").Strides;
const MAX_RANK = @import("shape.zig").MAX_RANK;

/// Zero-allocation coordinate and offset iterator over an n-dimensional array.
pub const NdIterator = struct {
    shape: Shape,
    strides: Strides,
    indices: [MAX_RANK]usize = [_]usize{0} ** MAX_RANK,
    current_indices: [MAX_RANK]usize = [_]usize{0} ** MAX_RANK,
    offset: isize = 0,
    total: usize,
    count: usize = 0,

    /// Creates an iterator over the given shape and strides.
    pub fn init(shape: Shape, strides: Strides) NdIterator {
        return .{
            .shape = shape,
            .strides = strides,
            .total = shape.elementCount(),
        };
    }

    /// Resets the iterator to the initial state.
    pub fn reset(self: *NdIterator) void {
        @memset(&self.indices, 0);
        @memset(&self.current_indices, 0);
        self.offset = 0;
        self.count = 0;
    }

    /// Advances to the next element, returning its current indices and memory offset.
    pub fn next(self: *NdIterator) ?struct { indices: []const usize, offset: isize } {
        if (self.count >= self.total) return null;

        self.current_indices = self.indices;
        const current_offset = self.offset;
        self.count += 1;

        if (self.count < self.total and self.shape.ndim > 0) {
            // Advance indices from right-to-left
            var dim: usize = self.shape.ndim;
            while (dim > 0) {
                dim -= 1;
                self.indices[dim] += 1;
                self.offset += self.strides.values[dim];

                if (self.indices[dim] < self.shape.dims[dim]) {
                    break;
                } else {
                    // Carry over
                    self.offset -= @as(isize, @intCast(self.indices[dim])) * self.strides.values[dim];
                    self.indices[dim] = 0;
                }
            }
        }

        return .{
            .indices = self.current_indices[0..self.shape.ndim],
            .offset = current_offset,
        };
    }
};

/// Joint iterator over two arrays broadcasting to a common target shape.
pub const BroadcastPairIterator = struct {
    shape: Shape,
    strides_a: Strides,
    strides_b: Strides,
    indices: [MAX_RANK]usize = [_]usize{0} ** MAX_RANK,
    offset_a: isize = 0,
    offset_b: isize = 0,
    total: usize,
    count: usize = 0,

    /// Initializes a broadcast pair iterator for target shape and two strided arrays.
    pub fn init(target_shape: Shape, strides_a: Strides, strides_b: Strides) BroadcastPairIterator {
        return .{
            .shape = target_shape,
            .strides_a = strides_a,
            .strides_b = strides_b,
            .total = target_shape.elementCount(),
        };
    }

    /// Advances and returns the pair of memory offsets for array A and array B.
    pub fn next(self: *BroadcastPairIterator) ?struct { offset_a: isize, offset_b: isize } {
        if (self.count >= self.total) return null;

        const curr_a = self.offset_a;
        const curr_b = self.offset_b;
        self.count += 1;

        if (self.count < self.total and self.shape.ndim > 0) {
            var dim: usize = self.shape.ndim;
            while (dim > 0) {
                dim -= 1;
                self.indices[dim] += 1;
                self.offset_a += self.strides_a.values[dim];
                self.offset_b += self.strides_b.values[dim];

                if (self.indices[dim] < self.shape.dims[dim]) {
                    break;
                } else {
                    self.offset_a -= @as(isize, @intCast(self.indices[dim])) * self.strides_a.values[dim];
                    self.offset_b -= @as(isize, @intCast(self.indices[dim])) * self.strides_b.values[dim];
                    self.indices[dim] = 0;
                }
            }
        }

        return .{
            .offset_a = curr_a,
            .offset_b = curr_b,
        };
    }
};

test "nd iterator single array traversal" {
    const shape = try Shape.init(&.{ 2, 3 });
    const strides = Strides.fromShape(shape, .c); // [3, 1]

    var it = NdIterator.init(shape, strides);
    var expected_offset: isize = 0;

    while (it.next()) |item| {
        try std.testing.expectEqual(expected_offset, item.offset);
        expected_offset += 1;
    }
    try std.testing.expectEqual(@as(usize, 6), it.count);
}

test "broadcast pair iterator traversal" {
    const target = try Shape.init(&.{ 2, 3 });
    // Array A: shape [2, 1], strides [1, 0] (broadcast dim 1)
    const st_a = Strides{ .values = [_]isize{ 1, 0, 0, 0, 0, 0, 0, 0 }, .ndim = 2 };
    // Array B: shape [1, 3], strides [0, 1] (broadcast dim 0)
    const st_b = Strides{ .values = [_]isize{ 0, 1, 0, 0, 0, 0, 0, 0 }, .ndim = 2 };

    var it = BroadcastPairIterator.init(target, st_a, st_b);

    const expected_a = [_]isize{ 0, 0, 0, 1, 1, 1 };
    const expected_b = [_]isize{ 0, 1, 2, 0, 1, 2 };

    var idx: usize = 0;
    while (it.next()) |pair| : (idx += 1) {
        try std.testing.expectEqual(expected_a[idx], pair.offset_a);
        try std.testing.expectEqual(expected_b[idx], pair.offset_b);
    }
    try std.testing.expectEqual(@as(usize, 6), idx);
}
