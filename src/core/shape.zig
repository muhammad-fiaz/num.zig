//! Shape, strides, memory layout order, and broadcasting resolution.
//!
//! Implements multi-dimensional array geometry with Small Buffer Optimization (SBO)
//! supporting up to MAX_RANK (8) dimensions with zero heap allocations.

const std = @import("std");
const ShapeError = @import("error.zig").ShapeError;
const IndexError = @import("error.zig").IndexError;
const DType = @import("dtype.zig").DType;

/// Maximum number of dimensions supported inline without heap allocation.
pub const MAX_RANK: usize = 8;

/// Memory storage ordering convention.
pub const Order = enum {
    /// Row-major order (standard C layout, last dimension varies fastest).
    c,
    /// Column-major order (Fortran layout, first dimension varies fastest).
    fortran,
};

/// Slice descriptor for dimension-wise indexing.
pub const Slice = struct {
    start: ?isize = null,
    stop: ?isize = null,
    step: isize = 1,

    /// Resolves canonical start, length, and step for a dimension of length `dim_len`.
    pub fn resolve(self: Slice, dim_len: usize) IndexError!struct { start: usize, len: usize, step: isize } {
        if (self.step == 0) return IndexError.StepCannotBeZero;
        const len_i: isize = @intCast(dim_len);

        var start: isize = 0;
        var stop: isize = 0;

        if (self.step > 0) {
            start = self.start orelse 0;
            stop = self.stop orelse len_i;

            if (start < 0) start += len_i;
            if (stop < 0) stop += len_i;

            start = std.math.clamp(start, 0, len_i);
            stop = std.math.clamp(stop, 0, len_i);

            if (start >= stop) {
                return .{ .start = @intCast(start), .len = 0, .step = self.step };
            }
            const count: usize = @intCast(@divFloor(stop - start - 1, self.step) + 1);
            return .{ .start = @intCast(start), .len = count, .step = self.step };
        } else {
            start = self.start orelse (len_i - 1);
            stop = self.stop orelse -1;

            if (start < 0) start += len_i;
            if (stop < 0 and self.stop != null) stop += len_i;

            start = std.math.clamp(start, -1, len_i - 1);
            stop = std.math.clamp(stop, -1, len_i - 1);

            if (start <= stop) {
                return .{ .start = if (start < 0) 0 else @intCast(start), .len = 0, .step = self.step };
            }
            const count: usize = @intCast(@divFloor(start - stop - 1, -self.step) + 1);
            return .{ .start = @intCast(start), .len = count, .step = self.step };
        }
    }
};

/// Multidimensional shape descriptor.
pub const Shape = struct {
    dims: [MAX_RANK]usize = [_]usize{0} ** MAX_RANK,
    ndim: u8 = 0,

    /// Construct a Shape from a slice of dimension sizes.
    pub fn init(dims_slice: []const usize) ShapeError!Shape {
        if (dims_slice.len > MAX_RANK) return ShapeError.RankExceeded;
        var self = Shape{ .ndim = @intCast(dims_slice.len) };
        for (dims_slice, 0..) |d, i| {
            self.dims[i] = d;
        }
        return self;
    }

    /// Construct a scalar (rank 0) shape.
    pub fn scalar() Shape {
        return .{ .dims = [_]usize{0} ** MAX_RANK, .ndim = 0 };
    }

    /// Construct a 1-D vector shape.
    pub fn vector(n: usize) Shape {
        var s = Shape{ .ndim = 1 };
        s.dims[0] = n;
        return s;
    }

    /// Construct a 2-D matrix shape.
    pub fn matrix(rows: usize, cols: usize) Shape {
        var s = Shape{ .ndim = 2 };
        s.dims[0] = rows;
        s.dims[1] = cols;
        return s;
    }

    /// Borrow dimension slice.
    pub fn slice(self: *const Shape) []const usize {
        return self.dims[0..self.ndim];
    }

    /// Returns the total number of elements represented by this shape, failing on overflow.
    pub fn elementCountChecked(self: Shape) ShapeError!usize {
        if (self.ndim == 0) return 1;
        var count: usize = 1;
        for (self.slice()) |d| {
            count = std.math.mul(usize, count, d) catch return ShapeError.InvalidDimension;
        }
        return count;
    }

    /// Returns the total number of elements represented by this shape (clamped/checked).
    pub fn elementCount(self: Shape) usize {
        return self.elementCountChecked() catch std.math.maxInt(usize);
    }

    /// Returns the total byte size for this shape with the given data type, failing on overflow.
    pub fn byteCountChecked(self: Shape, dtype: DType) ShapeError!usize {
        const elems = try self.elementCountChecked();
        return std.math.mul(usize, elems, dtype.sizeOf()) catch ShapeError.InvalidDimension;
    }

    /// Returns the total byte size for this shape with the given data type.
    pub fn byteCount(self: Shape, dtype: DType) usize {
        return self.byteCountChecked(dtype) catch std.math.maxInt(usize);
    }

    /// Compares two shapes for structural equality.
    pub fn equal(self: Shape, other: Shape) bool {
        if (self.ndim != other.ndim) return false;
        for (self.slice(), other.slice()) |a, b| {
            if (a != b) return false;
        }
        return true;
    }

    /// Normalizes an axis index in [-ndim, ndim - 1] to [0, ndim - 1].
    pub fn normalizeAxis(self: Shape, axis: isize) ShapeError!usize {
        if (self.ndim == 0) {
            if (axis == 0 or axis == -1) return 0;
            return ShapeError.AxisOutOfBounds;
        }
        const rank_i: isize = @intCast(self.ndim);
        if (axis < -rank_i or axis >= rank_i) {
            return ShapeError.AxisOutOfBounds;
        }
        if (axis < 0) {
            return @intCast(axis + rank_i);
        }
        return @intCast(axis);
    }
};

/// Multidimensional strides descriptor (in element counts).
pub const Strides = struct {
    values: [MAX_RANK]isize = [_]isize{0} ** MAX_RANK,
    ndim: u8 = 0,

    /// Borrow strides slice.
    pub fn slice(self: *const Strides) []const isize {
        return self.values[0..self.ndim];
    }

    /// Compute canonical contiguous strides for a given shape and memory layout order.
    pub fn fromShape(shape: Shape, order: Order) Strides {
        var s = Strides{ .ndim = shape.ndim };
        if (shape.ndim == 0) return s;

        switch (order) {
            .c => {
                var stride: isize = 1;
                var i: usize = shape.ndim;
                while (i > 0) {
                    i -= 1;
                    s.values[i] = stride;
                    stride *= @as(isize, @intCast(shape.dims[i]));
                }
            },
            .fortran => {
                var stride: isize = 1;
                for (0..shape.ndim) |i| {
                    s.values[i] = stride;
                    stride *= @as(isize, @intCast(shape.dims[i]));
                }
            },
        }
        return s;
    }

    /// Determines whether the strides represent a contiguous C-order (row-major) layout.
    pub fn isCContiguous(shape: Shape, strides: Strides) bool {
        if (shape.ndim == 0) return true;
        var expected: isize = 1;
        var i: usize = shape.ndim;
        while (i > 0) {
            i -= 1;
            const d = shape.dims[i];
            if (d == 0) return true;
            if (d > 1) {
                if (strides.values[i] != expected) return false;
                expected *= @as(isize, @intCast(d));
            }
        }
        return true;
    }

    /// Determines whether the strides represent a contiguous Fortran-order (column-major) layout.
    pub fn isFContiguous(shape: Shape, strides: Strides) bool {
        if (shape.ndim == 0) return true;
        var expected: isize = 1;
        for (0..shape.ndim) |i| {
            const d = shape.dims[i];
            if (d == 0) return true;
            if (d > 1) {
                if (strides.values[i] != expected) return false;
                expected *= @as(isize, @intCast(d));
            }
        }
        return true;
    }
};

/// Computes the broadcasted output shape resulting from combining shapes `a` and `b`.
pub fn broadcastShapes(a: Shape, b: Shape) ShapeError!Shape {
    const max_rank = @max(a.ndim, b.ndim);
    if (max_rank > MAX_RANK) return ShapeError.RankExceeded;

    var out = Shape{ .ndim = max_rank };

    var i: usize = 0;
    while (i < max_rank) : (i += 1) {
        const dim_a: usize = if (i < a.ndim) a.dims[a.ndim - 1 - i] else 1;
        const dim_b: usize = if (i < b.ndim) b.dims[b.ndim - 1 - i] else 1;

        if (dim_a == dim_b) {
            out.dims[max_rank - 1 - i] = dim_a;
        } else if (dim_a == 1) {
            out.dims[max_rank - 1 - i] = dim_b;
        } else if (dim_b == 1) {
            out.dims[max_rank - 1 - i] = dim_a;
        } else {
            return ShapeError.BroadcastError;
        }
    }

    return out;
}

/// Adapts the strides of an array to a broadcasted target shape.
/// Any dimension where the source has size 1 is assigned stride 0.
pub fn broadcastStrides(src_shape: Shape, src_strides: Strides, target_shape: Shape) ShapeError!Strides {
    if (src_shape.ndim > target_shape.ndim) return ShapeError.IncompatibleShapes;

    var out = Strides{ .ndim = target_shape.ndim };
    const lead = target_shape.ndim - src_shape.ndim;

    // Leading dimensions prepended to match rank get stride 0
    for (0..lead) |i| {
        out.values[i] = 0;
    }

    // Trailing dimensions
    for (0..src_shape.ndim) |i| {
        const src_dim = src_shape.dims[i];
        const target_dim = target_shape.dims[lead + i];

        if (src_dim == target_dim) {
            out.values[lead + i] = if (src_dim == 1) 0 else src_strides.values[i];
        } else if (src_dim == 1) {
            out.values[lead + i] = 0;
        } else {
            return ShapeError.BroadcastError;
        }
    }

    return out;
}

test "shape initialization and element count" {
    const s0 = Shape.scalar();
    try std.testing.expectEqual(@as(usize, 0), s0.ndim);
    try std.testing.expectEqual(@as(usize, 1), s0.elementCount());

    const s1 = try Shape.init(&.{ 3, 4, 5 });
    try std.testing.expectEqual(@as(usize, 3), s1.ndim);
    try std.testing.expectEqual(@as(usize, 60), s1.elementCount());
    try std.testing.expectEqual(@as(usize, 240), s1.byteCount(.f32));
}

test "strides c-contiguous and fortran-contiguous" {
    const s = try Shape.init(&.{ 2, 3, 4 });
    const sc = Strides.fromShape(s, .c);
    try std.testing.expectEqualSlices(isize, &.{ 12, 4, 1 }, sc.slice());
    try std.testing.expect(Strides.isCContiguous(s, sc));
    try std.testing.expect(!Strides.isFContiguous(s, sc));

    const sf = Strides.fromShape(s, .fortran);
    try std.testing.expectEqualSlices(isize, &.{ 1, 2, 6 }, sf.slice());
    try std.testing.expect(Strides.isFContiguous(s, sf));
    try std.testing.expect(!Strides.isCContiguous(s, sf));
}

test "broadcast shapes" {
    const a = try Shape.init(&.{ 2, 1, 3 });
    const b = try Shape.init(&.{ 5, 3 });
    const out = try broadcastShapes(a, b);
    try std.testing.expectEqualSlices(usize, &.{ 2, 5, 3 }, out.slice());

    // Incompatible
    const c = try Shape.init(&.{ 2, 4 });
    const d = try Shape.init(&.{ 3, 4 });
    try std.testing.expectError(ShapeError.BroadcastError, broadcastShapes(c, d));
}

test "broadcast strides" {
    const src_s = try Shape.init(&.{ 1, 3 });
    const src_st = Strides.fromShape(src_s, .c);
    const target_s = try Shape.init(&.{ 4, 5, 3 });
    const out_st = try broadcastStrides(src_s, src_st, target_s);

    try std.testing.expectEqualSlices(isize, &.{ 0, 0, 1 }, out_st.slice());
}

test "slice resolution" {
    const s1 = Slice{ .start = 1, .stop = 5, .step = 2 };
    const r1 = try s1.resolve(10);
    try std.testing.expectEqual(@as(usize, 1), r1.start);
    try std.testing.expectEqual(@as(usize, 2), r1.len);
    try std.testing.expectEqual(@as(isize, 2), r1.step);

    const s2 = Slice{ .start = null, .stop = null, .step = -1 };
    const r2 = try s2.resolve(5);
    try std.testing.expectEqual(@as(usize, 4), r2.start);
    try std.testing.expectEqual(@as(usize, 5), r2.len);
    try std.testing.expectEqual(@as(isize, -1), r2.step);
}
