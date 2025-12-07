const std = @import("std");
const core = @import("core.zig");
const NDArray = core.NDArray;
const Allocator = std.mem.Allocator;

/// Represents a slice operation on a dimension.
///
/// Can be:
/// - `all`: Select all elements along the dimension (like `:` in Python).
/// - `index`: Select a specific index (reduces rank by 1).
/// - `range`: Select a range of indices (start, end, step).
pub const Slice = union(enum) {
    all: void,
    index: usize,
    range: struct { start: usize, end: usize, step: isize = 1 },
};

/// Creates a view of the array using a sequence of slice operations.
///
/// This function allows for advanced indexing similar to NumPy's slicing.
/// It calculates the new shape and strides based on the provided slices.
///
/// Arguments:
///     allocator: The allocator to use for the new shape and strides arrays.
///     T: The data type of the array elements.
///     arr: The input NDArray.
///     slices: A slice of `Slice` objects, one for each dimension or fewer.
///
/// Returns:
///     A new NDArray representing the view. The data pointer is shared with the original array.
///     The caller owns the new NDArray structure (shape and strides), but not the data.
///
/// Example:
/// ```zig
/// var a = try NDArray(f32).arange(allocator, 0, 10, 1);
/// defer a.deinit();
/// // a is {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
///
/// const slices = &[_]indexing.Slice{
///     .{ .range = .{ .start = 2, .end = 8, .step = 2 } },
/// };
/// var view = try indexing.slice(allocator, f32, a, slices);
/// defer view.deinit();
/// // view is {2, 4, 6}
/// ```
pub fn slice(allocator: Allocator, comptime T: type, arr: NDArray(T), slices: []const Slice) !NDArray(T) {
    if (slices.len > arr.rank()) return core.Error.RankMismatch;

    var new_rank: usize = 0;
    for (slices) |s| {
        switch (s) {
            .all, .range => new_rank += 1,
            .index => {},
        }
    }
    new_rank += (arr.rank() - slices.len);

    const new_shape = try allocator.alloc(usize, new_rank);
    errdefer allocator.free(new_shape);

    const new_strides = try allocator.alloc(usize, new_rank);
    errdefer allocator.free(new_strides);

    var data_offset: usize = 0;
    var dim_idx: usize = 0;

    for (slices, 0..) |s, i| {
        const dim_size = arr.shape[i];
        const dim_stride = arr.strides[i];

        switch (s) {
            .all => {
                new_shape[dim_idx] = dim_size;
                new_strides[dim_idx] = dim_stride;
                dim_idx += 1;
            },
            .index => |idx| {
                if (idx >= dim_size) return core.Error.IndexOutOfBounds;
                data_offset += idx * dim_stride;
            },
            .range => |r| {
                const start = r.start;
                const end = @min(r.end, dim_size);
                const step = r.step;

                if (start >= dim_size) return core.Error.IndexOutOfBounds;

                // Calculate new size
                var count: usize = 0;
                if (step > 0) {
                    if (start < end) {
                        count = (end - start + @as(usize, @intCast(step)) - 1) / @as(usize, @intCast(step));
                    }
                } else {
                    return core.Error.UnsupportedType;
                }

                new_shape[dim_idx] = count;
                new_strides[dim_idx] = dim_stride * @as(usize, @intCast(step));
                data_offset += start * dim_stride;
                dim_idx += 1;
            },
        }
    }

    // Copy remaining dimensions
    for (slices.len..arr.rank()) |i| {
        new_shape[dim_idx] = arr.shape[i];
        new_strides[dim_idx] = arr.strides[i];
        dim_idx += 1;
    }

    return NDArray(T){
        .allocator = allocator,
        .data = arr.data[data_offset..],
        .shape = new_shape,
        .strides = new_strides,
        .owns_data = false,
    };
}

/// Return elements chosen from x or y depending on condition.
///
/// Arguments:
///     allocator: The allocator to use for the result array.
///     T: The data type of the array elements.
///     condition: The boolean condition array.
///     x: The values to use where condition is True.
///     y: The values to use where condition is False.
///
/// Returns:
///     A new NDArray with elements from x or y.
///
/// Example:
/// ```zig
/// var cond = try NDArray(bool).init(allocator, &.{2}, &.{true, false});
/// defer cond.deinit();
/// var x = try NDArray(f32).init(allocator, &.{2}, &.{1.0, 2.0});
/// defer x.deinit();
/// var y = try NDArray(f32).init(allocator, &.{2}, &.{3.0, 4.0});
/// defer y.deinit();
///
/// var result = try indexing.where(allocator, f32, cond, x, y);
/// defer result.deinit();
/// // result is {1.0, 4.0}
/// ```
pub fn where(allocator: Allocator, comptime T: type, condition: NDArray(bool), x: NDArray(T), y: NDArray(T)) !NDArray(T) {
    // Broadcast all three to common shape
    const shape1 = try core.broadcastShape(allocator, condition.shape, x.shape);
    defer allocator.free(shape1);
    const final_shape = try core.broadcastShape(allocator, shape1, y.shape);
    defer allocator.free(final_shape);

    var cond_b = try condition.broadcastTo(final_shape);
    defer cond_b.deinit();
    var x_b = try x.broadcastTo(final_shape);
    defer x_b.deinit();
    var y_b = try y.broadcastTo(final_shape);
    defer y_b.deinit();

    var result = try NDArray(T).init(allocator, final_shape);
    errdefer result.deinit();

    var iter = try core.NdIterator.init(allocator, final_shape);
    defer iter.deinit();

    var i: usize = 0;
    while (iter.next()) |coords| {
        const c = try cond_b.get(coords);
        const val_x = try x_b.get(coords);
        const val_y = try y_b.get(coords);

        result.data[i] = if (c) val_x else val_y;
        i += 1;
    }

    return result;
}

/// Select elements from an array using a boolean mask.
/// The mask must have the same shape as the array.
/// Returns a 1D array containing the selected elements.
pub fn booleanMask(allocator: Allocator, comptime T: type, arr: NDArray(T), mask: NDArray(bool)) !NDArray(T) {
    if (!std.mem.eql(usize, arr.shape, mask.shape)) return core.Error.ShapeMismatch;

    var count: usize = 0;
    // We need to iterate to count true values.
    // If contiguous, we can iterate linearly.
    if (mask.flags().c_contiguous) {
        for (mask.data) |b| {
            if (b) count += 1;
        }
    } else {
        var iter = try core.NdIterator.init(allocator, mask.shape);
        defer iter.deinit();
        while (iter.next()) |coords| {
            if (try mask.get(coords)) count += 1;
        }
    }

    var result = try NDArray(T).init(allocator, &.{count});

    var idx: usize = 0;

    if (arr.flags().c_contiguous and mask.flags().c_contiguous) {
        for (mask.data, 0..) |b, i| {
            if (b) {
                result.data[idx] = arr.data[i];
                idx += 1;
            }
        }
    } else {
        // Use iterator
        var iter = try core.NdIterator.init(allocator, arr.shape);
        defer iter.deinit();

        while (iter.next()) |coords| {
            const mask_val = try mask.get(coords);
            if (mask_val) {
                const val = try arr.get(coords);
                result.data[idx] = val;
                idx += 1;
            }
        }
    }

    return result;
}

/// Take elements from an array along an axis.
///
/// Arguments:
///     allocator: Allocator.
///     T: Element type.
///     arr: Input array.
///     indices: Array of indices (must be 1D for now).
///     axis: Axis to select along.
///
/// Returns:
///     New array with elements taken.
pub fn take(allocator: Allocator, comptime T: type, arr: NDArray(T), indices: NDArray(usize), axis: usize) !NDArray(T) {
    if (axis >= arr.rank()) return core.Error.IndexOutOfBounds;
    if (indices.rank() != 1) return core.Error.DimensionMismatch; // Support 1D indices for now

    const num_indices = indices.shape[0];

    // Calculate new shape
    var new_shape = try allocator.alloc(usize, arr.rank());
    defer allocator.free(new_shape);
    @memcpy(new_shape, arr.shape);
    new_shape[axis] = num_indices;

    var result = try NDArray(T).init(allocator, new_shape);
    errdefer result.deinit();

    var src_coords = try allocator.alloc(usize, arr.rank());
    defer allocator.free(src_coords);

    var iter = try core.NdIterator.init(allocator, result.shape);
    defer iter.deinit();

    while (iter.next()) |coords| {
        @memcpy(src_coords, coords);
        const idx_in_indices = coords[axis];
        const src_idx = try indices.get(&.{idx_in_indices});

        if (src_idx >= arr.shape[axis]) return core.Error.IndexOutOfBounds;

        src_coords[axis] = src_idx;

        const val = try arr.get(src_coords);
        try result.set(coords, val);
    }

    return result;
}
