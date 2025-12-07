const std = @import("std");
const core = @import("core.zig");
const NDArray = core.NDArray;
const Allocator = std.mem.Allocator;

/// Stack arrays in sequence vertically (row wise).
///
/// Arguments:
///     allocator: The allocator to use for the result array.
///     T: The data type of the array elements.
///     arrays: A slice of NDArray(T) to stack.
///
/// Returns:
///     A new NDArray with the arrays stacked vertically.
pub fn vstack(allocator: Allocator, comptime T: type, arrays: []const NDArray(T)) !NDArray(T) {
    if (arrays.len == 0) return core.Error.DimensionMismatch;

    const first = arrays[0];
    if (first.rank() == 1) {
        // Promote all to 2D (1, N) then concatenate along axis 0
        var promoted = try allocator.alloc(NDArray(T), arrays.len);
        defer allocator.free(promoted);

        var initialized_count: usize = 0;
        defer {
            for (0..initialized_count) |k| {
                promoted[k].deinit();
            }
        }

        for (arrays, 0..) |arr, i| {
            promoted[i] = try arr.expandDims(0);
            initialized_count += 1;
        }

        return NDArray(T).concatenate(allocator, promoted, 0);
    } else {
        return NDArray(T).concatenate(allocator, arrays, 0);
    }
}

/// Stack arrays in sequence horizontally (column wise).
///
/// Arguments:
///     allocator: The allocator to use for the result array.
///     T: The data type of the array elements.
///     arrays: A slice of NDArray(T) to stack.
///
/// Returns:
///     A new NDArray with the arrays stacked horizontally.
pub fn hstack(allocator: Allocator, comptime T: type, arrays: []const NDArray(T)) !NDArray(T) {
    if (arrays.len == 0) return core.Error.DimensionMismatch;
    const first = arrays[0];

    if (first.rank() == 1) {
        return NDArray(T).concatenate(allocator, arrays, 0);
    } else {
        return NDArray(T).concatenate(allocator, arrays, 1);
    }
}

/// Stack arrays in sequence depth wise (along third axis).
///
/// Arguments:
///     allocator: The allocator to use for the result array.
///     T: The data type of the array elements.
///     arrays: A slice of NDArray(T) to stack.
///
/// Returns:
///     A new NDArray with the arrays stacked depth-wise.
pub fn dstack(allocator: Allocator, comptime T: type, arrays: []const NDArray(T)) !NDArray(T) {
    if (arrays.len == 0) return core.Error.DimensionMismatch;

    var promoted = try allocator.alloc(NDArray(T), arrays.len);
    defer allocator.free(promoted);

    var initialized_count: usize = 0;
    defer {
        for (0..initialized_count) |k| {
            promoted[k].deinit();
        }
    }

    for (arrays, 0..) |arr, i| {
        if (arr.rank() == 1) {
            // (N,) -> (1, N, 1)
            var tmp = try arr.expandDims(0); // (1, N)
            promoted[i] = try tmp.expandDims(2); // (1, N, 1)
            tmp.deinit();
        } else if (arr.rank() == 2) {
            // (M, N) -> (M, N, 1)
            promoted[i] = try arr.expandDims(2);
        } else {
            promoted[i] = try arr.copy();
        }
        initialized_count += 1;
    }

    return NDArray(T).concatenate(allocator, promoted, 2);
}

/// Construct an array by repeating A the number of times given by reps.
///
/// Arguments:
///     allocator: The allocator to use for the result array.
///     T: The data type of the array elements.
///     arr: The input array.
///     reps: The number of repetitions along each axis.
///
/// Returns:
///     A new NDArray with the repeated elements.
pub fn tile(allocator: Allocator, comptime T: type, arr: NDArray(T), reps: []const usize) !NDArray(T) {
    const max_rank = @max(arr.rank(), reps.len);

    // Calculate new shape
    var new_shape = try allocator.alloc(usize, max_rank);
    defer allocator.free(new_shape);

    // Align shapes
    var arr_shape_aligned = try allocator.alloc(usize, max_rank);
    defer allocator.free(arr_shape_aligned);

    var reps_aligned = try allocator.alloc(usize, max_rank);
    defer allocator.free(reps_aligned);

    const offset_arr = max_rank - arr.rank();
    const offset_reps = max_rank - reps.len;

    for (0..max_rank) |i| {
        if (i < offset_arr) arr_shape_aligned[i] = 1 else arr_shape_aligned[i] = arr.shape[i - offset_arr];
        if (i < offset_reps) reps_aligned[i] = 1 else reps_aligned[i] = reps[i - offset_reps];

        new_shape[i] = arr_shape_aligned[i] * reps_aligned[i];
    }

    var result = try NDArray(T).init(allocator, new_shape);

    var iter = try core.NdIterator.init(allocator, new_shape);
    defer iter.deinit();

    var source_coords = try allocator.alloc(usize, arr.rank());
    defer allocator.free(source_coords);

    var i: usize = 0;
    while (iter.next()) |coords| {
        // Map coords to source coords
        for (0..arr.rank()) |d| {
            const aligned_idx = d + offset_arr;
            source_coords[d] = coords[aligned_idx] % arr.shape[d];
        }

        const val = try arr.get(source_coords);
        result.data[i] = val;
        i += 1;
    }

    return result;
}

/// Repeat elements of an array.
///
/// Arguments:
///     allocator: The allocator to use for the result array.
///     T: The data type of the array elements.
///     arr: The input array.
///     repeats: The number of repetitions.
///     axis: The axis along which to repeat values. If null, the array is flattened.
///
/// Returns:
///     A new NDArray with the repeated elements.
pub fn repeat(allocator: Allocator, comptime T: type, arr: NDArray(T), repeats: usize, axis: ?usize) !NDArray(T) {
    if (axis) |ax| {
        if (ax >= arr.rank()) return core.Error.IndexOutOfBounds;

        var new_shape = try allocator.alloc(usize, arr.rank());
        defer allocator.free(new_shape);
        @memcpy(new_shape, arr.shape);
        new_shape[ax] *= repeats;

        var result = try NDArray(T).init(allocator, new_shape);

        var iter = try core.NdIterator.init(allocator, result.shape);
        defer iter.deinit();

        var source_coords = try allocator.alloc(usize, arr.rank());
        defer allocator.free(source_coords);

        var i: usize = 0;
        while (iter.next()) |coords| {
            @memcpy(source_coords, coords);
            source_coords[ax] = coords[ax] / repeats;

            result.data[i] = try arr.get(source_coords);
            i += 1;
        }
        return result;
    } else {
        // Flatten and repeat
        const flat_size = arr.size() * repeats;
        var result = try NDArray(T).init(allocator, &.{flat_size});

        var iter = try core.NdIterator.init(allocator, arr.shape);
        defer iter.deinit();

        var idx: usize = 0;
        while (iter.next()) |coords| {
            const val = try arr.get(coords);
            for (0..repeats) |_| {
                result.data[idx] = val;
                idx += 1;
            }
        }
        return result;
    }
}

/// Move axes of an array to new positions.
///
/// Arguments:
///     allocator: The allocator to use for the result array.
///     T: The data type of the array elements.
///     arr: The input array.
///     source: The original positions of the axes to move.
///     destination: The destination positions for each of the original axes.
///
/// Returns:
///     A new NDArray with the axes moved.
pub fn moveaxis(allocator: Allocator, comptime T: type, arr: NDArray(T), source: []const usize, destination: []const usize) !NDArray(T) {
    if (source.len != destination.len) return core.Error.DimensionMismatch;

    var temp_order = std.ArrayList(usize).init(allocator);
    defer temp_order.deinit();

    // Add indices not in source
    for (0..arr.rank()) |i| {
        var in_source = false;
        for (source) |s| {
            if (s == i) {
                in_source = true;
                break;
            }
        }
        if (!in_source) {
            try temp_order.append(i);
        }
    }

    var final_order = try allocator.alloc(usize, arr.rank());
    defer allocator.free(final_order);
    @memset(final_order, 9999); // Sentinel

    for (source, destination) |s, d| {
        final_order[d] = s;
    }

    var temp_idx: usize = 0;
    for (final_order) |*val| {
        if (val.* == 9999) {
            val.* = temp_order.items[temp_idx];
            temp_idx += 1;
        }
    }

    return arr.permute(final_order);
}

/// Return a contiguous flattened array.
///
/// Arguments:
///     allocator: The allocator to use for the result array.
///     T: The data type of the array elements.
///     arr: The input array.
///
/// Returns:
///     A new 1D NDArray containing the flattened data.
pub fn ravel(allocator: Allocator, comptime T: type, arr: NDArray(T)) !NDArray(T) {
    _ = allocator;
    if (arr.flags().c_contiguous) {
        return arr.reshape(&.{arr.size()});
    } else {
        return arr.flatten();
    }
}

/// Reverse the order of elements in an array along the given axis.
///
/// Arguments:
///     allocator: The allocator to use for the result array.
///     T: The data type of the array elements.
///     arr: The input array.
///     axis: The axis along which to flip.
///
/// Returns:
///     A new NDArray with the elements flipped.
pub fn flip(allocator: Allocator, comptime T: type, arr: NDArray(T), axis: usize) !NDArray(T) {
    if (axis >= arr.rank()) return core.Error.IndexOutOfBounds;

    var result = try NDArray(T).init(allocator, arr.shape);

    var iter = try core.NdIterator.init(allocator, result.shape);
    defer iter.deinit();

    var source_coords = try allocator.alloc(usize, arr.rank());
    defer allocator.free(source_coords);

    var i: usize = 0;
    while (iter.next()) |coords| {
        @memcpy(source_coords, coords);
        source_coords[axis] = arr.shape[axis] - 1 - coords[axis];

        result.data[i] = try arr.get(source_coords);
        i += 1;
    }

    return result;
}

/// Roll array elements along a given axis.
///
/// Arguments:
///     allocator: The allocator to use for the result array.
///     T: The data type of the array elements.
///     arr: The input array.
///     shift: The number of places by which elements are shifted.
///     axis: The axis along which elements are shifted.
///
/// Returns:
///     A new NDArray with the elements rolled.
pub fn roll(allocator: Allocator, comptime T: type, arr: NDArray(T), shift: isize, axis: usize) !NDArray(T) {
    if (axis >= arr.rank()) return core.Error.IndexOutOfBounds;

    var result = try NDArray(T).init(allocator, arr.shape);
    const dim_size = arr.shape[axis];

    // Normalize shift
    var s = @mod(shift, @as(isize, @intCast(dim_size)));
    if (s < 0) s += @as(isize, @intCast(dim_size));
    const u_shift = @as(usize, @intCast(s));

    var iter = try core.NdIterator.init(allocator, result.shape);
    defer iter.deinit();

    var source_coords = try allocator.alloc(usize, arr.rank());
    defer allocator.free(source_coords);

    var i: usize = 0;
    while (iter.next()) |coords| {
        @memcpy(source_coords, coords);
        // source_idx = (idx + size - shift) % size
        source_coords[axis] = (coords[axis] + dim_size - u_shift) % dim_size;

        result.data[i] = try arr.get(source_coords);
        i += 1;
    }

    return result;
}

/// Interchange two axes of an array.
///
/// Arguments:
///     allocator: The allocator to use for the result array.
///     T: The data type of the array elements.
///     arr: The input array.
///     axis1: First axis.
///     axis2: Second axis.
///
/// Returns:
///     A view of the array with axes swapped.
pub fn swapaxes(allocator: Allocator, comptime T: type, arr: NDArray(T), axis1: usize, axis2: usize) !NDArray(T) {
    if (axis1 >= arr.rank() or axis2 >= arr.rank()) return core.Error.IndexOutOfBounds;
    if (axis1 == axis2) return arr.copy(); // Or view? Permute returns view.

    var perm = try allocator.alloc(usize, arr.rank());
    defer allocator.free(perm);
    for (0..arr.rank()) |i| perm[i] = i;

    perm[axis1] = axis2;
    perm[axis2] = axis1;

    return arr.permute(perm);
}

/// Find indices where a predicate is true.
///
/// Logic: indices = where(predicate(a))
///
/// Arguments:
///     allocator: Allocator.
///     T: Type.
///     a: Input array.
///     predicate: Function taking T returning bool.
///
/// Returns:
///     NDArray(usize) of flattened indices.
///
/// Example:
/// ```zig
/// fn isPositive(val: f32) bool { return val > 0; }
/// var indices = try manipulation.find(allocator, f32, &a, isPositive);
/// defer indices.deinit();
/// ```
pub fn find(allocator: Allocator, comptime T: type, a: *const NDArray(T), predicate: anytype) !NDArray(usize) {
    var count: usize = 0;
    if (a.flags().c_contiguous) {
        for (a.data) |val| {
            if (predicate(val)) count += 1;
        }
    } else {
        var iter = try core.NdIterator.init(allocator, a.shape);
        defer iter.deinit();
        while (iter.next()) |coords| {
            const val = try a.get(coords);
            if (predicate(val)) count += 1;
        }
    }

    var result = try NDArray(usize).init(allocator, &.{count});
    var idx: usize = 0;
    var current: usize = 0;

    if (a.flags().c_contiguous) {
        for (a.data) |val| {
            if (predicate(val)) {
                result.data[idx] = current;
                idx += 1;
            }
            current += 1;
        }
    } else {
        var iter = try core.NdIterator.init(allocator, a.shape);
        defer iter.deinit();
        while (iter.next()) |coords| {
            const val = try a.get(coords);
            if (predicate(val)) {
                result.data[idx] = current;
                idx += 1;
            }
            current += 1;
        }
    }
    return result;
}

/// Replace elements where a predicate is true with a new value.
///
/// Logic: result = a.copy(); result[predicate(a)] = value
///
/// Arguments:
///     allocator: Allocator.
///     T: Type.
///     a: Input array.
///     predicate: Function taking T returning bool.
///     value: New value.
///
/// Returns:
///     New NDArray(T).
///
/// Example:
/// ```zig
/// var res = try manipulation.replaceWhere(allocator, f32, &a, isNegative, 0.0);
/// defer res.deinit();
/// ```
pub fn replaceWhere(allocator: Allocator, comptime T: type, a: *const NDArray(T), predicate: anytype, value: T) !NDArray(T) {
    _ = allocator;
    const result = try a.copy();
    // Note: copy() returns a contiguous array, so we can iterate data directly
    for (result.data) |*val| {
        if (predicate(val.*)) {
            val.* = value;
        }
    }
    return result;
}

/// Replace all occurrences of a value with a new value.
///
/// Logic: result = a.copy(); result[a == old] = new
///
/// Arguments:
///     allocator: Allocator.
///     T: Type.
///     a: Input array.
///     old_val: Value to replace.
///     new_val: New value.
///
/// Returns:
///     New NDArray(T).
///
/// Example:
/// ```zig
/// var res = try manipulation.replace(allocator, f32, &a, -1.0, 0.0);
/// defer res.deinit();
/// ```
pub fn replace(allocator: Allocator, comptime T: type, a: *const NDArray(T), old_val: T, new_val: T) !NDArray(T) {
    _ = allocator;
    const result = try a.copy();
    for (result.data) |*val| {
        if (val.* == old_val) {
            val.* = new_val;
        }
    }
    return result;
}

/// Replace the first occurrence of a value with a new value.
///
/// Logic: result = a.copy(); result[first(a == old)] = new
///
/// Arguments:
///     allocator: Allocator.
///     T: Type.
///     a: Input array.
///     old_val: Value to replace.
///     new_val: New value.
///
/// Returns:
///     New NDArray(T).
pub fn replaceFirst(allocator: Allocator, comptime T: type, a: *const NDArray(T), old_val: T, new_val: T) !NDArray(T) {
    _ = allocator;
    const result = try a.copy();
    for (result.data) |*val| {
        if (val.* == old_val) {
            val.* = new_val;
            break;
        }
    }
    return result;
}

/// Replace the last occurrence of a value with a new value.
///
/// Logic: result = a.copy(); result[last(a == old)] = new
///
/// Arguments:
///     allocator: Allocator.
///     T: Type.
///     a: Input array.
///     old_val: Value to replace.
///     new_val: New value.
///
/// Returns:
///     New NDArray(T).
pub fn replaceLast(allocator: Allocator, comptime T: type, a: *const NDArray(T), old_val: T, new_val: T) !NDArray(T) {
    _ = allocator;
    const result = try a.copy();
    var i: usize = result.data.len;
    while (i > 0) {
        i -= 1;
        if (result.data[i] == old_val) {
            result.data[i] = new_val;
            break;
        }
    }
    return result;
}

/// Delete elements at specified indices (flattened).
///
/// Logic: result = a without indices
///
/// Arguments:
///     allocator: Allocator.
///     T: Type.
///     a: Input array.
///     indices: Array of indices to remove.
///
/// Returns:
///     New NDArray(T) (1D).
///
/// Example:
/// ```zig
/// var res = try manipulation.delete(allocator, f32, &a, &.{0, 2});
/// defer res.deinit();
/// ```
pub fn delete(allocator: Allocator, comptime T: type, a: *const NDArray(T), indices: []const usize) !NDArray(T) {
    const total_len = a.size();
    if (indices.len > total_len) return core.Error.IndexOutOfBounds;

    const new_len = total_len - indices.len;
    var result = try NDArray(T).init(allocator, &.{new_len});

    var keep = try allocator.alloc(bool, total_len);
    defer allocator.free(keep);
    @memset(keep, true);

    for (indices) |idx| {
        if (idx < total_len) keep[idx] = false;
    }

    var out_idx: usize = 0;
    if (a.flags().c_contiguous) {
        for (a.data, 0..) |val, i| {
            if (keep[i]) {
                result.data[out_idx] = val;
                out_idx += 1;
            }
        }
    } else {
        var iter = try core.NdIterator.init(allocator, a.shape);
        defer iter.deinit();
        var i: usize = 0;
        while (iter.next()) |coords| {
            const val = try a.get(coords);
            if (keep[i]) {
                result.data[out_idx] = val;
                out_idx += 1;
            }
            i += 1;
        }
    }
    return result;
}
