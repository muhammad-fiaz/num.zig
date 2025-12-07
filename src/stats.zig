const std = @import("std");
const core = @import("core.zig");
const NDArray = core.NDArray;
const Allocator = std.mem.Allocator;

/// Computes the sum of all elements in the array.
///
/// Logic:
/// result = sum(a)
///
/// Arguments:
///     T: The data type of the array elements.
///     a: The input array.
///
/// Returns:
///     The sum of all elements.
///
/// Example:
/// ```zig
/// const s = try stats.sum(f32, &a);
/// ```
pub fn sum(comptime T: type, a: *const NDArray(T)) !T {
    return a.sum();
}

/// Computes the product of all elements in the array.
///
/// Logic:
/// result = prod(a)
///
/// Arguments:
///     T: The data type of the array elements.
///     a: The input array.
///
/// Returns:
///     The product of all elements.
///
/// Example:
/// ```zig
/// const p = try stats.prod(f32, &a);
/// ```
pub fn prod(comptime T: type, a: *const NDArray(T)) !T {
    var p: T = 1;
    var iter = try core.NdIterator.init(a.allocator, a.shape);
    defer iter.deinit();
    while (iter.next()) |coords| {
        p *= try a.get(coords);
    }
    return p;
}

/// Computes the minimum value in the array.
///
/// Logic:
/// result = min(a)
///
/// Arguments:
///     T: The data type of the array elements.
///     a: The input array.
///
/// Returns:
///     The minimum value.
///
/// Example:
/// ```zig
/// const m = try stats.min(f32, &a);
/// ```
pub fn min(comptime T: type, a: *const NDArray(T)) !T {
    return a.min();
}

/// Computes the maximum value in the array.
///
/// Logic:
/// result = max(a)
///
/// Arguments:
///     T: The data type of the array elements.
///     a: The input array.
///
/// Returns:
///     The maximum value.
///
/// Example:
/// ```zig
/// const m = try stats.max(f32, &a);
/// ```
pub fn max(comptime T: type, a: *const NDArray(T)) !T {
    return a.max();
}

/// Computes the mean of all elements in the array.
///
/// Logic:
/// result = mean(a)
///
/// Arguments:
///     T: The data type of the array elements.
///     a: The input array.
///
/// Returns:
///     The mean value.
///
/// Example:
/// ```zig
/// const m = try stats.mean(f32, &a);
/// ```
pub fn mean(comptime T: type, a: *const NDArray(T)) !T {
    return a.mean();
}

/// Computes the variance.
///
/// Logic:
/// result = mean((a - mean(a))^2)
///
/// Arguments:
///     T: The data type of the array elements.
///     a: The input array.
///
/// Returns:
///     The variance.
///
/// Example:
/// ```zig
/// const v = try stats.var_val(f32, &a);
/// ```
pub fn var_val(comptime T: type, a: *const NDArray(T)) !T {
    if (a.size() == 0) return 0;
    const m = try mean(T, a);
    var ss: T = 0;
    var iter = try core.NdIterator.init(a.allocator, a.shape);
    defer iter.deinit();
    while (iter.next()) |coords| {
        const val = try a.get(coords);
        const diff = val - m;
        ss += diff * diff;
    }
    return ss / @as(T, @floatFromInt(a.size()));
}

/// Computes the standard deviation.
///
/// Logic:
/// result = sqrt(var(a))
///
/// Arguments:
///     T: The data type of the array elements.
///     a: The input array.
///
/// Returns:
///     The standard deviation.
///
/// Example:
/// ```zig
/// const s = try stats.std_val(f32, &a);
/// ```
pub fn std_val(comptime T: type, a: *const NDArray(T)) !T {
    return @sqrt(try var_val(T, a));
}

/// Computes the median.
/// Note: This allocates memory to sort a copy of the data.
///
/// Logic:
/// result = middle value of sorted a
///
/// Arguments:
///     allocator: The allocator to use for sorting.
///     T: The data type of the array elements.
///     a: The input array.
///
/// Returns:
///     The median value.
///
/// Example:
/// ```zig
/// const m = try stats.median(allocator, f32, &a);
/// ```
pub fn median(comptime T: type, a: *const NDArray(T)) !T {
    if (a.size() == 0) return 0;
    var sorted = try a.flatten();
    defer sorted.deinit();

    std.mem.sort(T, sorted.data, {}, std.sort.asc(T));

    const n = sorted.size();
    if (n % 2 == 1) {
        return sorted.data[n / 2];
    } else {
        return (sorted.data[n / 2 - 1] + sorted.data[n / 2]) / 2.0;
    }
}

/// Returns the linear index of the minimum value in the array.
///
/// Arguments:
///     T: The data type of the array elements.
///     a: The input array.
///
/// Returns:
///     The linear index of the minimum value.
///
/// Example:
/// ```zig
/// const idx = try stats.argmin(f32, &a);
/// ```
pub fn argmin(comptime T: type, a: *const NDArray(T)) !usize {
    if (a.size() == 0) return 0;
    var m: T = 0;
    var min_idx: usize = 0;
    var current_idx: usize = 0;

    var iter = try core.NdIterator.init(a.allocator, a.shape);
    defer iter.deinit();

    if (iter.next()) |coords| {
        m = try a.get(coords);
    }
    iter.reset();

    while (iter.next()) |coords| {
        const val = try a.get(coords);
        if (val < m) {
            m = val;
            min_idx = current_idx;
        }
        current_idx += 1;
    }
    return min_idx;
}

/// Returns the linear index of the maximum value in the array.
///
/// Arguments:
///     T: The data type of the array elements.
///     a: The input array.
///
/// Returns:
///     The linear index of the maximum value.
///
/// Example:
/// ```zig
/// const idx = try stats.argmax(f32, &a);
/// ```
pub fn argmax(comptime T: type, a: *const NDArray(T)) !usize {
    if (a.size() == 0) return 0;
    var m: T = 0;
    var max_idx: usize = 0;
    var current_idx: usize = 0;

    var iter = try core.NdIterator.init(a.allocator, a.shape);
    defer iter.deinit();

    if (iter.next()) |coords| {
        m = try a.get(coords);
    }
    iter.reset();

    while (iter.next()) |coords| {
        const val = try a.get(coords);
        if (val > m) {
            m = val;
            max_idx = current_idx;
        }
        current_idx += 1;
    }
    return max_idx;
}

/// Computes the sum of elements along a given axis.
///
/// Logic:
/// result = sum(a, axis)
///
/// Arguments:
///     T: The data type of the array elements.
///     allocator: The allocator to use for the result array.
///     a: The input array.
///     axis: The axis along which to sum.
///     keepdims: Whether to keep the reduced dimension (as size 1).
///
/// Returns:
///     A new NDArray containing the sums.
///
/// Example:
/// ```zig
/// var s = try stats.sumAxis(f32, allocator, &a, 0, false);
/// defer s.deinit();
/// ```
pub fn sumAxis(comptime T: type, allocator: Allocator, a: *const NDArray(T), axis: usize, keepdims: bool) !NDArray(T) {
    if (axis >= a.rank()) return core.Error.IndexOutOfBounds;

    var new_shape: []usize = undefined;
    if (keepdims) {
        new_shape = try allocator.alloc(usize, a.rank());
        @memcpy(new_shape, a.shape);
        new_shape[axis] = 1;
    } else {
        new_shape = try allocator.alloc(usize, a.rank() - 1);
        var j: usize = 0;
        for (a.shape, 0..) |dim, i| {
            if (i != axis) {
                new_shape[j] = dim;
                j += 1;
            }
        }
    }
    defer allocator.free(new_shape);

    var result = try NDArray(T).zeros(allocator, new_shape);
    errdefer result.deinit();

    var iter = try core.NdIterator.init(allocator, a.shape);
    defer iter.deinit();

    var res_coords = try allocator.alloc(usize, result.rank());
    defer allocator.free(res_coords);

    while (iter.next()) |coords| {
        const val = try a.get(coords);

        if (keepdims) {
            @memcpy(res_coords, coords);
            res_coords[axis] = 0;
        } else {
            var j: usize = 0;
            for (coords, 0..) |c, i| {
                if (i != axis) {
                    res_coords[j] = c;
                    j += 1;
                }
            }
        }

        const current_sum = try result.get(res_coords);
        try result.set(res_coords, current_sum + val);
    }

    return result;
}

/// Computes the mean of elements along a given axis.
///
/// Logic:
/// result = mean(a, axis)
///
/// Arguments:
///     T: The data type of the array elements.
///     allocator: The allocator to use for the result array.
///     a: The input array.
///     axis: The axis along which to compute the mean.
///     keepdims: Whether to keep the reduced dimension.
///
/// Returns:
///     A new NDArray containing the means.
///
/// Example:
/// ```zig
/// var m = try stats.meanAxis(f32, allocator, &a, 0, false);
/// defer m.deinit();
/// ```
pub fn meanAxis(comptime T: type, allocator: Allocator, a: *const NDArray(T), axis: usize, keepdims: bool) !NDArray(T) {
    if (@typeInfo(T) != .float) {
        @compileError("meanAxis() requires floating point type");
    }
    const s = try sumAxis(T, allocator, a, axis, keepdims);
    const dim_size = a.shape[axis];
    const factor = 1.0 / @as(T, @floatFromInt(dim_size));

    for (s.data) |*val| {
        val.* *= factor;
    }
    return s;
}

pub const varianceAll = var_val;
pub const stddev = std_val;

/// Count number of occurrences of each value in array of non-negative ints.
///
/// Arguments:
///     allocator: The allocator to use for the result array.
///     a: The input array of non-negative integers.
///
/// Returns:
///     A new NDArray containing the counts.
///
/// Example:
/// ```zig
/// var counts = try stats.bincount(allocator, &a);
/// defer counts.deinit();
/// ```
pub fn bincount(allocator: Allocator, a: *const NDArray(i32)) !NDArray(usize) {
    if (a.size() == 0) return NDArray(usize).zeros(allocator, &.{0});

    // Find max value
    var max_val: i32 = 0;
    var iter = try core.NdIterator.init(allocator, a.shape);
    defer iter.deinit();

    while (iter.next()) |coords| {
        const val = try a.get(coords);
        if (val < 0) return core.Error.UnsupportedType; // Must be non-negative
        if (val > max_val) max_val = val;
    }

    const len = @as(usize, @intCast(max_val)) + 1;
    var result = try NDArray(usize).zeros(allocator, &.{len});

    iter.reset();
    while (iter.next()) |coords| {
        const val = try a.get(coords);
        const idx = @as(usize, @intCast(val));
        result.data[idx] += 1;
    }

    return result;
}

/// Compute the variance along the specified axis.
///
/// Logic:
/// result = var(a, axis)
///
/// Arguments:
///     T: The data type of the array elements.
///     allocator: The allocator to use for the result array.
///     a: The input array.
///     axis: The axis along which to compute the variance.
///
/// Returns:
///     A new NDArray containing the variances.
///
/// Example:
/// ```zig
/// var v = try stats.variance(f32, allocator, &a, 0);
/// defer v.deinit();
/// ```
pub fn variance(comptime T: type, allocator: Allocator, a: *const NDArray(T), axis: usize) !NDArray(T) {
    var means = try meanAxis(T, allocator, a, axis, true);
    defer means.deinit();

    var res_shape = try allocator.alloc(usize, a.rank() - 1);
    defer allocator.free(res_shape);
    var j: usize = 0;
    for (a.shape, 0..) |dim, i| {
        if (i != axis) {
            res_shape[j] = dim;
            j += 1;
        }
    }

    var result = try NDArray(T).zeros(allocator, res_shape);
    errdefer result.deinit();

    var iter = try core.NdIterator.init(allocator, a.shape);
    defer iter.deinit();

    var res_coords = try allocator.alloc(usize, result.rank());
    defer allocator.free(res_coords);

    var mean_coords = try allocator.alloc(usize, means.rank());
    defer allocator.free(mean_coords);

    while (iter.next()) |coords| {
        const val = try a.get(coords);

        @memcpy(mean_coords, coords);
        mean_coords[axis] = 0;
        const m = try means.get(mean_coords);

        const diff = val - m;

        j = 0;
        for (coords, 0..) |c, i| {
            if (i != axis) {
                res_coords[j] = c;
                j += 1;
            }
        }

        const current_ss = try result.get(res_coords);
        try result.set(res_coords, current_ss + diff * diff);
    }

    const n = @as(T, @floatFromInt(a.shape[axis]));
    for (result.data) |*val| {
        val.* /= n;
    }

    return result;
}

/// Computes the standard deviation along the specified axis.
///
/// Logic:
/// result = std(a, axis)
///
/// Arguments:
///     T: The data type of the array elements.
///     allocator: The allocator to use for the result array.
///     a: The input array.
///     axis: The axis along which to compute the standard deviation.
///
/// Returns:
///     A new NDArray containing the standard deviations.
///
/// Example:
/// ```zig
/// var s = try stats.stdDev(f32, allocator, &a, 0);
/// defer s.deinit();
/// ```
pub fn stdDev(comptime T: type, allocator: Allocator, a: *const NDArray(T), axis: usize) !NDArray(T) {
    const v = try variance(T, allocator, a, axis);
    for (v.data) |*val| {
        val.* = @sqrt(val.*);
    }
    return v;
}

/// Returns the sorted unique elements of an array.
///
/// Logic:
/// result = unique(a)
///
/// Arguments:
///     T: The data type of the array elements.
///     allocator: The allocator to use for the result array.
///     a: The input array.
///
/// Returns:
///     A new NDArray containing the sorted unique elements.
///
/// Example:
/// ```zig
/// var u = try stats.unique(f32, allocator, &a);
/// defer u.deinit();
/// ```
pub fn unique(comptime T: type, allocator: Allocator, a: *const NDArray(T)) !NDArray(T) {
    if (a.size() == 0) return NDArray(T).zeros(allocator, &.{0});

    var flat = try a.flatten();
    defer flat.deinit();

    std.mem.sort(T, flat.data, {}, std.sort.asc(T));

    var count: usize = 1;
    var i: usize = 1;
    while (i < flat.size()) : (i += 1) {
        if (flat.data[i] != flat.data[i - 1]) {
            count += 1;
        }
    }

    var result = try NDArray(T).init(allocator, &.{count});
    result.data[0] = flat.data[0];

    var j: usize = 1;
    i = 1;
    while (i < flat.size()) : (i += 1) {
        if (flat.data[i] != flat.data[i - 1]) {
            result.data[j] = flat.data[i];
            j += 1;
        }
    }

    return result;
}
