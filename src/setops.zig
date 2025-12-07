const std = @import("std");
const core = @import("core.zig");
const NDArray = core.NDArray;
const Allocator = std.mem.Allocator;

/// Find the unique elements of an array.
/// Returns the sorted unique elements of an array.
///
/// Arguments:
///     allocator: Allocator.
///     T: Element type.
///     a: Input array.
///
/// Returns:
///     Sorted unique elements.
pub fn unique(allocator: Allocator, comptime T: type, a: NDArray(T)) !NDArray(T) {
    // Flatten array first
    var flat_tmp = try a.flatten();
    defer flat_tmp.deinit();

    var flat = try flat_tmp.copy();
    defer flat.deinit();

    // Sort
    std.mem.sort(T, flat.data, {}, std.sort.asc(T));

    // Count unique
    if (flat.size() == 0) return try NDArray(T).init(allocator, &.{0});

    var count: usize = 1;
    for (flat.data[1..], 0..) |val, i| {
        if (val != flat.data[i]) {
            count += 1;
        }
    }

    var result = try NDArray(T).init(allocator, &.{count});
    result.data[0] = flat.data[0];

    var idx: usize = 1;
    for (flat.data[1..], 0..) |val, i| {
        if (val != flat.data[i]) {
            result.data[idx] = val;
            idx += 1;
        }
    }

    return result;
}

/// Test whether each element of a 1-D array is also present in a second array.
/// Returns a boolean array the same length as ar1 that is True where an element of ar1 is in ar2 and False otherwise.
pub fn in1d(allocator: Allocator, comptime T: type, ar1: NDArray(T), ar2: NDArray(T)) !NDArray(bool) {
    // Flatten both
    var f1 = try ar1.flatten();
    defer f1.deinit();
    var f2 = try ar2.flatten();
    defer f2.deinit();

    // Sort f2 for binary search
    var f2_sorted = try f2.copy();
    defer f2_sorted.deinit();
    std.mem.sort(T, f2_sorted.data, {}, std.sort.asc(T));

    var result = try NDArray(bool).init(allocator, f1.shape);

    for (f1.data, 0..) |val, i| {
        var found = false;
        var left: usize = 0;
        var right: usize = f2_sorted.size();

        while (left < right) {
            const mid = left + (right - left) / 2;
            if (f2_sorted.data[mid] == val) {
                found = true;
                break;
            } else if (f2_sorted.data[mid] < val) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        result.data[i] = found;
    }

    return result;
}

/// Find the intersection of two arrays.
/// Returns the sorted, unique values that are in both of the input arrays.
pub fn intersect1d(allocator: Allocator, comptime T: type, ar1: NDArray(T), ar2: NDArray(T)) !NDArray(T) {
    var uniq1 = try unique(allocator, T, ar1);
    defer uniq1.deinit();
    var uniq2 = try unique(allocator, T, ar2);
    defer uniq2.deinit();

    // Both are sorted. Use merge-like approach.
    var temp = try std.ArrayList(T).initCapacity(allocator, @min(uniq1.size(), uniq2.size()));
    defer temp.deinit();

    var i: usize = 0;
    var j: usize = 0;
    while (i < uniq1.size() and j < uniq2.size()) {
        if (uniq1.data[i] < uniq2.data[j]) {
            i += 1;
        } else if (uniq1.data[i] > uniq2.data[j]) {
            j += 1;
        } else {
            try temp.append(uniq1.data[i]);
            i += 1;
            j += 1;
        }
    }

    return NDArray(T).init(allocator, &.{temp.items.len}, temp.items);
}

/// Find the union of two arrays.
/// Returns the sorted, unique values that are in either of the two input arrays.
pub fn union1d(allocator: Allocator, comptime T: type, ar1: NDArray(T), ar2: NDArray(T)) !NDArray(T) {
    // Concatenate and unique
    const size = ar1.size() + ar2.size();
    var concat = try NDArray(T).init(allocator, &.{size});
    defer concat.deinit();

    @memcpy(concat.data[0..ar1.size()], ar1.data);
    @memcpy(concat.data[ar1.size()..], ar2.data);

    return unique(allocator, T, concat);
}

/// Find the set difference of two arrays.
/// Return the unique values in ar1 that are not in ar2.
pub fn setdiff1d(allocator: Allocator, comptime T: type, ar1: NDArray(T), ar2: NDArray(T)) !NDArray(T) {
    var uniq1 = try unique(allocator, T, ar1);
    defer uniq1.deinit();
    var uniq2 = try unique(allocator, T, ar2);
    defer uniq2.deinit();

    var temp = try std.ArrayList(T).initCapacity(allocator, uniq1.size());
    defer temp.deinit();

    var i: usize = 0;
    var j: usize = 0;
    while (i < uniq1.size() and j < uniq2.size()) {
        if (uniq1.data[i] < uniq2.data[j]) {
            try temp.append(uniq1.data[i]);
            i += 1;
        } else if (uniq1.data[i] > uniq2.data[j]) {
            j += 1;
        } else {
            // Equal, skip
            i += 1;
            j += 1;
        }
    }
    // Append remaining
    while (i < uniq1.size()) : (i += 1) {
        try temp.append(uniq1.data[i]);
    }

    return NDArray(T).init(allocator, &.{temp.items.len}, temp.items);
}

/// Find the set exclusive-or of two arrays.
/// Return the sorted, unique values that are in only one (not both) of the input arrays.
pub fn setxor1d(allocator: Allocator, comptime T: type, ar1: NDArray(T), ar2: NDArray(T)) !NDArray(T) {
    var uniq1 = try unique(allocator, T, ar1);
    defer uniq1.deinit();
    var uniq2 = try unique(allocator, T, ar2);
    defer uniq2.deinit();

    var temp = try std.ArrayList(T).initCapacity(allocator, uniq1.size() + uniq2.size());
    defer temp.deinit();

    var i: usize = 0;
    var j: usize = 0;
    while (i < uniq1.size() and j < uniq2.size()) {
        if (uniq1.data[i] < uniq2.data[j]) {
            try temp.append(uniq1.data[i]);
            i += 1;
        } else if (uniq1.data[i] > uniq2.data[j]) {
            try temp.append(uniq2.data[j]);
            j += 1;
        } else {
            // Equal, skip both
            i += 1;
            j += 1;
        }
    }
    while (i < uniq1.size()) : (i += 1) {
        try temp.append(uniq1.data[i]);
    }
    while (j < uniq2.size()) : (j += 1) {
        try temp.append(uniq2.data[j]);
    }

    return NDArray(T).init(allocator, &.{temp.items.len}, temp.items);
}
