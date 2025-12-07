const std = @import("std");
const core = @import("../core.zig");
const NDArray = core.NDArray;
const Allocator = std.mem.Allocator;

/// Performs linear search on an array.
///
/// Logic: Iterate and find index where value matches.
///
/// Arguments:
///     allocator: Allocator.
///     T: Type.
///     a: Input array.
///     target: Value to search for.
///
/// Returns:
///     Optional usize (index if found, null otherwise).
///
/// Example:
/// ```zig
/// const idx = try algo.search.linearSearch(allocator, f32, &a, 5.0);
/// ```
pub fn linearSearch(allocator: Allocator, comptime T: type, a: *const NDArray(T), target: T) !?usize {
    if (a.flags().c_contiguous) {
        for (a.data, 0..) |val, i| {
            if (val == target) return i;
        }
    } else {
        var iter = try core.NdIterator.init(allocator, a.shape);
        defer iter.deinit();
        var i: usize = 0;
        while (iter.next()) |coords| {
            const val = try a.get(coords);
            if (val == target) return i;
            i += 1;
        }
    }
    return null;
}

/// Performs binary search on a sorted array (1D).
///
/// Logic: Standard binary search.
///
/// Arguments:
///     allocator: Allocator.
///     T: Type.
///     a: Input array (must be sorted and 1D).
///     target: Value to search for.
///
/// Returns:
///     Optional usize (index if found, null otherwise).
///
/// Example:
/// ```zig
/// const idx = try algo.search.binarySearch(allocator, f32, &a, 5.0);
/// ```
pub fn binarySearch(allocator: Allocator, comptime T: type, a: *const NDArray(T), target: T) !?usize {
    _ = allocator;
    if (a.rank() != 1) return core.Error.RankMismatch;

    var left: usize = 0;
    var right: usize = a.size();

    while (left < right) {
        const mid = left + (right - left) / 2;
        var val: T = undefined;

        if (a.flags().c_contiguous) {
            val = a.data[mid];
        } else {
            val = try a.get(&.{mid});
        }

        if (val == target) return mid;
        if (val < target) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return null;
}
