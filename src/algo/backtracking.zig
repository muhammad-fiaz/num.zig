const std = @import("std");
const core = @import("../core.zig");
const NDArray = core.NDArray;
const Allocator = std.mem.Allocator;

/// Solves the Subset Sum problem: is there a subset of `set` that sums to `target`?
///
/// Logic: Backtracking.
///
/// Arguments:
///     allocator: Allocator.
///     set: Input array (1D, i32).
///     target: Target sum.
///
/// Returns:
///     bool.
///
/// Example:
/// ```zig
/// const exists = try algo.backtracking.subsetSum(allocator, &set, 9);
/// ```
pub fn subsetSum(allocator: Allocator, set: *const NDArray(i32), target: i32) !bool {
    if (set.rank() != 1) return core.Error.RankMismatch;

    // Copy data to slice for easier recursion (or just use get)
    // Since we need recursion, let's use a helper.
    // We can't easily pass NDArray to recursive function if we want to slice it efficiently without views.
    // But we can pass index.

    // For simplicity, let's assume contiguous or copy to slice.
    // If not contiguous, copy.
    var data: []const i32 = undefined;
    var temp_data: []i32 = undefined;

    if (set.flags().c_contiguous) {
        data = set.data;
    } else {
        temp_data = try allocator.alloc(i32, set.size());
        errdefer allocator.free(temp_data);

        var iter = try core.NdIterator.init(allocator, set.shape);
        defer iter.deinit();
        var i: usize = 0;
        while (iter.next()) |coords| {
            temp_data[i] = try set.get(coords);
            i += 1;
        }
        data = temp_data;
    }
    defer if (!set.flags().c_contiguous) allocator.free(temp_data);

    return subsetSumRecursive(data, target, data.len);
}

fn subsetSumRecursive(set: []const i32, sum: i32, n: usize) bool {
    if (sum == 0) return true;
    if (n == 0) return false;

    if (set[n - 1] > sum) {
        return subsetSumRecursive(set, sum, n - 1);
    }

    return subsetSumRecursive(set, sum, n - 1) or subsetSumRecursive(set, sum - set[n - 1], n - 1);
}
