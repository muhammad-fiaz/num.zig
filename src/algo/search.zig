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
        defer iter.deinit(allocator);
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

/// Performs interpolation search on a sorted array (1D).
///
/// Logic: Interpolation search.
///
/// Arguments:
///     allocator: Allocator.
///     T: Type.
///     a: Input array (must be sorted and 1D).
///     target: Value to search for.
///
/// Returns:
///     Optional usize (index if found, null otherwise).
pub fn interpolationSearch(allocator: Allocator, comptime T: type, a: *const NDArray(T), target: T) !?usize {
    _ = allocator;
    if (a.rank() != 1) return core.Error.DimensionMismatch;

    // Only works for numeric types
    switch (@typeInfo(T)) {
        .Int, .Float => {},
        else => @compileError("Interpolation search only works for numeric types"),
    }

    var low: usize = 0;
    var high: usize = a.size() - 1;

    while (low <= high) {
        const val_low = try a.get(&.{low});
        const val_high = try a.get(&.{high});

        if (target < val_low or target > val_high) return null;

        if (val_high == val_low) {
            if (val_low == target) return low;
            return null;
        }

        // Convert to f64 for calculation
        const t_f64: f64 = switch (@typeInfo(T)) {
            .Int => @floatFromInt(target),
            .Float => @floatCast(target),
            else => unreachable,
        };
        const vl_f64: f64 = switch (@typeInfo(T)) {
            .Int => @floatFromInt(val_low),
            .Float => @floatCast(val_low),
            else => unreachable,
        };
        const vh_f64: f64 = switch (@typeInfo(T)) {
            .Int => @floatFromInt(val_high),
            .Float => @floatCast(val_high),
            else => unreachable,
        };

        const dist = @as(f64, @floatFromInt(high - low));
        const pos_f64 = @as(f64, @floatFromInt(low)) + ((t_f64 - vl_f64) * dist / (vh_f64 - vl_f64));
        const pos = @as(usize, @intFromFloat(pos_f64));

        if (pos < low or pos > high) return null;

        const val_pos = try a.get(&.{pos});

        if (val_pos == target) return pos;

        if (val_pos < target) {
            low = pos + 1;
        } else {
            high = pos - 1;
        }
    }
    return null;
}

test "search" {
    const allocator = std.testing.allocator;
    var a = try NDArray(f64).init(allocator, &.{5});
    defer a.deinit(allocator);
    a.data[0] = 1;
    a.data[1] = 3;
    a.data[2] = 5;
    a.data[3] = 7;
    a.data[4] = 9;

    try std.testing.expectEqual(try linearSearch(allocator, f64, &a, 5.0), 2);
    try std.testing.expectEqual(try linearSearch(allocator, f64, &a, 99.0), null);

    try std.testing.expectEqual(try binarySearch(allocator, f64, &a, 7.0), 3);
    try std.testing.expectEqual(try binarySearch(allocator, f64, &a, 2.0), null);
}
