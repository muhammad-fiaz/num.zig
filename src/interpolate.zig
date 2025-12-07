const std = @import("std");
const core = @import("core.zig");
const NDArray = core.NDArray;
const Allocator = std.mem.Allocator;

/// 1D Linear Interpolation.
///
/// Arguments:
///     allocator: Allocator.
///     T: Element type (must be float).
///     x: X coordinates of data points (must be sorted).
///     y: Y coordinates of data points.
///     xi: X coordinates to evaluate at.
///
/// Returns:
///     New array with interpolated values.
pub fn interp1d(allocator: Allocator, comptime T: type, x: NDArray(T), y: NDArray(T), xi: NDArray(T)) !NDArray(T) {
    if (x.rank() != 1 or y.rank() != 1 or xi.rank() != 1) return core.Error.RankMismatch;
    if (x.size() != y.size()) return core.Error.DimensionMismatch;
    if (x.size() < 2) return core.Error.DimensionMismatch;

    var result = try NDArray(T).init(allocator, xi.shape);

    // For each point in xi, find interval in x and interpolate
    for (0..xi.size()) |i| {
        const val = xi.data[i];

        // Binary search or linear search? Linear for simplicity now, or binary for speed.
        // Let's do binary search.
        // std.sort.binarySearch requires a slice.

        // Handle extrapolation: clamp or linear extension?
        // NumPy defaults to constant extrapolation (edge values) or linear?
        // Let's do constant extrapolation for now.

        if (val <= x.data[0]) {
            result.data[i] = y.data[0];
            continue;
        }
        if (val >= x.data[x.size() - 1]) {
            result.data[i] = y.data[y.size() - 1];
            continue;
        }

        // Find index k such that x[k] <= val < x[k+1]
        var low: usize = 0;
        var high: usize = x.size() - 1;

        while (low < high - 1) {
            const mid = low + (high - low) / 2;
            if (x.data[mid] <= val) {
                low = mid;
            } else {
                high = mid;
            }
        }

        const x0 = x.data[low];
        const x1 = x.data[low + 1];
        const y0 = y.data[low];
        const y1 = y.data[low + 1];

        const slope = (y1 - y0) / (x1 - x0);
        result.data[i] = y0 + slope * (val - x0);
    }

    return result;
}
