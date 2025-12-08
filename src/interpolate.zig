const std = @import("std");
const core = @import("core.zig");
const NDArray = core.NDArray;
const Allocator = std.mem.Allocator;

pub const InterpKind = enum {
    linear,
    nearest,
};

pub const BoundsCheck = enum {
    clamp, // Use edge values
    extrapolate, // Linear extrapolation (only for linear kind)
    fill, // Use fill_value
    raise, // Return error
};

pub fn InterpOptions(comptime T: type) type {
    return struct {
        kind: InterpKind = .linear,
        bounds_check: BoundsCheck = .clamp,
        fill_value: T = 0,
    };
}

/// 1D Interpolation.
///
/// Arguments:
///     allocator: Allocator.
///     T: Element type (must be float).
///     x: X coordinates of data points (must be sorted).
///     y: Y coordinates of data points.
///     xi: X coordinates to evaluate at.
///     options: Interpolation options.
///
/// Returns:
///     New array with interpolated values.
pub fn interp1d(allocator: Allocator, comptime T: type, x: NDArray(T), y: NDArray(T), xi: NDArray(T), options: InterpOptions(T)) !NDArray(T) {
    if (x.rank() != 1 or y.rank() != 1 or xi.rank() != 1) return core.Error.RankMismatch;
    if (x.size() != y.size()) return core.Error.DimensionMismatch;
    if (x.size() < 2) return core.Error.DimensionMismatch;

    var result = try NDArray(T).init(allocator, xi.shape);
    errdefer result.deinit(allocator);

    const x_data = x.data;
    const y_data = y.data;

    for (0..xi.size()) |i| {
        const val = xi.data[i];

        // Handle bounds
        if (val < x_data[0]) {
            switch (options.bounds_check) {
                .clamp => {
                    result.data[i] = y_data[0];
                    continue;
                },
                .fill => {
                    result.data[i] = options.fill_value;
                    continue;
                },
                .raise => return core.Error.IndexOutOfBounds,
                .extrapolate => {
                    if (options.kind != .linear) {
                        result.data[i] = y_data[0];
                        continue;
                    }
                    // Fall through for linear extrapolation
                },
            }
        } else if (val > x_data[x.size() - 1]) {
            switch (options.bounds_check) {
                .clamp => {
                    result.data[i] = y_data[y.size() - 1];
                    continue;
                },
                .fill => {
                    result.data[i] = options.fill_value;
                    continue;
                },
                .raise => return core.Error.IndexOutOfBounds,
                .extrapolate => {
                    if (options.kind != .linear) {
                        result.data[i] = y_data[y.size() - 1];
                        continue;
                    }
                    // Fall through
                },
            }
        }

        // Binary search to find the interval
        var search_low: usize = 0;
        var search_high: usize = x.size();
        while (search_low < search_high) {
            const mid = search_low + (search_high - search_low) / 2;
            if (x_data[mid] < val) {
                search_low = mid + 1;
            } else {
                search_high = mid;
            }
        }
        const idx = search_low;

        var low: usize = 0;
        var high: usize = 0;

        if (idx == 0) {
            // Extrapolating below (only reachable if bounds_check == .extrapolate)
            low = 0;
            high = 1;
        } else if (idx >= x.size()) {
            // Extrapolating above (only reachable if bounds_check == .extrapolate)
            low = x.size() - 2;
            high = x.size() - 1;
        } else {
            // val is between x[idx-1] and x[idx]
            low = idx - 1;
            high = idx;
        }

        switch (options.kind) {
            .linear => {
                const x0 = x_data[low];
                const x1 = x_data[high];
                const y0 = y_data[low];
                const y1 = y_data[high];
                const slope = (y1 - y0) / (x1 - x0);
                result.data[i] = y0 + slope * (val - x0);
            },
            .nearest => {
                const dist_low = @abs(val - x_data[low]);
                const dist_high = @abs(val - x_data[high]);
                if (dist_low <= dist_high) {
                    result.data[i] = y_data[low];
                } else {
                    result.data[i] = y_data[high];
                }
            },
        }
    }

    return result;
}

test "interpolate linear" {
    const allocator = std.testing.allocator;
    var x = try NDArray(f32).init(allocator, &.{3});
    defer x.deinit(allocator);
    x.data[0] = 0.0;
    x.data[1] = 1.0;
    x.data[2] = 2.0;

    var y = try NDArray(f32).init(allocator, &.{3});
    defer y.deinit(allocator);
    y.data[0] = 0.0;
    y.data[1] = 2.0;
    y.data[2] = 0.0;

    var xi = try NDArray(f32).init(allocator, &.{3});
    defer xi.deinit(allocator);
    xi.data[0] = 0.5;
    xi.data[1] = 1.5;
    xi.data[2] = 2.5;

    var res = try interp1d(allocator, f32, x, y, xi, .{ .kind = .linear, .bounds_check = .clamp });
    defer res.deinit(allocator);

    try std.testing.expectApproxEqAbs(res.data[0], 1.0, 1e-4);
    try std.testing.expectApproxEqAbs(res.data[1], 1.0, 1e-4);
    try std.testing.expectApproxEqAbs(res.data[2], 0.0, 1e-4);
}

test "interpolate nearest" {
    const allocator = std.testing.allocator;
    var x = try NDArray(f32).init(allocator, &.{3});
    defer x.deinit(allocator);
    x.data[0] = 0.0;
    x.data[1] = 1.0;
    x.data[2] = 2.0;

    var y = try NDArray(f32).init(allocator, &.{3});
    defer y.deinit(allocator);
    y.data[0] = 0.0;
    y.data[1] = 2.0;
    y.data[2] = 0.0;

    var xi = try NDArray(f32).init(allocator, &.{2});
    defer xi.deinit(allocator);
    xi.data[0] = 0.4; // Closer to 0.0 -> 0.0
    xi.data[1] = 0.6; // Closer to 1.0 -> 2.0

    var res = try interp1d(allocator, f32, x, y, xi, .{ .kind = .nearest });
    defer res.deinit(allocator);

    try std.testing.expectEqual(res.data[0], 0.0);
    try std.testing.expectEqual(res.data[1], 2.0);
}
