const std = @import("std");
const core = @import("../core.zig");
const NDArray = core.NDArray;

/// Helper to assert array equality with tolerance for floats.
pub fn expectEqual(comptime T: type, expected: NDArray(T), actual: NDArray(T)) !void {
    try std.testing.expectEqualSlices(usize, expected.shape, actual.shape);

    if (@typeInfo(T) == .Float) {
        for (expected.data, 0..) |exp, i| {
            const act = actual.data[i];
            try std.testing.expectApproxEqAbs(exp, act, 1e-5);
        }
    } else {
        try std.testing.expectEqualSlices(T, expected.data, actual.data);
    }
}
