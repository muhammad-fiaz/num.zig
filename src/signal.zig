const std = @import("std");
const core = @import("core.zig");
const NDArray = core.NDArray;
const Allocator = std.mem.Allocator;

pub const ConvolveMode = enum {
    full,
    valid,
    same,
};

/// Convolve two 1-dimensional arrays.
///
/// Arguments:
///     allocator: Allocator.
///     T: Element type.
///     a: First input array (signal).
///     v: Second input array (kernel).
///     mode: 'full', 'valid', or 'same'.
///
/// Returns:
///     Convolved array.
pub fn convolve(allocator: Allocator, comptime T: type, a: NDArray(T), v: NDArray(T), mode: ConvolveMode) !NDArray(T) {
    if (a.rank() != 1 or v.rank() != 1) return core.Error.RankMismatch;

    const n = a.size();
    const m = v.size();

    if (n == 0 or m == 0) return core.Error.DimensionMismatch;

    var out_size: usize = 0;
    switch (mode) {
        .full => out_size = n + m - 1,
        .valid => out_size = if (n >= m) n - m + 1 else m - n + 1,
        .same => out_size = @max(n, m),
    }

    var result = try NDArray(T).zeros(allocator, &.{out_size});
    errdefer result.deinit();

    // Standard definition: $(a * v)[n] = \sum_m a[m] v[n-m]$
    // Implements 'full' convolution first, then slices for other modes.

    const full_size = n + m - 1;
    var full_result = if (mode == .full) result else try NDArray(T).zeros(allocator, &.{full_size});
    defer if (mode != .full) full_result.deinit();

    // Iterate over result indices
    for (0..full_size) |k| {
        var sum: T = 0;
        // Indices i for a and j for v such that i + j = k
        // 0 <= i < n
        // 0 <= j < m => 0 <= k-i < m => k-m < i <= k

        const start_i = if (k >= m) k - m + 1 else 0;
        const end_i = @min(k + 1, n);

        var i = start_i;
        while (i < end_i) : (i += 1) {
            const j = k - i;
            // v is traversed backwards relative to correlation.
            const val_a = try a.get(&.{i});
            const val_v = try v.get(&.{j});
            sum += val_a * val_v;
        }
        full_result.data[k] = sum;
    }

    if (mode == .full) return result;

    // Slice for valid/same
    if (mode == .valid) {
        // Valid part is centered.
        // Size is max(M, N) - min(M, N) + 1.
        const min_len = @min(n, m);
        const start = min_len - 1;

        for (0..out_size) |i| {
            result.data[i] = full_result.data[start + i];
        }
    } else if (mode == .same) {
        // Centered part of 'full' with size max(M, N).
        const start = (full_size - out_size) / 2;

        for (0..out_size) |i| {
            result.data[i] = full_result.data[start + i];
        }
    }

    return result;
}

/// Cross-correlation of two 1-dimensional sequences.
///
/// This function computes the correlation as generally defined in signal processing texts:
/// c[k] = sum_n a[n] * conj(v[n+k])
///
/// Arguments:
///     allocator: Allocator.
///     T: Element type.
///     a: First input array.
///     v: Second input array.
///     mode: 'full', 'valid', or 'same'.
///
/// Returns:
///     Correlated array.
pub fn correlate(allocator: Allocator, comptime T: type, a: NDArray(T), v: NDArray(T), mode: ConvolveMode) !NDArray(T) {
    // Correlation is convolution with the second input reversed (and conjugated if complex, but we assume real for now).
    // correlate(a, v) == convolve(a, v[::-1])

    if (v.rank() != 1) return core.Error.RankMismatch;

    // Reverse v
    var v_rev = try v.copy();
    defer v_rev.deinit();
    std.mem.reverse(T, v_rev.data);

    return convolve(allocator, T, a, v_rev, mode);
}
