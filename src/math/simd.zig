const std = @import("std");

/// Adds two arrays using SIMD if available.
/// Fallback to scalar loop.
pub fn add(comptime T: type, dest: []T, a: []const T, b: []const T) void {
    const len = dest.len;
    std.debug.assert(a.len == len and b.len == len);

    // Check if T is supported for SIMD
    const vector_len = std.simd.suggestVectorLength(T) orelse {
        // Fallback
        for (dest, a, b) |*d, val_a, val_b| {
            d.* = val_a + val_b;
        }
        return;
    };

    var i: usize = 0;
    while (i + vector_len <= len) : (i += vector_len) {
        const va: @Vector(vector_len, T) = a[i..][0..vector_len].*;
        const vb: @Vector(vector_len, T) = b[i..][0..vector_len].*;
        dest[i..][0..vector_len].* = va + vb;
    }

    // Handle remaining
    while (i < len) : (i += 1) {
        dest[i] = a[i] + b[i];
    }
}

/// Multiplies two arrays using SIMD.
pub fn mul(comptime T: type, dest: []T, a: []const T, b: []const T) void {
    const len = dest.len;
    std.debug.assert(a.len == len and b.len == len);

    const vector_len = std.simd.suggestVectorLength(T) orelse {
        for (dest, a, b) |*d, val_a, val_b| {
            d.* = val_a * val_b;
        }
        return;
    };

    var i: usize = 0;
    while (i + vector_len <= len) : (i += vector_len) {
        const va: @Vector(vector_len, T) = a[i..][0..vector_len].*;
        const vb: @Vector(vector_len, T) = b[i..][0..vector_len].*;
        dest[i..][0..vector_len].* = va * vb;
    }

    while (i < len) : (i += 1) {
        dest[i] = a[i] * b[i];
    }
}

test "simd" {
    var a = [_]f32{ 1, 2, 3, 4, 5 };
    var b = [_]f32{ 10, 20, 30, 40, 50 };
    var dest = [_]f32{ 0, 0, 0, 0, 0 };

    add(f32, &dest, &a, &b);
    try std.testing.expectEqual(dest[0], 11);
    try std.testing.expectEqual(dest[4], 55);

    mul(f32, &dest, &a, &b);
    try std.testing.expectEqual(dest[0], 10);
    try std.testing.expectEqual(dest[4], 250);
}
