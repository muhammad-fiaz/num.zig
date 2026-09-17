//! Random number distributions and sampling functions.
//!
//! Provides generation for uniform, standard normal, integer, choice,
//! shuffle, and permutation distributions.

const std = @import("std");
const Array = @import("../core/array.zig").Array;
const empty = @import("../core/array.zig").empty;
const arange = @import("../core/array.zig").arange;
const DType = @import("../core/dtype.zig").DType;
const ShapeError = @import("../core/error.zig").ShapeError;
const DTypeError = @import("../core/error.zig").DTypeError;
const IndexError = @import("../core/error.zig").IndexError;
const Prng = @import("engine.zig").Prng;
const getDefaultPrng = @import("engine.zig").getDefaultPrng;

pub const UniformOptions = struct {
    shape: []const usize = &.{},
    low: f64 = 0.0,
    high: f64 = 1.0,
    dtype: DType = .f64,
    rng: ?*Prng = null,
};

/// Generates random numbers uniformly distributed in the half-open interval [low, high).
pub fn uniform(
    allocator: std.mem.Allocator,
    options: UniformOptions,
) (ShapeError || DTypeError || std.mem.Allocator.Error)!Array {
    if (options.high < options.low) return DTypeError.InvalidConversion;

    var out = try empty(allocator, .{
        .shape = options.shape,
        .dtype = options.dtype,
    });
    errdefer out.deinit();

    var prng = options.rng orelse getDefaultPrng();
    const r = prng.random();
    const range = options.high - options.low;

    inline for (std.meta.fields(DType)) |field| {
        const tag: DType = @enumFromInt(field.value);
        if (options.dtype == tag) {
            const T = tag.toType();
            const slice = out.asSlice(T) catch unreachable;

            for (slice) |*item| {
                const sample = options.low + r.float(f64) * range;
                if (@typeInfo(T) == .float) {
                    item.* = @floatCast(sample);
                } else if (@typeInfo(T) == .int) {
                    item.* = @intFromFloat(sample);
                } else if (T == bool) {
                    item.* = (sample >= 0.5);
                }
            }
            return out;
        }
    }

    return out;
}

/// Convenience function generating uniform [0.0, 1.0) numbers.
pub fn rand(
    allocator: std.mem.Allocator,
    shape: []const usize,
    rng: ?*Prng,
) !Array {
    return uniform(allocator, .{
        .shape = shape,
        .low = 0.0,
        .high = 1.0,
        .dtype = .f64,
        .rng = rng,
    });
}

pub const NormalOptions = struct {
    shape: []const usize = &.{},
    loc: f64 = 0.0,
    scale: f64 = 1.0,
    dtype: DType = .f64,
    rng: ?*Prng = null,
};

/// Generates random numbers from a normal (Gaussian) distribution.
pub fn normal(
    allocator: std.mem.Allocator,
    options: NormalOptions,
) (ShapeError || DTypeError || std.mem.Allocator.Error)!Array {
    if (options.scale < 0) return DTypeError.InvalidConversion;

    var out = try empty(allocator, .{
        .shape = options.shape,
        .dtype = options.dtype,
    });
    errdefer out.deinit();

    var prng = options.rng orelse getDefaultPrng();
    const r = prng.random();

    inline for (std.meta.fields(DType)) |field| {
        const tag: DType = @enumFromInt(field.value);
        if (options.dtype == tag) {
            const T = tag.toType();
            const slice = out.asSlice(T) catch unreachable;

            for (slice) |*item| {
                const sample = options.loc + r.floatNorm(f64) * options.scale;
                if (@typeInfo(T) == .float) {
                    item.* = @floatCast(sample);
                } else if (@typeInfo(T) == .int) {
                    item.* = @intFromFloat(sample);
                } else if (T == bool) {
                    item.* = (sample >= 0.0);
                }
            }
            return out;
        }
    }

    return out;
}

/// Convenience function generating standard normal (mean 0, std 1) numbers.
pub fn randn(
    allocator: std.mem.Allocator,
    shape: []const usize,
    rng: ?*Prng,
) !Array {
    return normal(allocator, .{
        .shape = shape,
        .loc = 0.0,
        .scale = 1.0,
        .dtype = .f64,
        .rng = rng,
    });
}

pub const IntegersOptions = struct {
    shape: []const usize = &.{},
    low: i64 = 0,
    high: i64 = 100,
    endpoint: bool = false,
    dtype: DType = .i64,
    rng: ?*Prng = null,
};

/// Generates random integers from low (inclusive) to high (exclusive or inclusive).
pub fn integers(
    allocator: std.mem.Allocator,
    options: IntegersOptions,
) (ShapeError || DTypeError || std.mem.Allocator.Error)!Array {
    const min_val = options.low;
    const max_val = if (options.endpoint) options.high else options.high - 1;

    if (max_val < min_val) return DTypeError.InvalidConversion;

    var out = try empty(allocator, .{
        .shape = options.shape,
        .dtype = options.dtype,
    });
    errdefer out.deinit();

    var prng = options.rng orelse getDefaultPrng();
    const r = prng.random();

    inline for (std.meta.fields(DType)) |field| {
        const tag: DType = @enumFromInt(field.value);
        if (options.dtype == tag) {
            const T = tag.toType();
            const slice = out.asSlice(T) catch unreachable;

            for (slice) |*item| {
                const val = r.intRangeAtMost(i64, min_val, max_val);
                if (@typeInfo(T) == .int) {
                    item.* = @intCast(val);
                } else if (@typeInfo(T) == .float) {
                    item.* = @floatFromInt(val);
                } else if (T == bool) {
                    item.* = (val != 0);
                }
            }
            return out;
        }
    }

    return out;
}

pub const ChoiceOptions = struct {
    size: ?usize = null,
    replace: bool = true,
    rng: ?*Prng = null,
};

/// Generates a random sample from a 1D array.
pub fn choice(
    allocator: std.mem.Allocator,
    arr: Array,
    options: ChoiceOptions,
) (ShapeError || DTypeError || IndexError || std.mem.Allocator.Error)!Array {
    const s = arr.shape();
    if (s.ndim != 1) return ShapeError.InvalidDimension;
    const n = s.dims[0];
    if (n == 0) return ShapeError.EmptyArray;

    var prng = options.rng orelse getDefaultPrng();
    const r = prng.random();

    // 0D scalar choice
    if (options.size == null) {
        const idx = r.intRangeLessThan(usize, 0, n);
        var out = try empty(allocator, .{ .shape = &.{}, .dtype = arr.dtype });
        errdefer out.deinit();

        inline for (std.meta.fields(DType)) |field| {
            const tag: DType = @enumFromInt(field.value);
            if (arr.dtype == tag) {
                const T = tag.toType();
                const val = try arr.get(T, &.{idx});
                try out.set(T, &.{}, val);
                return out;
            }
        }
        return out;
    }

    const k = options.size.?;
    if (!options.replace and k > n) {
        return ShapeError.InvalidDimension;
    }

    var out = try empty(allocator, .{ .shape = &.{k}, .dtype = arr.dtype });
    errdefer out.deinit();

    inline for (std.meta.fields(DType)) |field| {
        const tag: DType = @enumFromInt(field.value);
        if (arr.dtype == tag) {
            const T = tag.toType();
            const out_slice = out.asSlice(T) catch unreachable;

            if (options.replace) {
                for (0..k) |i| {
                    const idx = r.intRangeLessThan(usize, 0, n);
                    out_slice[i] = try arr.get(T, &.{idx});
                }
            } else {
                // Reservoir or partial shuffle of indices
                const indices = try allocator.alloc(usize, n);
                defer allocator.free(indices);
                for (0..n) |i| indices[i] = i;

                // Fisher-Yates up to k
                for (0..k) |i| {
                    const swap_idx = r.intRangeLessThan(usize, i, n);
                    const tmp = indices[i];
                    indices[i] = indices[swap_idx];
                    indices[swap_idx] = tmp;

                    out_slice[i] = try arr.get(T, &.{indices[i]});
                }
            }
            return out;
        }
    }

    return out;
}

pub const ShuffleOptions = struct {
    rng: ?*Prng = null,
};

/// Shuffles an array in-place along its first axis.
pub fn shuffle(
    arr: *Array,
    options: ShuffleOptions,
) (ShapeError || DTypeError || IndexError || std.mem.Allocator.Error)!void {
    const s = arr.shape();
    if (s.ndim == 0) return;
    const n = s.dims[0];
    if (n <= 1) return;

    var prng = options.rng orelse getDefaultPrng();
    const r = prng.random();

    if (s.ndim == 1 and arr.isContiguous()) {
        inline for (std.meta.fields(DType)) |field| {
            const tag: DType = @enumFromInt(field.value);
            if (arr.dtype == tag) {
                const T = tag.toType();
                const slice = arr.asSlice(T) catch unreachable;
                r.shuffle(T, slice);
                return;
            }
        }
    }

    // General multidimensional swap along axis 0
    const slice_elems = arr.elementCount() / n;
    const byte_size = slice_elems * arr.dtype.sizeOf();
    const temp_buf = try arr.allocator.alloc(u8, byte_size);
    defer arr.allocator.free(temp_buf);

    for (0..n - 1) |i| {
        const j = r.intRangeLessThan(usize, i, n);
        if (i == j) continue;

        // Swap subarray i and j
        const stride_elem = arr.stride_vals[0];
        const elem_sz = arr.dtype.sizeOf();
        const ptr_i = arr.data_ptr + @as(usize, @intCast(@as(isize, @intCast(i)) * stride_elem)) * elem_sz;
        const ptr_j = arr.data_ptr + @as(usize, @intCast(@as(isize, @intCast(j)) * stride_elem)) * elem_sz;

        @memcpy(temp_buf, ptr_i[0..byte_size]);
        @memcpy(ptr_i[0..byte_size], ptr_j[0..byte_size]);
        @memcpy(ptr_j[0..byte_size], temp_buf);
    }
}

/// Randomly permutes a sequence or returns a permuted range.
pub fn permutation(
    arr: Array,
    options: ShuffleOptions,
) (ShapeError || DTypeError || IndexError || std.mem.Allocator.Error)!Array {
    var out = try arr.clone();
    errdefer out.deinit();

    try shuffle(&out, options);
    return out;
}

test "uniform distribution and determinism" {
    const allocator = std.testing.allocator;
    var rng = Prng.init(42);

    var u = try uniform(allocator, .{
        .shape = &.{ 10, 10 },
        .low = 5.0,
        .high = 15.0,
        .dtype = .f64,
        .rng = &rng,
    });
    defer u.deinit();

    const slice = try u.asSlice(f64);
    for (slice) |val| {
        try std.testing.expect(val >= 5.0 and val < 15.0);
    }
}

test "normal distribution properties" {
    const allocator = std.testing.allocator;
    var rng = Prng.init(1234);

    const N = 1000;
    var norm_arr = try normal(allocator, .{
        .shape = &.{N},
        .loc = 10.0,
        .scale = 2.0,
        .dtype = .f64,
        .rng = &rng,
    });
    defer norm_arr.deinit();

    const slice = try norm_arr.asSlice(f64);
    var sum_val: f64 = 0;
    for (slice) |val| {
        sum_val += val;
    }
    const sample_mean = sum_val / @as(f64, @floatFromInt(N));

    // Sample mean of 1000 items should be within 3 standard errors of 10.0: 3 * (2 / sqrt(1000)) ~ 0.19
    try std.testing.expect(@abs(sample_mean - 10.0) < 0.5);
}

test "integers distribution" {
    const allocator = std.testing.allocator;
    var rng = Prng.init(777);

    var ints = try integers(allocator, .{
        .shape = &.{50},
        .low = 10,
        .high = 20,
        .dtype = .i32,
        .rng = &rng,
    });
    defer ints.deinit();

    const slice = try ints.asSlice(i32);
    for (slice) |val| {
        try std.testing.expect(val >= 10 and val < 20);
    }
}

test "choice with and without replacement" {
    const allocator = std.testing.allocator;
    var rng = Prng.init(999);

    const fromSlice = @import("../core/array.zig").fromSlice;
    const items = [_]i32{ 10, 20, 30, 40, 50 };
    var arr = try fromSlice(allocator, i32, .{ .data = &items, .shape = &.{5} });
    defer arr.deinit();

    // Choice with replacement
    var c_rep = try choice(allocator, arr, .{
        .size = 10,
        .replace = true,
        .rng = &rng,
    });
    defer c_rep.deinit();
    try std.testing.expectEqual(@as(usize, 10), c_rep.elementCount());

    // Choice without replacement
    var c_no_rep = try choice(allocator, arr, .{
        .size = 3,
        .replace = false,
        .rng = &rng,
    });
    defer c_no_rep.deinit();
    try std.testing.expectEqual(@as(usize, 3), c_no_rep.elementCount());

    const c_slice = try c_no_rep.asSlice(i32);
    // Elements must be unique
    try std.testing.expect(c_slice[0] != c_slice[1]);
    try std.testing.expect(c_slice[1] != c_slice[2]);
    try std.testing.expect(c_slice[0] != c_slice[2]);
}

test "shuffle and permutation" {
    const allocator = std.testing.allocator;
    var rng = Prng.init(54321);

    const fromSlice = @import("../core/array.zig").fromSlice;
    const items = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    var arr = try fromSlice(allocator, f64, .{ .data = &items, .shape = &.{8} });
    defer arr.deinit();

    var perm = try permutation(arr, .{ .rng = &rng });
    defer perm.deinit();

    try std.testing.expectEqual(arr.elementCount(), perm.elementCount());

    // In-place shuffle
    try shuffle(&arr, .{ .rng = &rng });
    try std.testing.expectEqual(@as(usize, 8), arr.elementCount());
}
