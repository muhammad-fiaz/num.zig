//! Descriptive statistics: mean, median, variance, standard deviation, and quantiles.
//!
//! Provides axis-aware and global statistical aggregations.

const std = @import("std");
const Array = @import("../core/array.zig").Array;
const zeros = @import("../core/array.zig").zeros;
const empty = @import("../core/array.zig").empty;
const fromSlice = @import("../core/array.zig").fromSlice;
const DType = @import("../core/dtype.zig").DType;
const Shape = @import("../core/shape.zig").Shape;
const Strides = @import("../core/shape.zig").Strides;
const MAX_RANK = @import("../core/shape.zig").MAX_RANK;
const ShapeError = @import("../core/error.zig").ShapeError;
const DTypeError = @import("../core/error.zig").DTypeError;
const IndexError = @import("../core/error.zig").IndexError;

const mean_fn = @import("../ops/reduce.zig").mean;
const sum_fn = @import("../ops/reduce.zig").sum;
const sub_fn = @import("../ops/elementwise.zig").subtract;
const mul_fn = @import("../ops/elementwise.zig").multiply;
const sqrt_fn = @import("../ops/elementwise.zig").sqrt;
const ravel = @import("../manip/reshape.zig").ravel;

pub const mean = mean_fn;

const VarianceOptions = struct {
    axis: ?isize = null,
    ddof: usize = 0,
    keepDims: bool = false,
    dtype: ?DType = null,
};

/// Computes the variance along the specified axis or over the entire array.
pub fn variance(
    arr: Array,
    options: VarianceOptions,
) (ShapeError || DTypeError || IndexError || std.mem.Allocator.Error)!Array {
    if (arr.elementCount() == 0) return ShapeError.EmptyArray;

    const out_dtype = options.dtype orelse if (arr.dtype.isFloat()) arr.dtype else .f64;

    var mu = try mean_fn(arr, .{
        .axis = options.axis,
        .keepDims = true,
        .dtype = out_dtype,
    });
    defer mu.deinit();

    var diff = try sub_fn(arr, mu, .{ .dtype = out_dtype });
    defer diff.deinit();

    var sq = try mul_fn(diff, diff, .{ .dtype = out_dtype });
    defer sq.deinit();

    var sum_sq = try sum_fn(sq, .{
        .axis = options.axis,
        .keepDims = options.keepDims,
        .dtype = out_dtype,
    });
    errdefer sum_sq.deinit();

    // Adjust for degrees of freedom (N - ddof)
    const count_dim: usize = if (options.axis) |ax| blk: {
        const norm_ax = try arr.shape().normalizeAxis(ax);
        break :blk arr.shape_dims[norm_ax];
    } else arr.elementCount();

    if (count_dim <= options.ddof) return ShapeError.InvalidDimension;
    const denom = @as(f64, @floatFromInt(count_dim - options.ddof));

    var denom_arr = try fromSlice(arr.allocator, f64, .{ .data = &[_]f64{denom}, .shape = &.{} });
    defer denom_arr.deinit();

    const div_fn = @import("../ops/elementwise.zig").divide;
    const res = try div_fn(sum_sq, denom_arr, .{ .dtype = out_dtype });
    sum_sq.deinit();
    return res;
}

const StdDevOptions = struct {
    axis: ?isize = null,
    ddof: usize = 0,
    keepDims: bool = false,
    dtype: ?DType = null,
};

/// Computes the standard deviation along the specified axis or over the entire array.
pub fn stdDev(
    arr: Array,
    options: StdDevOptions,
) (ShapeError || DTypeError || IndexError || std.mem.Allocator.Error)!Array {
    var var_res = try variance(arr, .{
        .axis = options.axis,
        .ddof = options.ddof,
        .keepDims = options.keepDims,
        .dtype = options.dtype,
    });
    defer var_res.deinit();

    return sqrt_fn(var_res, .{ .dtype = options.dtype });
}

/// Interpolation method for quantile/percentile evaluation.
pub const QuantileMethod = enum {
    linear,
    lower,
    higher,
    midpoint,
    nearest,
};

const MedianOptions = struct {
    axis: ?isize = null,
    keepDims: bool = false,
};

/// Computes the median value along the specified axis or over the entire array.
pub fn median(
    arr: Array,
    options: MedianOptions,
) (ShapeError || DTypeError || IndexError || std.mem.Allocator.Error)!Array {
    return quantileWithOptions(arr, 0.5, .{
        .axis = options.axis,
        .keepDims = options.keepDims,
    });
}

const QuantileOptions = struct {
    axis: ?isize = null,
    keepDims: bool = false,
    method: QuantileMethod = .linear,
};

/// Computes the q-th quantile (q in [0.0, 1.0]) along the specified axis or over the entire array.
pub fn quantile(
    arr: Array,
    q: f64,
) (ShapeError || DTypeError || IndexError || std.mem.Allocator.Error)!Array {
    return quantileWithOptions(arr, q, .{});
}

/// Computes the q-th quantile with configuration options.
pub fn quantileWithOptions(
    arr: Array,
    q: f64,
    options: QuantileOptions,
) (ShapeError || DTypeError || IndexError || std.mem.Allocator.Error)!Array {
    if (arr.elementCount() == 0) return ShapeError.EmptyArray;
    if (q < 0.0 or q > 1.0) return DTypeError.InvalidConversion;

    // Case 1: Global quantile
    if (options.axis == null) {
        var flat = try ravel(arr);
        defer flat.deinit();

        const N = flat.elementCount();
        const buffer = try arr.allocator.alloc(f64, N);
        defer arr.allocator.free(buffer);

        for (0..N) |i| {
            buffer[i] = try flat.get(f64, &.{i});
        }

        std.sort.pdq(f64, buffer, {}, std.sort.asc(f64));

        const val = interpolateQuantile(buffer, q, options.method);

        const kd_dims = [_]usize{1} ** MAX_RANK;
        const out_shape: []const usize = if (options.keepDims) kd_dims[0..arr.ndim] else &.{};

        var out = try empty(arr.allocator, .{
            .shape = out_shape,
            .dtype = .f64,
        });
        errdefer out.deinit();

        const zero_coords = [_]usize{0} ** MAX_RANK;
        try out.set(f64, if (options.keepDims) zero_coords[0..arr.ndim] else &.{}, val);
        return out;
    }

    // Case 2: Quantile along specified axis
    const axis = try arr.shape().normalizeAxis(options.axis.?);
    const in_shape = arr.shape();
    const axis_len = in_shape.dims[axis];

    var out_dims: [MAX_RANK]usize = undefined;
    var out_d: usize = 0;
    for (0..in_shape.ndim) |d| {
        if (d == axis) {
            if (options.keepDims) {
                out_dims[out_d] = 1;
                out_d += 1;
            }
        } else {
            out_dims[out_d] = in_shape.dims[d];
            out_d += 1;
        }
    }

    const target_out_ndim = out_d;
    var out = try zeros(arr.allocator, .{
        .shape = out_dims[0..target_out_ndim],
        .dtype = .f64,
    });
    errdefer out.deinit();

    const temp_buf = try arr.allocator.alloc(f64, axis_len);
    defer arr.allocator.free(temp_buf);

    const NdIterator = @import("../core/iterator.zig").NdIterator;
    var out_it = NdIterator.init(out.shape(), out.strides());
    while (out_it.next()) |item| {
        var in_coords: [MAX_RANK]usize = undefined;
        var in_d: usize = 0;
        for (0..in_shape.ndim) |d| {
            if (d == axis) {
                in_coords[d] = 0;
                if (options.keepDims) in_d += 1;
            } else {
                in_coords[d] = item.indices[in_d];
                in_d += 1;
            }
        }

        for (0..axis_len) |k| {
            in_coords[axis] = k;
            temp_buf[k] = try arr.get(f64, in_coords[0..in_shape.ndim]);
        }

        std.sort.pdq(f64, temp_buf, {}, std.sort.asc(f64));
        const val = interpolateQuantile(temp_buf, q, options.method);

        try out.set(f64, item.indices, val);
    }

    return out;
}

const PercentileOptions = struct {
    axis: ?isize = null,
    keepDims: bool = false,
    method: QuantileMethod = .linear,
};

/// Computes the q-th percentile (q in [0.0, 100.0]).
pub fn percentile(
    arr: Array,
    q: f64,
    options: PercentileOptions,
) (ShapeError || DTypeError || IndexError || std.mem.Allocator.Error)!Array {
    return quantileWithOptions(arr, q / 100.0, .{
        .axis = options.axis,
        .keepDims = options.keepDims,
        .method = options.method,
    });
}

/// Minimum over a given axis or globally. Direct alias of the shared reduction kernel.
pub const min = @import("../ops/reduce.zig").min;

/// Maximum over a given axis or globally. Direct alias of the shared reduction kernel.
pub const max = @import("../ops/reduce.zig").max;

/// Peak-to-peak range (max - min) over a given axis or globally.
/// Reuses the shared min/max reduction kernels and elementwise subtraction.
pub fn range(
    arr: Array,
    options: struct { axis: ?isize = null, keepDims: bool = false },
) (ShapeError || DTypeError || std.mem.Allocator.Error)!Array {
    const reduce_min = @import("../ops/reduce.zig").min;
    const reduce_max = @import("../ops/reduce.zig").max;
    const subtract_fn = @import("../ops/elementwise.zig").subtract;
    var lo = try reduce_min(arr, .{ .axis = options.axis, .keepDims = options.keepDims });
    defer lo.deinit();
    var hi = try reduce_max(arr, .{ .axis = options.axis, .keepDims = options.keepDims });
    defer hi.deinit();
    return subtract_fn(hi, lo, .{});
}

fn interpolateQuantile(sorted: []const f64, q: f64, method: QuantileMethod) f64 {
    const N = sorted.len;
    if (N == 1) return sorted[0];

    const idx_float = q * @as(f64, @floatFromInt(N - 1));
    const lower: usize = @intFromFloat(idx_float);
    const fraction = idx_float - @as(f64, @floatFromInt(lower));

    if (lower + 1 >= N) return sorted[N - 1];
    return switch (method) {
        .linear => sorted[lower] + fraction * (sorted[lower + 1] - sorted[lower]),
        .lower => sorted[lower],
        .higher => sorted[lower + 1],
        .midpoint => (sorted[lower] + sorted[lower + 1]) / 2.0,
        .nearest => if (fraction < 0.5) sorted[lower] else sorted[lower + 1],
    };
}

test "variance and stdDev" {
    const allocator = std.testing.allocator;
    const items = [_]f64{ 2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0 };
    var arr = try fromSlice(allocator, f64, .{ .data = &items, .shape = &.{8} });
    defer arr.deinit();

    // Population variance (ddof=0): sum((x-5)^2) / 8 = 32 / 8 = 4.0
    var v = try variance(arr, .{ .ddof = 0 });
    defer v.deinit();
    const v_val = try v.get(f64, &.{});
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), v_val, 1e-5);

    // Standard deviation: sqrt(4.0) = 2.0
    var s = try stdDev(arr, .{ .ddof = 0 });
    defer s.deinit();
    const s_val = try s.get(f64, &.{});
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), s_val, 1e-5);

    // Sample variance (ddof=1): 32 / 7 ~ 4.5714
    var v_sample = try variance(arr, .{ .ddof = 1 });
    defer v_sample.deinit();
    const vs_val = try v_sample.get(f64, &.{});
    try std.testing.expectApproxEqAbs(@as(f64, 4.57142857), vs_val, 1e-5);
}

test "median, quantile, and percentile" {
    const allocator = std.testing.allocator;
    const items = [_]f64{ 1.0, 3.0, 3.0, 6.0, 7.0, 8.0, 9.0 };
    var arr = try fromSlice(allocator, f64, .{ .data = &items, .shape = &.{7} });
    defer arr.deinit();

    // Median of {1, 3, 3, 6, 7, 8, 9} is 6.0
    var med = try median(arr, .{});
    defer med.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 6.0), try med.get(f64, &.{}), 1e-5);

    // 25th percentile: q = 0.25, idx = 0.25 * 6 = 1.5 -> 3 + 0.5 * (3 - 3) = 3.0
    var p25 = try percentile(arr, 25.0, .{});
    defer p25.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), try p25.get(f64, &.{}), 1e-5);

    // 75th percentile: q = 0.75, idx = 0.75 * 6 = 4.5 -> 7 + 0.5 * (8 - 7) = 7.5
    var p75 = try percentile(arr, 75.0, .{});
    defer p75.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 7.5), try p75.get(f64, &.{}), 1e-5);
}

test "quantile interpolation methods" {
    const allocator = std.testing.allocator;
    const items = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    var arr = try fromSlice(allocator, f64, .{ .data = &items, .shape = &.{4} });
    defer arr.deinit();
    // idx = 0.5 * 3 = 1.5 between sorted[1] = 2 and sorted[2] = 3
    var qlin = try quantileWithOptions(arr, 0.5, .{ .method = .linear });
    defer qlin.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 2.5), try qlin.get(f64, &.{}), 1e-12);
    var qlo = try quantileWithOptions(arr, 0.5, .{ .method = .lower });
    defer qlo.deinit();
    try std.testing.expectEqual(@as(f64, 2.0), try qlo.get(f64, &.{}));
    var qhi = try quantileWithOptions(arr, 0.5, .{ .method = .higher });
    defer qhi.deinit();
    try std.testing.expectEqual(@as(f64, 3.0), try qhi.get(f64, &.{}));
    var qmid = try quantileWithOptions(arr, 0.5, .{ .method = .midpoint });
    defer qmid.deinit();
    try std.testing.expectEqual(@as(f64, 2.5), try qmid.get(f64, &.{}));
    var qnear = try quantileWithOptions(arr, 0.5, .{ .method = .nearest });
    defer qnear.deinit();
    try std.testing.expectEqual(@as(f64, 3.0), try qnear.get(f64, &.{}));
}

test "stats min, max, and range" {
    const allocator = std.testing.allocator;
    const items = [_]f64{ 3.0, 1.0, 4.0, 1.0, 5.0, 9.0 };
    var arr = try fromSlice(allocator, f64, .{ .data = &items, .shape = &.{6} });
    defer arr.deinit();
    var lo = try min(arr, .{});
    defer lo.deinit();
    try std.testing.expectEqual(@as(f64, 1.0), try lo.get(f64, &.{}));
    var hi = try max(arr, .{});
    defer hi.deinit();
    try std.testing.expectEqual(@as(f64, 9.0), try hi.get(f64, &.{}));
    var r = try range(arr, .{});
    defer r.deinit();
    try std.testing.expectEqual(@as(f64, 8.0), try r.get(f64, &.{}));
}
