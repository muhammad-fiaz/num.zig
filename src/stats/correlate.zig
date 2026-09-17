//! Correlation, covariance, and histogram statistics.
//!
//! Provides covariance matrices, Pearson correlation coefficients, and 1D histograms.

const std = @import("std");
const Array = @import("../core/array.zig").Array;
const zeros = @import("../core/array.zig").zeros;
const empty = @import("../core/array.zig").empty;
const fromSlice = @import("../core/array.zig").fromSlice;
const DType = @import("../core/dtype.zig").DType;
const Shape = @import("../core/shape.zig").Shape;
const ShapeError = @import("../core/error.zig").ShapeError;
const DTypeError = @import("../core/error.zig").DTypeError;
const IndexError = @import("../core/error.zig").IndexError;
const LinalgError = @import("../core/error.zig").LinalgError;

const matmul = @import("../linalg/matmul.zig").matmul;
const mean_fn = @import("../ops/reduce.zig").mean;
const min_fn = @import("../ops/reduce.zig").min;
const max_fn = @import("../ops/reduce.zig").max;
const sub_fn = @import("../ops/elementwise.zig").subtract;
const ravel = @import("../manip/reshape.zig").ravel;
const transpose = @import("../manip/transpose.zig").transpose;

pub const CovarianceOptions = struct {
    rowvar: bool = true,
    bias: bool = false,
    ddof: ?usize = null,
};

/// Computes the covariance matrix of an array or pair of arrays.
pub fn covariance(
    x: Array,
    y: ?Array,
    options: CovarianceOptions,
) (ShapeError || DTypeError || IndexError || LinalgError || std.mem.Allocator.Error)!Array {
    const expandDims = @import("../manip/reshape.zig").expandDims;
    const concat = @import("../manip/concat.zig").concat;

    // Prepare 2D matrix where each row is a variable, each col is an observation
    var mat_2d: Array = undefined;
    var mat_allocated = false;
    defer if (mat_allocated) mat_2d.deinit();

    if (y) |y_arr| {
        // x and y must both be 1D
        if (x.ndim != 1 or y_arr.ndim != 1) return ShapeError.InvalidDimension;
        if (x.shape_dims[0] != y_arr.shape_dims[0]) return ShapeError.IncompatibleShapes;

        const x_2d = try expandDims(x, .{ .axis = 0 }); // (1, N)
        const y_2d = try expandDims(y_arr, .{ .axis = 0 }); // (1, N)

        const arrays = [_]Array{ x_2d, y_2d };
        mat_2d = try concat(&arrays, .{ .axis = 0 });
        mat_allocated = true;
    } else {
        if (x.ndim == 1) {
            mat_2d = try expandDims(x, .{ .axis = 0 });
        } else if (x.ndim == 2) {
            if (options.rowvar) {
                mat_2d = x;
            } else {
                mat_2d = try transpose(x, .{});
                mat_allocated = true;
            }
        } else {
            return ShapeError.InvalidDimension;
        }
    }

    const n_obs = mat_2d.shape_dims[1];

    if (n_obs == 0) return ShapeError.EmptyArray;

    const ddof: usize = options.ddof orelse (if (options.bias) 0 else 1);
    if (n_obs <= ddof) return DTypeError.InvalidConversion;

    // 1. Mean-center along rows (observations)
    var row_means = try mean_fn(mat_2d, .{ .axis = 1, .keepDims = true, .dtype = .f64 });
    defer row_means.deinit();

    var centered = try sub_fn(mat_2d, row_means, .{ .dtype = .f64 });
    defer centered.deinit();

    // 2. Transpose centered
    var centered_t = try transpose(centered, .{});
    defer centered_t.deinit();

    // 3. Matrix product: C = centered * centered_t (shape: [n_vars, n_vars])
    var cov_mat = try matmul(centered, centered_t, .{});
    errdefer cov_mat.deinit();

    // 4. Divide by (n_obs - ddof)
    const factor: f64 = 1.0 / @as(f64, @floatFromInt(n_obs - ddof));
    const factor_slice = [_]f64{factor};
    var factor_arr = try fromSlice(x.allocator, f64, .{ .data = &factor_slice, .shape = &.{} });
    defer factor_arr.deinit();

    const mul_fn = @import("../ops/elementwise.zig").multiply;
    const scaled_cov = try mul_fn(cov_mat, factor_arr, .{ .dtype = .f64 });
    cov_mat.deinit();
    return scaled_cov;
}

/// Computes the Pearson correlation coefficient matrix.
pub fn corrcoef(
    x: Array,
    y: ?Array,
    options: CovarianceOptions,
) (ShapeError || DTypeError || IndexError || LinalgError || std.mem.Allocator.Error)!Array {
    var cov = try covariance(x, y, options);
    defer cov.deinit();

    const n = cov.shape_dims[0];
    var r = try zeros(x.allocator, .{ .shape = &.{ n, n }, .dtype = .f64 });
    errdefer r.deinit();

    // Diagonals: std_dev_i = sqrt(cov[i, i])
    const stds = try x.allocator.alloc(f64, n);
    defer x.allocator.free(stds);

    for (0..n) |i| {
        const var_i = try cov.get(f64, &.{ i, i });
        stds[i] = if (var_i > 0) std.math.sqrt(var_i) else 0;
    }

    for (0..n) |i| {
        for (0..n) |j| {
            if (i == j) {
                try r.set(f64, &.{ i, j }, 1.0);
            } else {
                const cov_ij = try cov.get(f64, &.{ i, j });
                const denom = stds[i] * stds[j];
                const corr_val = if (denom > 1e-15) cov_ij / denom else 0.0;
                try r.set(f64, &.{ i, j }, corr_val);
            }
        }
    }

    return r;
}

pub const HistogramResult = struct {
    counts: Array,
    bin_edges: Array,

    pub fn deinit(self: *HistogramResult) void {
        self.counts.deinit();
        self.bin_edges.deinit();
    }
};

pub const HistogramOptions = struct {
    bins: usize = 10,
    range: ?[2]f64 = null,
    density: bool = false,
};

/// Computes the histogram of a set of data.
pub fn histogram(
    arr: Array,
    options: HistogramOptions,
) (ShapeError || DTypeError || IndexError || std.mem.Allocator.Error)!HistogramResult {
    if (arr.elementCount() == 0) return ShapeError.EmptyArray;
    if (options.bins == 0) return ShapeError.InvalidDimension;

    var flat = try ravel(arr);
    defer flat.deinit();

    const n_elem = flat.elementCount();

    var min_v: f64 = undefined;
    var max_v: f64 = undefined;

    if (options.range) |rng| {
        min_v = rng[0];
        max_v = rng[1];
    } else {
        var min_a = try min_fn(flat, .{});
        defer min_a.deinit();
        var max_a = try max_fn(flat, .{});
        defer max_a.deinit();

        min_v = try min_a.get(f64, &.{});
        max_v = try max_a.get(f64, &.{});
    }

    if (max_v < min_v) return DTypeError.InvalidConversion;
    if (max_v == min_v) {
        max_v += 1.0;
    }

    const n_bins = options.bins;
    const bin_width = (max_v - min_v) / @as(f64, @floatFromInt(n_bins));

    // Construct bin edges: (bins + 1)
    var bin_edges = try empty(arr.allocator, .{ .shape = &.{n_bins + 1}, .dtype = .f64 });
    errdefer bin_edges.deinit();

    const edges_slice = bin_edges.asSlice(f64) catch unreachable;
    for (0..n_bins + 1) |b| {
        edges_slice[b] = min_v + @as(f64, @floatFromInt(b)) * bin_width;
    }

    // Counts array: (bins)
    const count_dtype: DType = if (options.density) .f64 else .i64;
    var counts = try zeros(arr.allocator, .{ .shape = &.{n_bins}, .dtype = count_dtype });
    errdefer counts.deinit();

    // Bin accumulation
    const raw_counts = try arr.allocator.alloc(usize, n_bins);
    defer arr.allocator.free(raw_counts);
    @memset(raw_counts, 0);

    for (0..n_elem) |i| {
        const val = try flat.get(f64, &.{i});
        if (val < min_v or val > max_v) continue;

        var bin_idx: usize = @intFromFloat((val - min_v) / bin_width);
        if (bin_idx >= n_bins) bin_idx = n_bins - 1; // Upper bound inclusion
        raw_counts[bin_idx] += 1;
    }

    if (options.density) {
        const total_count: f64 = @floatFromInt(n_elem);
        const c_slice = counts.asSlice(f64) catch unreachable;
        for (0..n_bins) |b| {
            c_slice[b] = @as(f64, @floatFromInt(raw_counts[b])) / (total_count * bin_width);
        }
    } else {
        const c_slice = counts.asSlice(i64) catch unreachable;
        for (0..n_bins) |b| {
            c_slice[b] = @intCast(raw_counts[b]);
        }
    }

    return HistogramResult{
        .counts = counts,
        .bin_edges = bin_edges,
    };
}

test "covariance and correlation coefficient" {
    const allocator = std.testing.allocator;
    const x_vals = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const y_vals = [_]f64{ 2.0, 4.0, 6.0, 8.0, 10.0 };

    var x = try fromSlice(allocator, f64, .{ .data = &x_vals, .shape = &.{5} });
    defer x.deinit();

    var y = try fromSlice(allocator, f64, .{ .data = &y_vals, .shape = &.{5} });
    defer y.deinit();

    // Perfect positive correlation
    var r = try corrcoef(x, y, .{});
    defer r.deinit();

    try std.testing.expectEqual(@as(usize, 2), r.ndim);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), try r.get(f64, &.{ 0, 0 }), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), try r.get(f64, &.{ 0, 1 }), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), try r.get(f64, &.{ 1, 0 }), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), try r.get(f64, &.{ 1, 1 }), 1e-5);
}

test "histogram counts and bin edges" {
    const allocator = std.testing.allocator;
    const data = [_]f64{ 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0 };
    var arr = try fromSlice(allocator, f64, .{ .data = &data, .shape = &.{7} });
    defer arr.deinit();

    var hist = try histogram(arr, .{ .bins = 3, .range = .{ 1.0, 4.0 } });
    defer hist.deinit();

    try std.testing.expectEqual(@as(usize, 3), hist.counts.elementCount());
    try std.testing.expectEqual(@as(usize, 4), hist.bin_edges.elementCount());

    const c0 = try hist.counts.get(i64, &.{0});
    const c1 = try hist.counts.get(i64, &.{1});
    const c2 = try hist.counts.get(i64, &.{2});

    // Total elements accounted for
    try std.testing.expectEqual(@as(i64, 7), c0 + c1 + c2);
}
