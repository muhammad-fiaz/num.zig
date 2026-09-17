//! Vector and matrix norm calculations.
//!
//! Provides Frobenius, L1, L2, and Infinity norms over full arrays or specified axes.

const std = @import("std");
const Array = @import("../core/array.zig").Array;
const zeros = @import("../core/array.zig").zeros;
const empty = @import("../core/array.zig").empty;
const DType = @import("../core/dtype.zig").DType;
const Shape = @import("../core/shape.zig").Shape;
const MAX_RANK = @import("../core/shape.zig").MAX_RANK;
const ShapeError = @import("../core/error.zig").ShapeError;
const DTypeError = @import("../core/error.zig").DTypeError;
const NdIterator = @import("../core/iterator.zig").NdIterator;

pub const NormOrder = enum {
    frobenius,
    l1,
    l2,
    inf,
    neg_inf,
};

const NormOptions = struct {
    ord: NormOrder = .l2,
    axis: ?isize = null,
    keepDims: bool = false,
};

/// Computes the vector or matrix norm.
pub fn norm(
    arr: Array,
    options: NormOptions,
) (ShapeError || DTypeError || std.mem.Allocator.Error)!Array {
    if (arr.elementCount() == 0) return ShapeError.EmptyArray;

    const abs_fn = @import("../ops/elementwise.zig").abs;
    const pow_fn = @import("../ops/elementwise.zig").pow;
    const sum_fn = @import("../ops/reduce.zig").sum;
    const max_fn = @import("../ops/reduce.zig").max;
    const min_fn = @import("../ops/reduce.zig").min;
    const sqrt_fn = @import("../ops/elementwise.zig").sqrt;
    const fromSlice = @import("../core/array.zig").fromSlice;

    const float_dtype: DType = if (arr.dtype == .f32) .f32 else .f64;

    switch (options.ord) {
        .l2, .frobenius => {
            // sqrt(sum(|x|^2))
            const two_data = [_]f64{2.0};
            var two_arr = try fromSlice(arr.allocator, f64, .{ .data = &two_data, .shape = &.{} });
            defer two_arr.deinit();

            var abs_a = try abs_fn(arr, .{ .dtype = float_dtype });
            defer abs_a.deinit();

            var sq = try pow_fn(abs_a, two_arr, .{ .dtype = float_dtype });
            defer sq.deinit();

            var sum_sq = try sum_fn(sq, .{
                .axis = options.axis,
                .keepDims = options.keepDims,
                .dtype = float_dtype,
            });
            defer sum_sq.deinit();

            return sqrt_fn(sum_sq, .{ .dtype = float_dtype });
        },
        .l1 => {
            // sum(|x|)
            var abs_a = try abs_fn(arr, .{ .dtype = float_dtype });
            defer abs_a.deinit();

            return sum_fn(abs_a, .{
                .axis = options.axis,
                .keepDims = options.keepDims,
                .dtype = float_dtype,
            });
        },
        .inf => {
            // max(|x|)
            var abs_a = try abs_fn(arr, .{ .dtype = float_dtype });
            defer abs_a.deinit();

            return max_fn(abs_a, .{
                .axis = options.axis,
                .keepDims = options.keepDims,
                .dtype = float_dtype,
            });
        },
        .neg_inf => {
            // min(|x|)
            var abs_a = try abs_fn(arr, .{ .dtype = float_dtype });
            defer abs_a.deinit();

            return min_fn(abs_a, .{
                .axis = options.axis,
                .keepDims = options.keepDims,
                .dtype = float_dtype,
            });
        },
    }
}

test "vector and matrix norms" {
    const allocator = std.testing.allocator;
    const fromSlice = @import("../core/array.zig").fromSlice;

    // Vector [3, 4] -> L2 norm is 5.0
    const v_data = [_]f64{ 3.0, 4.0 };
    var v = try fromSlice(allocator, f64, .{ .data = &v_data, .shape = &.{2} });
    defer v.deinit();

    var n_l2 = try norm(v, .{ .ord = .l2 });
    defer n_l2.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), try n_l2.get(f64, &.{}), 1e-6);

    var n_l1 = try norm(v, .{ .ord = .l1 });
    defer n_l1.deinit();
    try std.testing.expectEqual(@as(f64, 7.0), try n_l1.get(f64, &.{}));

    var n_inf = try norm(v, .{ .ord = .inf });
    defer n_inf.deinit();
    try std.testing.expectEqual(@as(f64, 4.0), try n_inf.get(f64, &.{}));
}
