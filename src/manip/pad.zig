//! Multidimensional array padding.
//!
//! Provides array padding with constant values across all dimensions.

const std = @import("std");
const Array = @import("../core/array.zig").Array;
const full = @import("../core/array.zig").full;
const Shape = @import("../core/shape.zig").Shape;
const MAX_RANK = @import("../core/shape.zig").MAX_RANK;
const ShapeError = @import("../core/error.zig").ShapeError;
const DType = @import("../core/dtype.zig").DType;
const NdIterator = @import("../core/iterator.zig").NdIterator;

/// Pads an array with a constant value according to `pad_width` [before, after] per dimension.
pub fn pad(
    arr: Array,
    options: anytype,
) (ShapeError || std.mem.Allocator.Error)!Array {
    const s = arr.shape();
    if (options.pad_width.len != s.ndim) return ShapeError.RankExceeded;

    const const_val = if (@hasField(@TypeOf(options), "constant_value")) options.constant_value else 0;

    var before_pads: [MAX_RANK]usize = undefined;
    var out_dims: [MAX_RANK]usize = undefined;
    inline for (0..MAX_RANK) |i| {
        if (i < options.pad_width.len) {
            before_pads[i] = options.pad_width[i][0];
            out_dims[i] = s.dims[i] + options.pad_width[i][0] + options.pad_width[i][1];
        }
    }

    const out_shape = Shape{ .dims = out_dims, .ndim = s.ndim };
    var out = try full(arr.allocator, .{
        .shape = out_shape.slice(),
        .value = const_val,
        .dtype = arr.dtype,
    });

    inline for (std.meta.fields(DType)) |field| {
        const tag: DType = @enumFromInt(field.value);
        if (arr.dtype == tag) {
            const T = tag.toType();

            var in_it = NdIterator.init(s, arr.strides());
            while (in_it.next()) |in_item| {
                var out_indices: [MAX_RANK]usize = undefined;
                for (0..s.ndim) |dim| {
                    out_indices[dim] = in_item.indices[dim] + before_pads[dim];
                }
                const val = arr.get(T, in_item.indices) catch unreachable;
                out.set(T, out_indices[0..s.ndim], val) catch unreachable;
            }

            return out;
        }
    }

    return out;
}

test "array constant padding" {
    const allocator = std.testing.allocator;
    const fromSlice = @import("../core/array.zig").fromSlice;

    const data = [_]f32{ 1, 2, 3, 4 };
    var a = try fromSlice(allocator, f32, .{ .data = &data, .shape = &.{ 2, 2 } });
    defer a.deinit();

    // Pad with 1 row before, 1 row after; 1 col before, 1 col after with constant 0
    var p = try pad(a, .{
        .pad_width = &.{ .{ 1, 1 }, .{ 1, 1 } },
        .constant_value = 0.0,
    });
    defer p.deinit();

    try std.testing.expectEqualSlices(usize, &.{ 4, 4 }, p.shapeSlice());
    try std.testing.expectEqual(@as(f32, 0.0), try p.get(f32, &.{ 0, 0 }));
    try std.testing.expectEqual(@as(f32, 1.0), try p.get(f32, &.{ 1, 1 }));
    try std.testing.expectEqual(@as(f32, 2.0), try p.get(f32, &.{ 1, 2 }));
    try std.testing.expectEqual(@as(f32, 4.0), try p.get(f32, &.{ 2, 2 }));
    try std.testing.expectEqual(@as(f32, 0.0), try p.get(f32, &.{ 3, 3 }));
}
