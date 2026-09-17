//! Multidimensional array padding.
//!
//! Provides array padding with constant, edge-clamped, and reflected values
//! across all dimensions.

const std = @import("std");
const Array = @import("../core/array.zig").Array;
const full = @import("../core/array.zig").full;
const Shape = @import("../core/shape.zig").Shape;
const MAX_RANK = @import("../core/shape.zig").MAX_RANK;
const ShapeError = @import("../core/error.zig").ShapeError;
const DType = @import("../core/dtype.zig").DType;
const NdIterator = @import("../core/iterator.zig").NdIterator;

/// Padding mode selecting how out-of-bounds elements are filled.
pub const PadMode = enum {
    /// Fill with a constant scalar value.
    constant,
    /// Replicate the nearest edge element (clamp).
    edge,
    /// Mirror about the edge without repeating it: d c b a | a b c d | d c b a.
    reflect,
};

/// Pads an array according to `pad_width` [before, after] per dimension.
/// Modes: `.constant` fills with `constant_value`; `.edge` clamps to the
/// nearest element; `.reflect` mirrors about the edge element.
pub fn pad(
    arr: Array,
    options: anytype,
) (ShapeError || std.mem.Allocator.Error)!Array {
    const s = arr.shape();
    if (options.pad_width.len != s.ndim) return ShapeError.RankExceeded;

    const OptType = @TypeOf(options);
    const mode: PadMode = if (@hasField(OptType, "mode")) options.mode else .constant;
    const const_val = if (@hasField(OptType, "constant_value")) options.constant_value else 0;

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

            if (mode == .constant) {
                var in_it = NdIterator.init(s, arr.strides());
                while (in_it.next()) |in_item| {
                    var out_indices: [MAX_RANK]usize = undefined;
                    for (0..s.ndim) |dim| {
                        out_indices[dim] = in_item.indices[dim] + before_pads[dim];
                    }
                    const val = arr.get(T, in_item.indices) catch unreachable;
                    out.set(T, out_indices[0..s.ndim], val) catch unreachable;
                }
            } else {
                var out_it = NdIterator.init(out_shape, out.strides());
                while (out_it.next()) |out_item| {
                    var src_indices: [MAX_RANK]usize = undefined;
                    for (0..s.ndim) |dim| {
                        const dim_len = s.dims[dim];
                        const rel: isize = @as(isize, @intCast(out_item.indices[dim])) - @as(isize, @intCast(before_pads[dim]));
                        src_indices[dim] = mapPadIndex(rel, dim_len, mode);
                    }
                    const val = arr.get(T, src_indices[0..s.ndim]) catch unreachable;
                    out.set(T, out_item.indices[0..s.ndim], val) catch unreachable;
                }
            }

            return out;
        }
    }

    return out;
}

/// Maps an output coordinate (relative to the unpadded array) to a source index.
fn mapPadIndex(rel: isize, dim_len: usize, mode: PadMode) usize {
    const len: isize = @intCast(dim_len);
    switch (mode) {
        .constant => return 0, // Unused; constant regions keep the fill value.
        .edge => {
            if (rel < 0) return 0;
            if (rel >= len) return dim_len - 1;
            return @intCast(rel);
        },
        .reflect => {
            if (len <= 1) return 0;
            const period = 2 * (len - 1);
            var r = @mod(rel, period);
            if (r < 0) r += period;
            if (r >= len) r = period - r;
            return @intCast(r);
        },
    }
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

test "edge and reflect padding modes" {
    const allocator = std.testing.allocator;
    const fromSlice = @import("../core/array.zig").fromSlice;

    const data = [_]f32{ 1, 2, 3 };
    var a = try fromSlice(allocator, f32, .{ .data = &data, .shape = &.{3} });
    defer a.deinit();

    // Edge: [1, 1, 2, 3, 3]
    var e = try pad(a, .{ .pad_width = &.{.{ 1, 1 }}, .mode = PadMode.edge });
    defer e.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 1, 2, 3, 3 }, try e.asSlice(f32));

    // Reflect: [2, 1, 2, 3, 2]
    var r = try pad(a, .{ .pad_width = &.{.{ 1, 1 }}, .mode = PadMode.reflect });
    defer r.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 2, 1, 2, 3, 2 }, try r.asSlice(f32));
}
