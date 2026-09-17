//! Demonstrates matrix and tensor transposition, axis swapping, and axis moving.

const std = @import("std");
const num = @import("num");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    const data = [_]f64{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    };
    var mat = try num.fromSlice(allocator, f64, .{ .data = &data, .shape = &.{ 2, 3 } });
    defer mat.deinit();

    // Standard 2D transpose (2, 3) -> (3, 2)
    var t = try num.manip.transpose(mat, .{});
    defer t.deinit();

    std.debug.print("Matrix Transpose (2x3) -> (3x2):\n", .{});
    for (0..3) |r| {
        std.debug.print("  row {d}: [{d:.1}, {d:.1}]\n", .{
            r,
            try t.get(f64, &.{ r, 0 }),
            try t.get(f64, &.{ r, 1 }),
        });
    }
    std.debug.print("  Transposed view is contiguous: {}\n", .{t.isContiguous()});

    // 3D swapAxes (2, 3, 4) -> swap axes 0 and 2 -> (4, 3, 2)
    var tensor = try num.zeros(allocator, .{ .shape = &.{ 2, 3, 4 }, .dtype = .f32 });
    defer tensor.deinit();

    var swapped = try num.manip.swapAxes(tensor, 0, 2);
    defer swapped.deinit();
    std.debug.print("SwapAxes (0, 2) on (2, 3, 4) -> shape: [{d}, {d}, {d}]\n", .{
        swapped.shape_dims[0],
        swapped.shape_dims[1],
        swapped.shape_dims[2],
    });
}
