const std = @import("std");
const num = @import("num");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // 3x2 Matrix
    const data = [_]f64{
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
    };
    var a = try num.fromSlice(allocator, f64, .{ .data = &data, .shape = &.{ 3, 2 } });
    defer a.deinit();

    std.debug.print("Input Matrix (3x2):\n", .{});
    std.debug.print("  [ {d:.1}, {d:.1} ]\n", .{ try a.get(f64, &.{ 0, 0 }), try a.get(f64, &.{ 0, 1 }) });
    std.debug.print("  [ {d:.1}, {d:.1} ]\n", .{ try a.get(f64, &.{ 1, 0 }), try a.get(f64, &.{ 1, 1 }) });
    std.debug.print("  [ {d:.1}, {d:.1} ]\n", .{ try a.get(f64, &.{ 2, 0 }), try a.get(f64, &.{ 2, 1 }) });

    // Perform SVD: A = U * S * Vt
    var res = try num.linalg.svd(a, .{ .full_matrices = true });
    defer res.deinit();

    std.debug.print("\nSingular values (S):\n", .{});
    for (0..res.s.elementCount()) |i| {
        std.debug.print("  sigma {d}: {d:.4}\n", .{ i, try res.s.get(f64, &.{i}) });
    }

    std.debug.print("\nLeft singular vectors U (3x3):\n", .{});
    std.debug.print("  shape: [{d}, {d}]\n", .{ res.u.shape_dims[0], res.u.shape_dims[1] });

    std.debug.print("Right singular vectors Vt (2x2):\n", .{});
    std.debug.print("  shape: [{d}, {d}]\n", .{ res.vt.shape_dims[0], res.vt.shape_dims[1] });
}
