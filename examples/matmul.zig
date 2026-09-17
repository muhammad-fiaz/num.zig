//! Demonstrates matrix multiplication, vector dot products, and outer products.

const std = @import("std");
const num = @import("num");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // Matrix A (2x3)
    const a_data = [_]f64{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    };
    var a = try num.fromSlice(allocator, f64, .{ .data = &a_data, .shape = &.{ 2, 3 } });
    defer a.deinit();

    // Matrix B (3x2)
    const b_data = [_]f64{
        7.0, 8.0,
        9.0, 1.0,
        2.0, 3.0,
    };
    var b = try num.fromSlice(allocator, f64, .{ .data = &b_data, .shape = &.{ 3, 2 } });
    defer b.deinit();

    // Matrix multiply: (2, 3) x (3, 2) -> (2, 2)
    var c = try num.linalg.matmul(a, b, .{});
    defer c.deinit();

    std.debug.print("Matrix Multiplication C = A x B (2x2):\n", .{});
    std.debug.print("  [{d:.1}, {d:.1}]\n", .{ try c.get(f64, &.{ 0, 0 }), try c.get(f64, &.{ 0, 1 }) });
    std.debug.print("  [{d:.1}, {d:.1}]\n", .{ try c.get(f64, &.{ 1, 0 }), try c.get(f64, &.{ 1, 1 }) });

    // Vector Dot Product
    const u_data = [_]f64{ 1.0, 3.0, -5.0 };
    const v_data = [_]f64{ 4.0, -2.0, -1.0 };
    var u = try num.fromSlice(allocator, f64, .{ .data = &u_data, .shape = &.{3} });
    defer u.deinit();
    var v = try num.fromSlice(allocator, f64, .{ .data = &v_data, .shape = &.{3} });
    defer v.deinit();

    var d = try num.linalg.dot(u, v, .{});
    defer d.deinit();
    std.debug.print("Vector dot product u . v: {d:.1}\n", .{try d.get(f64, &.{})});

    // Outer Product: (3,) x (3,) -> (3, 3)
    var out_prod = try num.linalg.outer(u, v, .{});
    defer out_prod.deinit();
    std.debug.print("Outer product shape: [{d}, {d}], val[0, 0]: {d:.1}\n", .{
        out_prod.shape_dims[0],
        out_prod.shape_dims[1],
        try out_prod.get(f64, &.{ 0, 0 }),
    });
}
