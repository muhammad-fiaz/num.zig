//! Demonstrates linear system solving, matrix inversion, and determinant computation.

const std = @import("std");
const num = @import("num");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // System:
    // 3x + y = 9
    //  x + 2y = 8
    // Solution: x = 2, y = 3
    const a_data = [_]f64{
        3.0, 1.0,
        1.0, 2.0,
    };
    var a = try num.fromSlice(allocator, f64, .{ .data = &a_data, .shape = &.{ 2, 2 } });
    defer a.deinit();

    const b_data = [_]f64{ 9.0, 8.0 };
    var b = try num.fromSlice(allocator, f64, .{ .data = &b_data, .shape = &.{2} });
    defer b.deinit();

    // 1. Solve A x = b
    var x = try num.linalg.solve(a, b);
    defer x.deinit();

    std.debug.print("Solved System Ax = b:\n", .{});
    std.debug.print("  x[0] = {d:.4} (expected 2.0)\n", .{try x.get(f64, &.{0})});
    std.debug.print("  x[1] = {d:.4} (expected 3.0)\n", .{try x.get(f64, &.{1})});

    // 2. Determinant of A: 3*2 - 1*1 = 5.0
    var det_a = try num.linalg.det(a);
    defer det_a.deinit();
    std.debug.print("Determinant det(A): {d:.4} (expected 5.0)\n", .{try det_a.get(f64, &.{})});

    // 3. Inverse of A
    var inv_a = try num.linalg.inv(a);
    defer inv_a.deinit();
    std.debug.print("Inverse A^-1:\n", .{});
    std.debug.print("  [{d:.4}, {d:.4}]\n", .{ try inv_a.get(f64, &.{ 0, 0 }), try inv_a.get(f64, &.{ 0, 1 }) });
    std.debug.print("  [{d:.4}, {d:.4}]\n", .{ try inv_a.get(f64, &.{ 1, 0 }), try inv_a.get(f64, &.{ 1, 1 }) });

    // 4. Slogdet
    var sl = try num.linalg.slogdet(a);
    defer sl.deinit();
    std.debug.print("slogdet sign: {d:.1}, logabsdet: {d:.4}\n", .{
        try sl.sign.get(f64, &.{}),
        try sl.logabsdet.get(f64, &.{}),
    });
}
