//! Demonstrates vector and matrix norm computations in num.zig.

const std = @import("std");
const num = @import("num");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // 3D vector: [3, 4, 0] -> Euclidean length should be 5.0
    const v_data = [_]f64{ 3.0, 4.0, 0.0 };
    var v = try num.fromSlice(allocator, f64, .{ .data = &v_data, .shape = &.{3} });
    defer v.deinit();

    // Vector L2 norm
    var l2 = try num.linalg.norm(v, .{ .ord = .l2 });
    defer l2.deinit();

    // Vector L1 norm (|3| + |4| + |0| = 7.0)
    var l1 = try num.linalg.norm(v, .{ .ord = .l1 });
    defer l1.deinit();

    // Vector Linf norm (max(|3|, |4|, |0|) = 4.0)
    var linf = try num.linalg.norm(v, .{ .ord = .inf });
    defer linf.deinit();

    std.debug.print("Vector Norms of [3.0, 4.0, 0.0]:\n", .{});
    std.debug.print("  L2 norm:   {d:.1}\n", .{try l2.get(f64, &.{})});
    std.debug.print("  L1 norm:   {d:.1}\n", .{try l1.get(f64, &.{})});
    std.debug.print("  Linf norm: {d:.1}\n", .{try linf.get(f64, &.{})});

    // Matrix Frobenius norm of [[1, 2], [3, 4]]
    const m_data = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    var m = try num.fromSlice(allocator, f64, .{ .data = &m_data, .shape = &.{ 2, 2 } });
    defer m.deinit();

    var frob = try num.linalg.norm(m, .{ .ord = .frobenius });
    defer frob.deinit();
    std.debug.print("Matrix Frobenius norm: {d:.4}\n", .{try frob.get(f64, &.{})});
}
