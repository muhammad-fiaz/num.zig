const std = @import("std");
const num = @import("num");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // Construct a 2x2 matrix:
    // [ 4.0,  1.0 ]
    // [ 2.0,  3.0 ]
    // Eigenvalues should be 5.0 and 2.0.
    const data = [_]f64{ 4.0, 1.0, 2.0, 3.0 };
    var mat = try num.fromSlice(allocator, f64, .{ .data = &data, .shape = &.{ 2, 2 } });
    defer mat.deinit();

    std.debug.print("Original 2x2 Matrix:\n", .{});
    std.debug.print("  [ {d:.1}, {d:.1} ]\n", .{ try mat.get(f64, &.{ 0, 0 }), try mat.get(f64, &.{ 0, 1 }) });
    std.debug.print("  [ {d:.1}, {d:.1} ]\n", .{ try mat.get(f64, &.{ 1, 0 }), try mat.get(f64, &.{ 1, 1 }) });

    // 1. Eigenvalues only
    var vals = try num.linalg.eigvals(mat);
    defer vals.deinit();

    std.debug.print("\nEigenvalues:\n", .{});
    std.debug.print("  lambda 0: {d:.4}\n", .{try vals.get(f64, &.{0})});
    std.debug.print("  lambda 1: {d:.4}\n", .{try vals.get(f64, &.{1})});

    // 2. Eigenpairs (values + vectors)
    var result = try num.linalg.eig(mat, .{ .compute_vectors = true });
    defer result.deinit();

    if (result.vectors) |vecs| {
        std.debug.print("\nEigenvectors (columns):\n", .{});
        std.debug.print("  v0: [ {d:.4}, {d:.4} ]\n", .{ try vecs.get(f64, &.{ 0, 0 }), try vecs.get(f64, &.{ 1, 0 }) });
        std.debug.print("  v1: [ {d:.4}, {d:.4} ]\n", .{ try vecs.get(f64, &.{ 0, 1 }), try vecs.get(f64, &.{ 1, 1 }) });
    }
}
