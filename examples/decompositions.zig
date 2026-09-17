//! Demonstrates matrix factorizations: LU, QR, and Cholesky decompositions.

const std = @import("std");
const num = @import("num");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // 1. QR Decomposition: A = Q * R
    const a_data = [_]f64{
        12.0, -51.0, 4.0,
        6.0,  167.0, -68.0,
        -4.0, 24.0,  -41.0,
    };
    var a = try num.fromSlice(allocator, f64, .{ .data = &a_data, .shape = &.{ 3, 3 } });
    defer a.deinit();

    var qr_res = try num.linalg.qr(a);
    defer qr_res.deinit();

    std.debug.print("QR Decomposition A = Q * R (3x3):\n", .{});
    std.debug.print("  Q shape: [{d}, {d}], R shape: [{d}, {d}]\n", .{
        qr_res.q.shape_dims[0],
        qr_res.q.shape_dims[1],
        qr_res.r.shape_dims[0],
        qr_res.r.shape_dims[1],
    });

    // 2. Cholesky Decomposition: symmetric positive-definite matrix A = L * L^T
    const spd_data = [_]f64{
        4.0,   12.0,  -16.0,
        12.0,  37.0,  -43.0,
        -16.0, -43.0, 98.0,
    };
    var spd = try num.fromSlice(allocator, f64, .{ .data = &spd_data, .shape = &.{ 3, 3 } });
    defer spd.deinit();

    var chol = try num.linalg.cholesky(spd);
    defer chol.deinit();

    std.debug.print("Cholesky Decomposition L:\n", .{});
    std.debug.print("  L[0,0] = {d:.1} (expected 2.0)\n", .{try chol.get(f64, &.{ 0, 0 })});
    std.debug.print("  L[1,0] = {d:.1} (expected 6.0)\n", .{try chol.get(f64, &.{ 1, 0 })});
    std.debug.print("  L[1,1] = {d:.1} (expected 1.0)\n", .{try chol.get(f64, &.{ 1, 1 })});

    // 3. LU Decomposition: P * A = L * U
    var lu_res = try num.linalg.lu(a);
    defer lu_res.deinit();
    std.debug.print("LU Decomposition:\n  L shape: [{d}, {d}], U shape: [{d}, {d}]\n", .{
        lu_res.l.shape_dims[0],
        lu_res.l.shape_dims[1],
        lu_res.u.shape_dims[0],
        lu_res.u.shape_dims[1],
    });
}
