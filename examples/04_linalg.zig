const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== Linear Algebra Example ===\n\n", .{});

    // Matmul
    var a = try num.NDArray(f32).init(allocator, &.{ 2, 3 });
    defer a.deinit(allocator);
    a.fill(1.0);

    var b = try num.NDArray(f32).init(allocator, &.{ 3, 2 });
    defer b.deinit(allocator);
    b.fill(2.0);

    var c = try num.linalg.matmul(f32, allocator, &a, &b);
    defer c.deinit(allocator);

    std.debug.print("Matmul (2x3 * 3x2 = 2x2):\n", .{});
    for (0..c.shape[0]) |i| {
        for (0..c.shape[1]) |j| {
            std.debug.print("{d:.1} ", .{try c.get(&.{ i, j })});
        }
        std.debug.print("\n", .{});
    }

    // Solve
    // 2x + y = 5
    // x + y = 3
    var A = try num.NDArray(f32).init(allocator, &.{ 2, 2 });
    defer A.deinit(allocator);
    try A.set(&.{ 0, 0 }, 2);
    try A.set(&.{ 0, 1 }, 1);
    try A.set(&.{ 1, 0 }, 1);
    try A.set(&.{ 1, 1 }, 1);

    var B = try num.NDArray(f32).init(allocator, &.{2});
    defer B.deinit(allocator);
    try B.set(&.{0}, 5);
    try B.set(&.{1}, 3);

    var x = try num.linalg.solve(f32, allocator, &A, &B);
    defer x.deinit(allocator);

    std.debug.print("\nSolve Ax=B:\n", .{});
    std.debug.print("x: {d:.2}, y: {d:.2}\n", .{ try x.get(&.{0}), try x.get(&.{1}) });

    // Norm
    const n = try num.linalg.norm(f32, allocator, &x);
    std.debug.print("\nL2 Norm of solution: {d:.4}\n", .{n});

    // Cholesky
    std.debug.print("\n--- Cholesky Decomposition ---\n", .{});
    // Matrix must be positive definite.
    // [ 4  12 -16 ]
    // [ 12 37 -43 ]
    // [ -16 -43 98 ]
    var P = try num.NDArray(f32).init(allocator, &.{ 3, 3 });
    defer P.deinit(allocator);
    try P.set(&.{ 0, 0 }, 4);
    try P.set(&.{ 0, 1 }, 12);
    try P.set(&.{ 0, 2 }, -16);
    try P.set(&.{ 1, 0 }, 12);
    try P.set(&.{ 1, 1 }, 37);
    try P.set(&.{ 1, 2 }, -43);
    try P.set(&.{ 2, 0 }, -16);
    try P.set(&.{ 2, 1 }, -43);
    try P.set(&.{ 2, 2 }, 98);

    var L = try num.linalg.cholesky(f32, allocator, &P);
    defer L.deinit(allocator);

    std.debug.print("Lower Triangular L:\n", .{});
    for (0..L.shape[0]) |i| {
        for (0..L.shape[1]) |j| {
            std.debug.print("{d:.1} ", .{try L.get(&.{ i, j })});
        }
        std.debug.print("\n", .{});
    }

    // QR Decomposition
    std.debug.print("\n--- QR Decomposition ---\n", .{});
    var qr_res = try num.linalg.qr(f32, allocator, &P);
    defer qr_res.q.deinit(allocator);
    defer qr_res.r.deinit(allocator);

    std.debug.print("Q (Orthogonal):\n", .{});
    for (0..qr_res.q.shape[0]) |i| {
        for (0..qr_res.q.shape[1]) |j| {
            std.debug.print("{d:.2} ", .{try qr_res.q.get(&.{ i, j })});
        }
        std.debug.print("\n", .{});
    }

    std.debug.print("R (Upper Triangular):\n", .{});
    for (0..qr_res.r.shape[0]) |i| {
        for (0..qr_res.r.shape[1]) |j| {
            std.debug.print("{d:.2} ", .{try qr_res.r.get(&.{ i, j })});
        }
        std.debug.print("\n", .{});
    }

    // SVD
    std.debug.print("\n--- SVD ---\n", .{});
    var svd_res = try num.linalg.svd(f32, allocator, &P, 100);
    defer svd_res.u.deinit(allocator);
    defer svd_res.s.deinit(allocator);
    defer svd_res.vt.deinit(allocator);

    std.debug.print("Singular Values:\n", .{});
    for (0..svd_res.s.shape[0]) |i| {
        std.debug.print("{d:.4} ", .{try svd_res.s.get(&.{i})});
    }
    std.debug.print("\n", .{});

    // Eigenvalues
    std.debug.print("\n--- Eigenvalues ---\n", .{});
    // Use a simple symmetric matrix for real eigenvalues
    var S = try num.NDArray(f32).init(allocator, &.{ 2, 2 });
    defer S.deinit(allocator);
    try S.set(&.{ 0, 0 }, 1);
    try S.set(&.{ 0, 1 }, 2);
    try S.set(&.{ 1, 0 }, 2);
    try S.set(&.{ 1, 1 }, 1);

    var eig = try num.linalg.eigvals(f32, allocator, &S, 20);
    defer eig.deinit(allocator);

    std.debug.print("Eigenvalues of [[1, 2], [2, 1]]: ", .{});
    for (0..eig.size()) |i| {
        std.debug.print("{d:.4} ", .{try eig.get(&.{i})});
    }
    std.debug.print("\n", .{});
}
