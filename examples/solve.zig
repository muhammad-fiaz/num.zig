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

    // 5. Triangular solve: [[2, 0], [1, 3]] x = [4, 7] -> [2, 5/3]
    const lt_data = [_]f64{ 2.0, 0.0, 1.0, 3.0 };
    var lt = try num.fromSlice(allocator, f64, .{ .data = &lt_data, .shape = &.{ 2, 2 } });
    defer lt.deinit();
    const ltb_data = [_]f64{ 4.0, 7.0 };
    var ltb = try num.fromSlice(allocator, f64, .{ .data = &ltb_data, .shape = &.{2} });
    defer ltb.deinit();
    var xt = try num.linalg.solveTriangular(lt, ltb, .{ .lower = true });
    defer xt.deinit();
    std.debug.print("Triangular solution: [{d:.4}, {d:.4}]\n", .{
        try xt.get(f64, &.{0}), try xt.get(f64, &.{1}),
    });

    // 6. SPD solve via Cholesky: [[4, 1], [1, 3]] x = [5, 4] -> [1, 1]
    const spd_data = [_]f64{ 4.0, 1.0, 1.0, 3.0 };
    var spd = try num.fromSlice(allocator, f64, .{ .data = &spd_data, .shape = &.{ 2, 2 } });
    defer spd.deinit();
    const spdb_data = [_]f64{ 5.0, 4.0 };
    var spdb = try num.fromSlice(allocator, f64, .{ .data = &spdb_data, .shape = &.{2} });
    defer spdb.deinit();
    var xs = try num.linalg.solveSpd(spd, spdb);
    defer xs.deinit();
    std.debug.print("SPD solution: [{d:.4}, {d:.4}]\n", .{
        try xs.get(f64, &.{0}), try xs.get(f64, &.{1}),
    });

    // 7. Least-squares: fit y = 2x + 1
    const ls_a = [_]f64{ 0, 1, 1, 1, 2, 1 };
    var lsa = try num.fromSlice(allocator, f64, .{ .data = &ls_a, .shape = &.{ 3, 2 } });
    defer lsa.deinit();
    const ls_y = [_]f64{ 1.0, 3.0, 5.0 };
    var lsy = try num.fromSlice(allocator, f64, .{ .data = &ls_y, .shape = &.{3} });
    defer lsy.deinit();
    var xls = try num.linalg.lstsq(lsa, lsy);
    defer xls.deinit();
    std.debug.print("Least-squares slope/intercept: [{d:.4}, {d:.4}]\n", .{
        try xls.get(f64, &.{0}), try xls.get(f64, &.{1}),
    });
}
