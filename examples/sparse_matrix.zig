const std = @import("std");
const num = @import("num");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // Dense 3x3 matrix with many zeros:
    // [ 1.0, 0.0, 2.0 ]
    // [ 0.0, 0.0, 3.0 ]
    // [ 4.0, 5.0, 0.0 ]
    const data = [_]f64{
        1.0, 0.0, 2.0,
        0.0, 0.0, 3.0,
        4.0, 5.0, 0.0,
    };
    var dense = try num.fromSlice(allocator, f64, .{ .data = &data, .shape = &.{ 3, 3 } });
    defer dense.deinit();

    std.debug.print("Dense Matrix (3x3):\n", .{});
    for (0..3) |r| {
        std.debug.print("  [ {d:.1}, {d:.1}, {d:.1} ]\n", .{
            try dense.get(f64, &.{ r, 0 }),
            try dense.get(f64, &.{ r, 1 }),
            try dense.get(f64, &.{ r, 2 }),
        });
    }

    // 1. Convert to Compressed Sparse Row (CSR) format
    var csr = try num.sparse.CsrMatrix.fromDense(allocator, dense, 1e-12);
    defer csr.deinit();

    std.debug.print("\nCSR Matrix Representation:\n", .{});
    std.debug.print("  Shape: [{d}, {d}], NNZ (Non-zeros): {d}\n", .{ csr.rows, csr.cols, csr.nnz() });
    std.debug.print("  Values: {any}\n", .{csr.data});
    std.debug.print("  Col Indices: {any}\n", .{csr.indices});
    std.debug.print("  Row Pointers: {any}\n", .{csr.indptr});

    // 2. Sparse Matrix - Vector Multiplication
    const vec_data = [_]f64{ 1.0, 2.0, 3.0 };
    var x = try num.fromSlice(allocator, f64, .{ .data = &vec_data, .shape = &.{3} });
    defer x.deinit();

    var y = try csr.dotVector(x);
    defer y.deinit();

    std.debug.print("\nSparse MatVec Product y = A * x where x = [1, 2, 3]:\n", .{});
    std.debug.print("  y = [ {d:.1}, {d:.1}, {d:.1} ]\n", .{
        try y.get(f64, &.{0}),
        try y.get(f64, &.{1}),
        try y.get(f64, &.{2}),
    });

    // 3. Convert back to Dense Matrix to verify exact round-trip
    var recovered = try csr.toDense();
    defer recovered.deinit();

    var eq_arr = try num.equal(dense, recovered);
    defer eq_arr.deinit();
    var all_arr = try num.all(eq_arr, .{});
    defer all_arr.deinit();
    const all_match = try all_arr.get(bool, &.{});

    std.debug.print("\nRecovered Dense Matrix matches original: {s}\n", .{
        if (all_match) "yes" else "no",
    });

    // 4. Solve a symmetric positive-definite sparse system using Conjugate Gradient
    // A_spd = [ 4, 1 ]
    //         [ 1, 3 ]
    // b_spd = [ 5, 4 ] -> solution x = [ 1, 1 ]
    const spd_data = [_]f64{
        4.0, 1.0,
        1.0, 3.0,
    };
    var spd_dense = try num.fromSlice(allocator, f64, .{ .data = &spd_data, .shape = &.{ 2, 2 } });
    defer spd_dense.deinit();

    var spd_csr = try num.sparse.CsrMatrix.fromDense(allocator, spd_dense, 1e-12);
    defer spd_csr.deinit();

    const spd_b_data = [_]f64{ 5.0, 4.0 };
    var spd_b = try num.fromSlice(allocator, f64, .{ .data = &spd_b_data, .shape = &.{2} });
    defer spd_b.deinit();

    var cg_sol = try num.sparse.cg(spd_csr, spd_b, .{ .tol = 1e-8 });
    defer cg_sol.deinit();

    std.debug.print("\nConjugate Gradient Solver (CG):\n", .{});
    std.debug.print("  Converged: {s} in {d} iterations\n", .{ if (cg_sol.converged) "yes" else "no", cg_sol.iterations });
    std.debug.print("  Solution x: [ {d:.4}, {d:.4} ] (expected [ 1.0000, 1.0000 ])\n", .{
        try cg_sol.x.get(f64, &.{0}),
        try cg_sol.x.get(f64, &.{1}),
    });
}
