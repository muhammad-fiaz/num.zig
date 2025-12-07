const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("Testing All Features (Comprehensive)...\n", .{});

    // 1. Shape & Slice
    std.debug.print("\n--- Shape & Slice ---\n", .{});
    const s1 = try num.shape.broadcastShapes(allocator, &.{ 3, 1 }, &.{2});
    defer allocator.free(s1);
    std.debug.print("Broadcasted shape: {any}\n", .{s1});

    // 2. Reduction
    std.debug.print("\n--- Reduction ---\n", .{});
    var arr = try num.NDArray(f64).init(allocator, &.{ 2, 2 });
    defer arr.deinit();
    arr.data[0] = 1;
    arr.data[1] = 2;
    arr.data[2] = 3;
    arr.data[3] = 4;

    var sum_val = try num.reduction.sum(allocator, f64, arr, null);
    defer sum_val.deinit();
    std.debug.print("Sum all: {d}\n", .{sum_val.data[0]});

    // 3. Sparse
    std.debug.print("\n--- Sparse ---\n", .{});
    var dense = try num.NDArray(f64).zeros(allocator, &.{ 3, 3 });
    defer dense.deinit();
    try dense.set(&.{ 0, 0 }, 1.0);
    try dense.set(&.{ 2, 2 }, 5.0);

    var csr = try num.sparse.CSRMatrix(f64).fromDense(allocator, dense);
    defer csr.deinit();
    std.debug.print("CSR NNZ: {d}\n", .{csr.values.items.len});

    var back_dense = try csr.toDense();
    defer back_dense.deinit();
    std.debug.print("Back to dense (0,0): {d}\n", .{try back_dense.get(&.{ 0, 0 })});

    // 4. Autograd
    std.debug.print("\n--- Autograd ---\n", .{});
    const t_data = try num.NDArray(f64).zeros(allocator, &.{ 2, 2 });
    var t = try num.autograd.Tensor(f64).init(allocator, t_data, true);
    defer t.deinit();
    try t.backward();
    if (t.grad) |g| {
        std.debug.print("Grad (0,0): {d}\n", .{try g.get(&.{ 0, 0 })});
    }

    // 5. DataFrame
    std.debug.print("\n--- DataFrame ---\n", .{});
    var df = num.dataframe.DataFrame.init(allocator);
    defer df.deinit();

    var s_data = try num.NDArray(f64).init(allocator, &.{3});
    s_data.data[0] = 10;
    s_data.data[1] = 20;
    s_data.data[2] = 30;
    const series = try num.dataframe.Series(f64).init(allocator, "Age", s_data);

    try df.addColumn("Age", .{ .Float = series });
    if (df.getColumn("Age")) |col| {
        switch (col) {
            .Float => |s| try s.print(),
            else => {},
        }
    }

    // 6. Optimize
    std.debug.print("\n--- Optimize ---\n", .{});
    var opt = num.optimize.GradientDescent(f64).init(0.1);
    var param = try num.NDArray(f64).init(allocator, &.{1});
    defer param.deinit();
    param.data[0] = 5.0;
    var grad = try num.NDArray(f64).init(allocator, &.{1});
    defer grad.deinit();
    grad.data[0] = 1.0;

    try opt.step(&param, grad);
    std.debug.print("Updated param: {d}\n", .{param.data[0]});

    // 7. Sorting
    std.debug.print("\n--- Sorting ---\n", .{});
    var to_sort = try num.NDArray(f64).init(allocator, &.{5});
    defer to_sort.deinit();
    to_sort.data[0] = 5;
    to_sort.data[1] = 1;
    to_sort.data[2] = 4;
    to_sort.data[3] = 2;
    to_sort.data[4] = 8;

    var sorted_qs = try num.sort.sortByAlgo(allocator, f64, to_sort, 0, .QuickSort);
    defer sorted_qs.deinit();
    std.debug.print("QuickSort: {d}, {d}, {d}, {d}, {d}\n", .{ sorted_qs.data[0], sorted_qs.data[1], sorted_qs.data[2], sorted_qs.data[3], sorted_qs.data[4] });

    // 8. Interpolation
    std.debug.print("\n--- Interpolation ---\n", .{});
    var x_interp = try num.NDArray(f64).init(allocator, &.{3});
    defer x_interp.deinit();
    x_interp.data[0] = 0;
    x_interp.data[1] = 1;
    x_interp.data[2] = 2;

    var y_interp = try num.NDArray(f64).init(allocator, &.{3});
    defer y_interp.deinit();
    y_interp.data[0] = 0;
    y_interp.data[1] = 10;
    y_interp.data[2] = 0;

    var xi = try num.NDArray(f64).init(allocator, &.{1});
    defer xi.deinit();
    xi.data[0] = 0.5;

    var yi = try num.interpolate.interp1d(allocator, f64, x_interp, y_interp, xi);
    defer yi.deinit();
    std.debug.print("Interp at 0.5: {d}\n", .{yi.data[0]});

    // 9. Polynomial
    std.debug.print("\n--- Polynomial ---\n", .{});
    var p_coeffs = try num.NDArray(f64).init(allocator, &.{3});
    defer p_coeffs.deinit();
    p_coeffs.data[0] = 1; // x^2
    p_coeffs.data[1] = 0; // x
    p_coeffs.data[2] = -1; // constant
    // x^2 - 1

    var x_poly = try num.NDArray(f64).init(allocator, &.{1});
    defer x_poly.deinit();
    x_poly.data[0] = 2.0;

    var y_poly = try num.poly.polyval(allocator, f64, p_coeffs, x_poly);
    defer y_poly.deinit();
    std.debug.print("Polyval(2.0): {d}\n", .{y_poly.data[0]});

    // 10. Complex
    std.debug.print("\n--- Complex ---\n", .{});
    const c1 = num.complex.Complex(f64).init(1.0, 2.0);
    const c2 = num.complex.Complex(f64).init(3.0, 4.0);
    const c_sum = c1.add(c2);
    std.debug.print("Complex Add: {d} + {d}i\n", .{ c_sum.re, c_sum.im });

    // 11. FFT
    std.debug.print("\n--- FFT ---\n", .{});
    var fft_in = try num.NDArray(f32).init(allocator, &.{4});
    defer fft_in.deinit();
    fft_in.data[0] = 1.0;
    fft_in.data[1] = 1.0;
    fft_in.data[2] = 1.0;
    fft_in.data[3] = 1.0;

    var fft_out = try num.fft.FFT.fft(allocator, &fft_in);
    defer fft_out.deinit();
    std.debug.print("FFT[0]: {d} + {d}i\n", .{ fft_out.data[0].re, fft_out.data[0].im });

    // 12. Signal
    std.debug.print("\n--- Signal ---\n", .{});
    var sig_in = try num.NDArray(f64).init(allocator, &.{5});
    defer sig_in.deinit();
    sig_in.data[0] = 1;
    sig_in.data[1] = 2;
    sig_in.data[2] = 3;
    sig_in.data[3] = 4;
    sig_in.data[4] = 5;

    var kernel = try num.NDArray(f64).init(allocator, &.{3});
    defer kernel.deinit();
    kernel.data[0] = 0.5;
    kernel.data[1] = 1.0;
    kernel.data[2] = 0.5;

    var conv_res = try num.signal.convolve(allocator, f64, sig_in, kernel, .same);
    defer conv_res.deinit();
    std.debug.print("Convolve center: {d}\n", .{conv_res.data[2]});

    // 13. SetOps
    std.debug.print("\n--- SetOps ---\n", .{});
    var set_a = try num.NDArray(f64).init(allocator, &.{3});
    defer set_a.deinit();
    set_a.data[0] = 1;
    set_a.data[1] = 2;
    set_a.data[2] = 3;

    var set_b = try num.NDArray(f64).init(allocator, &.{3});
    defer set_b.deinit();
    set_b.data[0] = 2;
    set_b.data[1] = 3;
    set_b.data[2] = 4;

    var set_union = try num.setops.union1d(allocator, f64, set_a, set_b);
    defer set_union.deinit();
    std.debug.print("Union len: {d}\n", .{set_union.shape[0]});

    // 14. IO
    std.debug.print("\n--- IO ---\n", .{});
    const filename = "test_io_comprehensive.npy";
    var io_arr = try num.NDArray(f64).init(allocator, &.{ 2, 2 });
    defer io_arr.deinit();
    io_arr.data[0] = 1.1;
    io_arr.data[1] = 2.2;
    io_arr.data[2] = 3.3;
    io_arr.data[3] = 4.4;

    try num.io.save(f64, io_arr, filename);
    std.debug.print("Saved {s}\n", .{filename});

    var loaded_arr = try num.io.load(allocator, f64, filename);
    defer loaded_arr.deinit();
    std.debug.print("Loaded (0,0): {d}\n", .{loaded_arr.data[0]});

    // Cleanup
    std.fs.cwd().deleteFile(filename) catch {};
}
