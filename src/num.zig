const std = @import("std");

/// Core data structures and basic operations.
///
/// This module defines the `NDArray` struct and fundamental operations like initialization,
/// reshaping, and basic accessors.
pub const core = @import("core.zig");

/// Advanced indexing and slicing.
///
/// Provides functionality for slicing, boolean masking, and integer array indexing.
pub const indexing = @import("indexing.zig");

/// Mathematical operations and broadcasting.
///
/// Contains general mathematical operations that support broadcasting rules.
pub const ops = @import("ops.zig");

/// Element-wise operations.
///
/// Functions that operate element-wise on arrays, such as arithmetic, trigonometric,
/// and logical operations.
pub const elementwise = @import("elementwise.zig");

/// Linear algebra operations.
///
/// Includes matrix multiplication, decomposition (Cholesky, QR, SVD), solving linear systems,
/// and other linear algebra routines.
pub const linalg = @import("linalg.zig");

/// Algorithms collection.
///
/// Includes Search, Lists, Stack, Queue, Graph, and Backtracking algorithms.
pub const algo = @import("algo.zig");

// Aliases for easier access
pub const Stack = algo.stack.Stack;
pub const Queue = algo.queue.Queue;
pub const LinkedList = algo.list.LinkedList;
pub const DoublyLinkedList = algo.list.DoublyLinkedList;
pub const CircularLinkedList = algo.list.CircularLinkedList;
pub const Graph = algo.graph.Graph;
pub const sort = algo.sort;

/// URL to report issues.
pub const report_issue_url = "https://github.com/muhammad-fiaz/num.zig/issues";

/// Statistical functions.
///
/// Provides statistical operations like mean, variance, standard deviation, min, max, etc.
pub const stats = @import("stats.zig");

/// Shape manipulation utilities.
pub const shape = @import("shape.zig");

/// Slicing utilities.
pub const slice = @import("slice.zig");

/// Reduction operations.
pub const reduction = @import("reduction.zig");

/// Sparse matrix support.
pub const sparse = @import("sparse/matrix.zig");

/// Autograd and Tensor operations.
pub const autograd = @import("autograd/tensor.zig");

/// DataFrame and Series.
pub const dataframe = struct {
    pub const Series = @import("dataframe/series.zig").Series;
    pub const DataFrame = @import("dataframe/dataframe.zig").DataFrame;
};

/// Optimization algorithms.
pub const optimize = struct {
    pub const GradientDescent = @import("optimize/gradient.zig").GradientDescent;
};

/// Math utilities including SIMD.
pub const math = struct {
    pub const simd = @import("math/simd.zig");
};

/// Random number generation.
///
/// Utilities for generating random numbers and sampling from distributions.
pub const random = @import("random.zig");

/// Fast Fourier Transform.
///
/// Implementation of 1D and 2D Fast Fourier Transforms and their inverses.
pub const fft = @import("fft.zig");

/// Signal processing.
pub const signal = @import("signal.zig");

/// Polynomial operations.
pub const poly = @import("poly.zig");

/// Interpolation.
pub const interpolate = @import("interpolate.zig");

/// Array manipulation routines.
///
/// Functions for changing array shapes, joining arrays, splitting arrays, and rearranging elements.
pub const manipulation = @import("manipulation.zig");

/// File I/O.
///
/// Utilities for saving and loading arrays to/from disk.
pub const io = @import("io.zig");

/// Machine Learning sub-module.
///
/// Contains building blocks for machine learning, including layers, activation functions,
/// loss functions, and optimizers.
pub const ml = @import("ml.zig");

/// Broadcasting utilities.
pub const broadcast = @import("broadcast.zig");

/// Finite differences and gradients.
pub const diff = @import("diff.zig");

/// Set operations.
pub const setops = @import("setops.zig");

/// Complex number support.
pub const complex = @import("complex.zig");

/// The main N-dimensional array type.
///
/// `NDArray(T)` is the central data structure of the library, representing a multidimensional
/// homogeneous array of fixed-size items.
///
/// Example:
/// ```zig
/// const allocator = std.heap.page_allocator;
/// var arr = try NDArray(f32).zeros(allocator, &.{2, 3});
/// defer arr.deinit();
/// try arr.set(&.{0, 0}, 1.0);
/// ```
pub const NDArray = core.NDArray;

test "NDArray init and access" {
    const allocator = std.testing.allocator;
    var arr = try NDArray(f32).zeros(allocator, &.{ 2, 3 });
    defer arr.deinit();

    try std.testing.expectEqual(arr.rank(), 2);
    try std.testing.expectEqual(arr.size(), 6);
    try std.testing.expectEqual(try arr.get(&.{ 0, 0 }), 0.0);

    try arr.set(&.{ 1, 2 }, 5.0);
    try std.testing.expectEqual(try arr.get(&.{ 1, 2 }), 5.0);
}

test "ops add broadcasting" {
    const allocator = std.testing.allocator;

    var a = try NDArray(f32).ones(allocator, &.{ 2, 3 });
    defer a.deinit();

    var b = try NDArray(f32).full(allocator, &.{ 1, 3 }, 2.0);
    defer b.deinit();

    var c = try ops.add(f32, allocator, &a, &b);
    defer c.deinit();

    try std.testing.expectEqual(c.shape.len, 2);
    try std.testing.expectEqual(c.shape[0], 2);
    try std.testing.expectEqual(c.shape[1], 3);

    // 1.0 + 2.0 = 3.0
    try std.testing.expectEqual(try c.get(&.{ 0, 0 }), 3.0);
    try std.testing.expectEqual(try c.get(&.{ 1, 2 }), 3.0);
}

test "creation and manipulation" {
    const allocator = std.testing.allocator;

    // arange
    var a = try NDArray(f32).arange(allocator, 0, 5, 1);
    defer a.deinit();
    try std.testing.expectEqual(a.size(), 5);
    try std.testing.expectEqual(a.data[4], 4.0);

    // linspace
    var l = try NDArray(f32).linspace(allocator, 0, 10, 5);
    defer l.deinit();
    try std.testing.expectEqual(l.size(), 5);
    try std.testing.expectEqual(l.data[0], 0.0);
    try std.testing.expectEqual(l.data[4], 10.0);
    try std.testing.expectEqual(l.data[2], 5.0);

    // reshape
    var r = try a.reshape(&.{ 5, 1 });
    defer r.deinit();
    try std.testing.expectEqual(r.rank(), 2);
    try std.testing.expectEqual(r.shape[0], 5);
    try std.testing.expectEqual(r.shape[1], 1);

    // flatten
    var f = try r.flatten();
    defer f.deinit();
    try std.testing.expectEqual(f.rank(), 1);
    try std.testing.expectEqual(f.size(), 5);

    // transpose
    var t = try r.transpose();
    defer t.deinit();
    try std.testing.expectEqual(t.shape[0], 1);
    try std.testing.expectEqual(t.shape[1], 5);
}

test "random" {
    const allocator = std.testing.allocator;
    var rng = random.Random.init(42);

    var r = try rng.randint(allocator, &.{10}, 0, 10);
    defer r.deinit();
    try std.testing.expectEqual(r.size(), 10);

    var s = try NDArray(f32).arange(allocator, 0, 10, 1);
    defer s.deinit();
    rng.shuffle(f32, &s);

    var sum: f32 = 0;
    for (s.data) |val| sum += val;
    try std.testing.expectEqual(sum, 45.0);
}

test "linalg matmul" {
    const allocator = std.testing.allocator;

    // 2x3 matrix
    var a = try NDArray(f32).init(allocator, &.{ 2, 3 });
    defer a.deinit();
    @memset(a.data, 1.0); // All ones

    // 3x2 matrix
    var b = try NDArray(f32).init(allocator, &.{ 3, 2 });
    defer b.deinit();
    @memset(b.data, 2.0); // All twos

    // Result should be 2x2
    // Each element = sum(1.0 * 2.0) for k=0..2 => 2.0 * 3 = 6.0
    var c = try linalg.matmul(f32, allocator, &a, &b);
    defer c.deinit();

    try std.testing.expectEqual(c.shape.len, 2);
    try std.testing.expectEqual(c.shape[0], 2);
    try std.testing.expectEqual(c.shape[1], 2);

    try std.testing.expectEqual(try c.get(&.{ 0, 0 }), 6.0);
    try std.testing.expectEqual(try c.get(&.{ 1, 1 }), 6.0);
}

test "stats reductions" {
    const allocator = std.testing.allocator;
    var a = try NDArray(f32).init(allocator, &.{4});
    defer a.deinit();

    // 1, 2, 3, 4
    a.data[0] = 1.0;
    a.data[1] = 2.0;
    a.data[2] = 3.0;
    a.data[3] = 4.0;

    try std.testing.expectEqual(stats.sum(f32, &a), 10.0);
    try std.testing.expectEqual(stats.mean(f32, &a), 2.5);
    try std.testing.expectEqual(stats.max(f32, &a), 4.0);
    try std.testing.expectEqual(stats.min(f32, &a), 1.0);
}

test "random generation" {
    const allocator = std.testing.allocator;
    var rng = random.Random.init(42);

    var u = try rng.uniform(allocator, &.{10});
    defer u.deinit();

    try std.testing.expectEqual(u.size(), 10);
    for (u.data) |val| {
        try std.testing.expect(val >= 0.0 and val < 1.0);
    }

    var n = try rng.normal(allocator, &.{100}, 0.0, 1.0);
    defer n.deinit();

    // Check basic stats of normal distribution
    const mean = try stats.mean(f32, &n);
    // Mean should be close to 0
    try std.testing.expect(mean > -0.5 and mean < 0.5);
}

test "ml activations" {
    const allocator = std.testing.allocator;
    var a = try NDArray(f32).init(allocator, &.{3});
    defer a.deinit();
    a.data[0] = -1.0;
    a.data[1] = 0.0;
    a.data[2] = 1.0;

    var r = try ml.activations.relu(allocator, &a);
    defer r.deinit();

    try std.testing.expectEqual(try r.get(&.{0}), 0.0);
    try std.testing.expectEqual(try r.get(&.{1}), 0.0);
    try std.testing.expectEqual(try r.get(&.{2}), 1.0);
}

test "ml loss" {
    const allocator = std.testing.allocator;
    var y_true = try NDArray(f32).init(allocator, &.{2});
    defer y_true.deinit();
    y_true.data[0] = 1.0;
    y_true.data[1] = 0.0;

    var y_pred = try NDArray(f32).init(allocator, &.{2});
    defer y_pred.deinit();
    y_pred.data[0] = 0.9;
    y_pred.data[1] = 0.1;

    const loss = try ml.loss.mse(allocator, &y_true, &y_pred);
    // (0.1^2 + (-0.1)^2) / 2 = (0.01 + 0.01) / 2 = 0.01
    try std.testing.expectApproxEqAbs(loss, 0.01, 1e-5);
}

test "ml layers and optim" {
    const allocator = std.testing.allocator;

    // Dense layer 2 -> 1
    var layer = try ml.layers.Dense.init(allocator, 2, 1, .XavierUniform);
    defer layer.deinit();
    layer.weights.fill(0.01);
    layer.bias.fill(0.0);

    // Input batch size 1
    var input = try NDArray(f32).init(allocator, &.{ 1, 2 });
    defer input.deinit();
    input.data[0] = 1.0;
    input.data[1] = 2.0;

    // Forward
    var output = try layer.forward(allocator, &input);
    defer output.deinit();

    // weights are 0.01, bias 0
    // 1*0.01 + 2*0.01 = 0.03
    try std.testing.expectApproxEqAbs(try output.get(&.{ 0, 0 }), 0.03, 1e-5);

    // Optim
    var sgd = ml.optim.SGD.init(0.1);
    var grad = try NDArray(f32).full(allocator, layer.weights.shape, 0.1);
    defer grad.deinit();

    sgd.update(&layer.weights, &grad);

    // New weight = 0.01 - 0.1 * 0.1 = 0.0
    try std.testing.expectApproxEqAbs(try layer.weights.get(&.{ 0, 0 }), 0.0, 1e-5);
}

test "core transpose" {
    const allocator = std.testing.allocator;
    var a = try NDArray(f32).init(allocator, &.{ 2, 3 });
    defer a.deinit();

    // [[0, 1, 2], [3, 4, 5]]
    for (a.data, 0..) |*val, i| val.* = @as(f32, @floatFromInt(i));

    var t = try a.transpose();
    defer t.deinit();

    try std.testing.expectEqual(t.shape.len, 2);
    try std.testing.expectEqual(t.shape[0], 3);
    try std.testing.expectEqual(t.shape[1], 2);

    // t[0, 0] = a[0, 0] = 0
    // t[0, 1] = a[1, 0] = 3
    // t[1, 0] = a[0, 1] = 1
    try std.testing.expectEqual(try t.get(&.{ 0, 1 }), 3.0);
    try std.testing.expectEqual(try t.get(&.{ 1, 0 }), 1.0);
}

test "ml softmax arbitrary axis" {
    const allocator = std.testing.allocator;

    // 2x2 matrix
    // [[1, 2],
    //  [3, 4]]
    var a = try NDArray(f32).init(allocator, &.{ 2, 2 });
    defer a.deinit();
    a.data[0] = 1.0;
    a.data[1] = 2.0;
    a.data[2] = 3.0;
    a.data[3] = 4.0;

    // Axis 1 (rows)
    // exp(1)/(exp(1)+exp(2)) = 0.2689
    // exp(2)/(exp(1)+exp(2)) = 0.7311
    var s1 = try ml.activations.softmax(allocator, &a, 1);
    defer s1.deinit();

    try std.testing.expectApproxEqAbs(try s1.get(&.{ 0, 0 }), 0.26894, 1e-4);
    try std.testing.expectApproxEqAbs(try s1.get(&.{ 0, 1 }), 0.73105, 1e-4);

    // Axis 0 (cols)
    // Col 0: 1, 3 -> exp(1)/(exp(1)+exp(3)) = 0.1192
    // Col 1: 2, 4 -> exp(2)/(exp(2)+exp(4)) = 0.1192
    var s0 = try ml.activations.softmax(allocator, &a, 0);
    defer s0.deinit();

    try std.testing.expectApproxEqAbs(try s0.get(&.{ 0, 0 }), 0.11920, 1e-4);
    try std.testing.expectApproxEqAbs(try s0.get(&.{ 1, 0 }), 0.88079, 1e-4);
}

test "core arange and linspace" {
    const allocator = std.testing.allocator;

    // arange
    var a = try NDArray(f32).arange(allocator, 0.0, 5.0, 1.0);
    defer a.deinit();
    try std.testing.expectEqual(a.size(), 5);
    try std.testing.expectEqual(try a.get(&.{0}), 0.0);
    try std.testing.expectEqual(try a.get(&.{4}), 4.0);

    // linspace
    var b = try NDArray(f32).linspace(allocator, 0.0, 10.0, 5);
    defer b.deinit();
    try std.testing.expectEqual(b.size(), 5);
    try std.testing.expectEqual(try b.get(&.{0}), 0.0);
    try std.testing.expectEqual(try b.get(&.{2}), 5.0); // Middle
    try std.testing.expectEqual(try b.get(&.{4}), 10.0);
}

test "core reshape" {
    const allocator = std.testing.allocator;
    var a = try NDArray(f32).arange(allocator, 0.0, 6.0, 1.0);
    defer a.deinit();

    var b = try a.reshape(&.{ 2, 3 });
    defer b.deinit();

    try std.testing.expectEqual(b.rank(), 2);
    try std.testing.expectEqual(b.shape[0], 2);
    try std.testing.expectEqual(b.shape[1], 3);
    try std.testing.expectEqual(try b.get(&.{ 0, 0 }), 0.0);
    try std.testing.expectEqual(try b.get(&.{ 1, 2 }), 5.0);
}

test "stats sumAxis" {
    const allocator = std.testing.allocator;
    var a = try NDArray(f32).ones(allocator, &.{ 2, 3 });
    defer a.deinit();

    // Sum along axis 0 (rows) -> result shape (3,)
    var s0 = try stats.sumAxis(f32, allocator, &a, 0, false);
    defer s0.deinit();
    try std.testing.expectEqual(s0.rank(), 1);
    try std.testing.expectEqual(s0.shape[0], 3);
    try std.testing.expectEqual(try s0.get(&.{0}), 2.0); // 1+1

    // Sum along axis 1 (cols) -> result shape (2,)
    var s1 = try stats.sumAxis(f32, allocator, &a, 1, false);
    defer s1.deinit();
    try std.testing.expectEqual(s1.rank(), 1);
    try std.testing.expectEqual(s1.shape[0], 2);
    try std.testing.expectEqual(try s1.get(&.{0}), 3.0); // 1+1+1
}

test "core eye" {
    const allocator = std.testing.allocator;
    var a = try NDArray(f32).eye(allocator, 3);
    defer a.deinit();

    try std.testing.expectEqual(a.rank(), 2);
    try std.testing.expectEqual(a.shape[0], 3);
    try std.testing.expectEqual(a.shape[1], 3);
    try std.testing.expectEqual(try a.get(&.{ 0, 0 }), 1.0);
    try std.testing.expectEqual(try a.get(&.{ 1, 1 }), 1.0);
    try std.testing.expectEqual(try a.get(&.{ 2, 2 }), 1.0);
    try std.testing.expectEqual(try a.get(&.{ 0, 1 }), 0.0);
}

test "core flatten" {
    const allocator = std.testing.allocator;
    var a = try NDArray(f32).init(allocator, &.{ 2, 3 });
    defer a.deinit();
    // 0, 1, 2, 3, 4, 5
    for (a.data, 0..) |*val, i| {
        val.* = @as(f32, @floatFromInt(i));
    }

    var flat = try a.flatten();
    defer flat.deinit();

    try std.testing.expectEqual(flat.rank(), 1);
    try std.testing.expectEqual(flat.shape[0], 6);
    try std.testing.expectEqual(flat.data[5], 5.0);
}

test "stats extended" {
    const allocator = std.testing.allocator;
    var a = try NDArray(f32).init(allocator, &.{5});
    defer a.deinit();
    // 1, 2, 3, 4, 5
    a.data[0] = 1.0;
    a.data[1] = 2.0;
    a.data[2] = 3.0;
    a.data[3] = 4.0;
    a.data[4] = 5.0;

    // Mean = 3
    // Var = ((1-3)^2 + (2-3)^2 + 0 + (4-3)^2 + (5-3)^2) / 5
    //     = (4 + 1 + 0 + 1 + 4) / 5 = 10 / 5 = 2.0
    try std.testing.expectEqual(try stats.varianceAll(f32, &a), 2.0);
    try std.testing.expectApproxEqAbs(try stats.stddev(f32, &a), 1.41421356, 1e-5);

    try std.testing.expectEqual(try stats.argmax(f32, &a), 4);
    try std.testing.expectEqual(try stats.argmin(f32, &a), 0);
}

test "FFT" {
    const allocator = std.testing.allocator;
    var a = try NDArray(f32).init(allocator, &.{4});
    defer a.deinit();
    // Simple signal: 1, 1, 1, 1 -> FFT -> 4, 0, 0, 0
    a.data[0] = 1.0;
    a.data[1] = 1.0;
    a.data[2] = 1.0;
    a.data[3] = 1.0;

    var f = try fft.FFT.fft(allocator, &a);
    defer f.deinit();

    try std.testing.expectApproxEqAbs(f.data[0].re, 4.0, 1e-5);
    try std.testing.expectApproxEqAbs(f.data[0].im, 0.0, 1e-5);
    try std.testing.expectApproxEqAbs(f.data[1].re, 0.0, 1e-5);
    try std.testing.expectApproxEqAbs(f.data[2].re, 0.0, 1e-5);

    var inv = try fft.FFT.ifft(allocator, &f);
    defer inv.deinit();

    try std.testing.expectApproxEqAbs(inv.data[0].re, 1.0, 1e-5);
    try std.testing.expectApproxEqAbs(inv.data[1].re, 1.0, 1e-5);
}

test "advanced features" {
    const allocator = std.testing.allocator;

    // Squeeze
    var a = try NDArray(f32).zeros(allocator, &.{ 1, 2, 1 });
    defer a.deinit();
    var s = try a.squeeze();
    defer s.deinit();
    try std.testing.expectEqual(s.rank(), 1);
    try std.testing.expectEqual(s.shape[0], 2);

    // ExpandDims
    var e = try s.expandDims(1);
    defer e.deinit();
    try std.testing.expectEqual(e.rank(), 2);
    try std.testing.expectEqual(e.shape[0], 2);
    try std.testing.expectEqual(e.shape[1], 1);

    // Stack
    var s1 = try NDArray(f32).zeros(allocator, &.{2});
    defer s1.deinit();
    var s2 = try NDArray(f32).zeros(allocator, &.{2});
    defer s2.deinit();
    var stacked = try NDArray(f32).stack(allocator, &.{ s1, s2 }, 0);
    defer stacked.deinit();
    try std.testing.expectEqual(stacked.shape[0], 2);
    try std.testing.expectEqual(stacked.shape[1], 2);

    // Ops
    var x = try NDArray(f32).init(allocator, &.{1});
    defer x.deinit();
    x.data[0] = 0.0;
    var exp_x = try ops.exp(f32, allocator, &x);
    defer exp_x.deinit();
    try std.testing.expectEqual(exp_x.data[0], 1.0);

    // Stats
    x.data[0] = 2.0;
    try std.testing.expectEqual(stats.prod(f32, &x), 2.0);
}

test "stats extra" {
    const allocator = std.testing.allocator;

    // bincount
    var a = try NDArray(i32).init(allocator, &.{5});
    defer a.deinit();
    a.data[0] = 0;
    a.data[1] = 1;
    a.data[2] = 1;
    a.data[3] = 3;
    a.data[4] = 2;

    var bc = try stats.bincount(allocator, &a);
    defer bc.deinit();
    try std.testing.expectEqual(bc.size(), 4); // 0, 1, 2, 3
    try std.testing.expectEqual(bc.data[0], 1);
    try std.testing.expectEqual(bc.data[1], 2);
    try std.testing.expectEqual(bc.data[2], 1);
    try std.testing.expectEqual(bc.data[3], 1);

    // unique
    var u = try stats.unique(i32, allocator, &a);
    defer u.deinit();
    try std.testing.expectEqual(u.size(), 4);
    try std.testing.expectEqual(u.data[0], 0);
    try std.testing.expectEqual(u.data[1], 1);
    try std.testing.expectEqual(u.data[2], 2);
    try std.testing.expectEqual(u.data[3], 3);
}

test "sort argsort" {
    const allocator = std.testing.allocator;
    var a = try NDArray(f32).init(allocator, &.{3});
    defer a.deinit();
    a.data[0] = 3.0;
    a.data[1] = 1.0;
    a.data[2] = 2.0;

    var idx = try sort.argsort(allocator, f32, a, 0);
    defer idx.deinit();

    try std.testing.expectEqual(idx.data[0], 1); // 1.0 is at index 1
    try std.testing.expectEqual(idx.data[1], 2); // 2.0 is at index 2
    try std.testing.expectEqual(idx.data[2], 0); // 3.0 is at index 0
}

test {
    _ = setops;
    _ = signal;
    _ = poly;
    _ = stats;
}
