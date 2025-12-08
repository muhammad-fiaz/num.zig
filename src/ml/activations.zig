const std = @import("std");
const core = @import("../core.zig");
const autograd = @import("../autograd/tensor.zig");
const NDArray = core.NDArray;
const Tensor = autograd.Tensor;
const Allocator = std.mem.Allocator;

fn apply(allocator: Allocator, a: *const NDArray(f32), context: anytype, op: anytype) !NDArray(f32) {
    const result = try NDArray(f32).init(allocator, a.shape);
    if (a.flags().c_contiguous) {
        for (a.data[0..a.size()], 0..) |val, i| {
            result.data[i] = op(context, val);
        }
    } else {
        var iter = try core.NdIterator.init(allocator, a.shape);
        defer iter.deinit(allocator);
        var i: usize = 0;
        while (iter.next()) |coords| {
            const val = try a.get(coords);
            result.data[i] = op(context, val);
            i += 1;
        }
    }
    return result;
}

fn reluOp(_: void, val: f32) f32 {
    return if (val > 0) val else 0;
}

/// Computes the Rectified Linear Unit (ReLU) activation function element-wise.
///
/// Logic: f(x) = max(0, x)
///
/// Arguments:
///     allocator: The allocator to use for the result array.
///     a: The input array.
///
/// Returns:
///     A new array containing the ReLU of the input elements.
///
/// Example:
/// ```zig
/// var a = try NDArray(f32).init(allocator, &.{3}, &.{ -1.0, 0.0, 1.0 });
/// defer a.deinit(allocator);
///
/// var res = try activations.relu(allocator, &a);
/// defer res.deinit(allocator);
/// // res is {0.0, 0.0, 1.0}
/// ```
pub fn relu(allocator: Allocator, a: *const NDArray(f32)) !NDArray(f32) {
    return apply(allocator, a, {}, reluOp);
}

fn sigmoidOp(_: void, val: f32) f32 {
    return 1.0 / (1.0 + std.math.exp(-val));
}

/// Computes the Sigmoid activation function element-wise.
///
/// Logic: f(x) = 1 / (1 + exp(-x))
///
/// It maps input values to the range (0, 1).
///
/// Arguments:
///     allocator: The allocator to use for the result array.
///     a: The input array.
///
/// Returns:
///     A new array containing the Sigmoid of the input elements.
///
/// Example:
/// ```zig
/// var a = try NDArray(f32).init(allocator, &.{3}, &.{ 0.0, 2.0, -2.0 });
/// defer a.deinit(allocator);
///
/// var res = try activations.sigmoid(allocator, &a);
/// defer res.deinit(allocator);
/// // res is {0.5, 0.8808, 0.1192}
/// ```
pub fn sigmoid(allocator: Allocator, a: *const NDArray(f32)) !NDArray(f32) {
    return apply(allocator, a, {}, sigmoidOp);
}

fn tanhOp(_: void, val: f32) f32 {
    return std.math.tanh(val);
}

/// Computes the Hyperbolic Tangent (tanh) activation function element-wise.
///
/// Logic: f(x) = tanh(x)
///
/// It maps input values to the range (-1, 1).
///
/// Arguments:
///     allocator: The allocator to use for the result array.
///     a: The input array.
///
/// Returns:
///     A new array containing the tanh of the input elements.
///
/// Example:
/// ```zig
/// var a = try NDArray(f32).init(allocator, &.{3}, &.{ 0.0, 1.0, -1.0 });
/// defer a.deinit(allocator);
///
/// var res = try activations.tanh(allocator, &a);
/// defer res.deinit(allocator);
/// // res is {0.0, 0.7616, -0.7616}
/// ```
pub fn tanh(allocator: Allocator, a: *const NDArray(f32)) !NDArray(f32) {
    return apply(allocator, a, {}, tanhOp);
}

fn leakyReluOp(alpha: f32, val: f32) f32 {
    return if (val > 0) val else alpha * val;
}

/// Computes the Leaky ReLU activation function element-wise.
///
/// The Leaky ReLU function is defined as:
/// f(x) = x if x > 0 else alpha * x
///
/// Arguments:
///     allocator: The allocator to use for the result array.
///     a: The input array.
///     alpha: The slope of the function for x < 0.
///
/// Returns:
///     A new array containing the Leaky ReLU of the input elements.
///
/// Example:
/// ```zig
/// var a = try NDArray(f32).init(allocator, &.{3}, &.{ -1.0, 0.0, 1.0 });
/// defer a.deinit(allocator);
///
/// var res = try activations.leakyRelu(allocator, &a, 0.01);
/// defer res.deinit(allocator);
/// // res is {-0.01, 0.0, 1.0}
/// ```
pub fn leakyRelu(allocator: Allocator, a: *const NDArray(f32), alpha: f32) !NDArray(f32) {
    return apply(allocator, a, alpha, leakyReluOp);
}

fn softplusOp(_: void, val: f32) f32 {
    return std.math.log(f32, std.math.e, 1.0 + std.math.exp(val));
}

/// Computes the Softplus activation function element-wise.
///
/// The Softplus function is defined as:
/// f(x) = ln(1 + exp(x))
///
/// Arguments:
///     allocator: The allocator to use for the result array.
///     a: The input array.
///
/// Returns:
///     A new array containing the Softplus of the input elements.
///
/// Example:
/// ```zig
/// var a = try NDArray(f32).init(allocator, &.{2}, &.{ 0.0, 1.0 });
/// defer a.deinit(allocator);
///
/// var res = try activations.softplus(allocator, &a);
/// defer res.deinit(allocator);
/// // res is {0.6931, 1.3133}
/// ```
pub fn softplus(allocator: Allocator, a: *const NDArray(f32)) !NDArray(f32) {
    return apply(allocator, a, {}, softplusOp);
}

/// Computes the Softmax activation function along the specified axis.
///
/// The Softmax function normalizes the input vector into a probability distribution.
/// It is defined as:
/// softmax(z)_i = exp(z_i) / sum(exp(z_j))
///
/// Arguments:
///     allocator: The allocator to use for the result array.
///     a: The input array.
///     axis_opt: The axis along which to compute the softmax. Defaults to the last axis.
///
/// Returns:
///     A new array containing the Softmax of the input elements.
///
/// Example:
/// ```zig
/// var a = try NDArray(f32).init(allocator, &.{2}, &.{ 1.0, 2.0 });
/// defer a.deinit(allocator);
///
/// var probs = try activations.softmax(allocator, &a, null);
/// defer probs.deinit(allocator);
/// // probs is {0.2689, 0.7311}
/// ```
pub fn softmax(allocator: Allocator, a: *const NDArray(f32), axis_opt: ?usize) !NDArray(f32) {
    const axis = axis_opt orelse (a.rank() - 1);
    if (axis >= a.rank()) return core.Error.IndexOutOfBounds;

    const result = try NDArray(f32).init(allocator, a.shape);
    const shape = a.shape;
    const rank = a.rank();

    // Calculate number of vectors to process (product of all dims except axis)
    var num_vectors: usize = 1;
    for (shape, 0..) |dim, i| {
        if (i != axis) num_vectors *= dim;
    }

    // Coordinate counter for dimensions other than axis
    var coords = try allocator.alloc(usize, rank);
    defer allocator.free(coords);
    @memset(coords, 0);

    const axis_stride = a.strides[axis];
    const axis_dim = shape[axis];

    var vec_idx: usize = 0;
    while (vec_idx < num_vectors) : (vec_idx += 1) {
        // Calculate base offset for this vector
        var base_offset: usize = 0;
        var result_base_offset: usize = 0;
        for (coords, 0..) |c, d| {
            if (d != axis) {
                base_offset += c * a.strides[d];
                result_base_offset += c * result.strides[d];
            }
        }

        // 1. Find max for numerical stability
        var max_val: f32 = -std.math.inf(f32);
        var k: usize = 0;
        while (k < axis_dim) : (k += 1) {
            const val = a.data[base_offset + k * axis_stride];
            if (val > max_val) max_val = val;
        }

        // 2. Compute exp and sum
        var sum: f32 = 0;
        k = 0;
        while (k < axis_dim) : (k += 1) {
            const val = a.data[base_offset + k * axis_stride];
            const exp_val = std.math.exp(val - max_val);
            result.data[result_base_offset + k * result.strides[axis]] = exp_val;
            sum += exp_val;
        }

        // 3. Normalize
        k = 0;
        while (k < axis_dim) : (k += 1) {
            result.data[result_base_offset + k * result.strides[axis]] /= sum;
        }

        // Increment coordinates (skipping axis)
        var dim_idx: usize = rank;
        while (dim_idx > 0) {
            dim_idx -= 1;
            if (dim_idx == axis) continue;

            coords[dim_idx] += 1;
            if (coords[dim_idx] < shape[dim_idx]) {
                break;
            }
            coords[dim_idx] = 0;
        }
    }

    return result;
}

/// Computes the ReLU activation for a Tensor.
pub fn reluTensor(allocator: Allocator, t: *Tensor(f32)) !*Tensor(f32) {
    return t.relu(allocator);
}

/// Computes the Sigmoid activation for a Tensor.
pub fn sigmoidTensor(allocator: Allocator, t: *Tensor(f32)) !*Tensor(f32) {
    return t.sigmoid(allocator);
}

/// Computes the Tanh activation for a Tensor.
pub fn tanhTensor(allocator: Allocator, t: *Tensor(f32)) !*Tensor(f32) {
    return t.tanh(allocator);
}

/// Computes the Softmax activation for a Tensor.
pub fn softmaxTensor(allocator: Allocator, t: *Tensor(f32)) !*Tensor(f32) {
    return t.softmax(allocator);
}

test "ml activations relu" {
    const allocator = std.testing.allocator;
    var a = try NDArray(f32).init(allocator, &.{3});
    defer a.deinit(allocator);
    try a.set(&.{0}, -1.0);
    try a.set(&.{1}, 0.0);
    try a.set(&.{2}, 1.0);

    var res = try relu(allocator, &a);
    defer res.deinit(allocator);

    try std.testing.expectEqual(try res.get(&.{0}), 0.0);
    try std.testing.expectEqual(try res.get(&.{1}), 0.0);
    try std.testing.expectEqual(try res.get(&.{2}), 1.0);
}

test "ml activations sigmoid" {
    const allocator = std.testing.allocator;
    var a = try NDArray(f32).init(allocator, &.{1});
    defer a.deinit(allocator);
    try a.set(&.{0}, 0.0);

    var res = try sigmoid(allocator, &a);
    defer res.deinit(allocator);

    try std.testing.expectEqual(try res.get(&.{0}), 0.5);
}

test "ml activations softmax" {
    const allocator = std.testing.allocator;
    var a = try NDArray(f32).init(allocator, &.{2});
    defer a.deinit(allocator);
    try a.set(&.{0}, 1.0);
    try a.set(&.{1}, 1.0);

    var res = try softmax(allocator, &a, 0);
    defer res.deinit(allocator);

    try std.testing.expectApproxEqAbs(try res.get(&.{0}), 0.5, 1e-6);
    try std.testing.expectApproxEqAbs(try res.get(&.{1}), 0.5, 1e-6);
}
