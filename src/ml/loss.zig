const std = @import("std");
const core = @import("../core.zig");
const autograd = @import("../autograd/tensor.zig");
const NDArray = core.NDArray;
const Tensor = autograd.Tensor;
const Allocator = std.mem.Allocator;

/// Computes the Mean Squared Error (MSE) loss between true and predicted values.
///
/// MSE is calculated as the average of the squared differences between the estimated values and the actual value.
///
/// Logic:
/// loss = mean((y_true - y_pred)^2)
///
/// Arguments:
///     allocator: The allocator (unused, but kept for API consistency).
///     y_true: The array of true values.
///     y_pred: The array of predicted values.
///
/// Returns:
///     The computed MSE loss.
///
/// Example:
/// ```zig
/// var y_true = try NDArray(f32).init(allocator, &.{2}, &.{ 1.0, 0.0 });
/// defer y_true.deinit();
/// var y_pred = try NDArray(f32).init(allocator, &.{2}, &.{ 0.9, 0.1 });
/// defer y_pred.deinit();
///
/// const loss_val = try loss.mse(allocator, &y_true, &y_pred);
/// // loss_val is 0.01
/// ```
pub fn mse(allocator: Allocator, y_true: *const NDArray(f32), y_pred: *const NDArray(f32)) !f32 {
    if (y_true.size() != y_pred.size()) return core.Error.ShapeMismatch;

    var sum_sq_diff: f32 = 0;

    // Optimization: Check for contiguity for faster access
    if (y_true.flags().c_contiguous and y_pred.flags().c_contiguous) {
        for (y_true.data, 0..) |val, i| {
            const diff = val - y_pred.data[i];
            sum_sq_diff += diff * diff;
        }
    } else {
        // Fallback for non-contiguous arrays
        var iter = try core.NdIterator.init(allocator, y_true.shape); // Allocates!
        defer iter.deinit(allocator);

        // This is slow because of allocation and iteration.
        // Ideally we should have a non-allocating iterator or just iterate indices if shapes match.
        // Since shapes match, we can iterate 0..size and map to coords? No, strides differ.
        // We need two iterators or just use get() with one iterator if shapes are identical.
        while (iter.next()) |coords| {
            const val_true = try y_true.get(coords);
            const val_pred = try y_pred.get(coords);
            const diff = val_true - val_pred;
            sum_sq_diff += diff * diff;
        }
    }

    return sum_sq_diff / @as(f32, @floatFromInt(y_true.size()));
}

/// Computes the Categorical Cross-Entropy loss between true labels and predicted probabilities.
///
/// This loss function is commonly used in multi-class classification tasks.
/// It assumes `y_pred` contains probabilities (e.g., from softmax).
///
/// Logic:
/// loss = -sum(y_true * log(y_pred)) / batch_size
///
/// Arguments:
///     allocator: The allocator (unused, but kept for API consistency).
///     y_true: The array of true labels (one-hot encoded).
///     y_pred: The array of predicted probabilities.
///
/// Returns:
///     The computed Categorical Cross-Entropy loss.
///
/// Example:
/// ```zig
/// var y_true = try NDArray(f32).init(allocator, &.{2}, &.{ 1.0, 0.0 });
/// defer y_true.deinit();
/// var y_pred = try NDArray(f32).init(allocator, &.{2}, &.{ 0.9, 0.1 });
/// defer y_pred.deinit();
///
/// const loss_val = try loss.categoricalCrossEntropy(allocator, &y_true, &y_pred);
/// ```
pub fn categoricalCrossEntropy(allocator: Allocator, y_true: *const NDArray(f32), y_pred: *const NDArray(f32)) !f32 {
    if (y_true.size() != y_pred.size()) return core.Error.ShapeMismatch;

    var sum: f32 = 0;
    const epsilon: f32 = 1e-7;

    if (y_true.flags().c_contiguous and y_pred.flags().c_contiguous) {
        for (y_true.data, 0..) |val, i| {
            var pred = y_pred.data[i];
            if (pred < epsilon) pred = epsilon;
            if (pred > 1.0 - epsilon) pred = 1.0 - epsilon;
            sum += -val * std.math.log(f32, std.math.e, pred);
        }
    } else {
        var iter = try core.NdIterator.init(allocator, y_true.shape);
        defer iter.deinit(allocator);
        while (iter.next()) |coords| {
            const val = try y_true.get(coords);
            var pred = try y_pred.get(coords);
            if (pred < epsilon) pred = epsilon;
            if (pred > 1.0 - epsilon) pred = 1.0 - epsilon;
            sum += -val * std.math.log(f32, std.math.e, pred);
        }
    }

    const batch_size = if (y_true.rank() > 0) y_true.shape[0] else 1;
    return sum / @as(f32, @floatFromInt(batch_size));
}

/// Computes the Hinge loss.
///
/// Used for SVMs.
///
/// Logic:
/// loss = mean(max(0, 1 - y_true * y_pred))
///
/// Arguments:
///     allocator: The allocator.
///     y_true: True labels (-1 or 1).
///     y_pred: Predicted values.
///
/// Returns:
///     The computed loss.
///
/// Example:
/// ```zig
/// var y_true = try NDArray(f32).init(allocator, &.{2}, &.{ 1.0, -1.0 });
/// defer y_true.deinit();
/// var y_pred = try NDArray(f32).init(allocator, &.{2}, &.{ 0.8, -0.5 });
/// defer y_pred.deinit();
///
/// const loss_val = try loss.hinge(allocator, &y_true, &y_pred);
/// ```
pub fn hinge(allocator: Allocator, y_true: *const NDArray(f32), y_pred: *const NDArray(f32)) !f32 {
    _ = allocator;
    if (y_true.size() != y_pred.size()) return core.Error.ShapeMismatch;

    var sum: f32 = 0;
    for (y_true.data, 0..) |val, i| {
        const pred = y_pred.data[i];
        const loss_val = 1.0 - val * pred;
        if (loss_val > 0) {
            sum += loss_val;
        }
    }

    return sum / @as(f32, @floatFromInt(y_true.size()));
}

/// Computes the Mean Absolute Error (MAE) loss between true and predicted values.
///
/// MAE is calculated as the average of the absolute differences between the estimated values and the actual value.
///
/// Logic:
/// loss = mean(abs(y_true - y_pred))
///
/// Arguments:
///     allocator: The allocator (unused, but kept for API consistency).
///     y_true: The array of true values.
///     y_pred: The array of predicted values.
///
/// Returns:
///     The computed MAE loss.
///
/// Example:
/// ```zig
/// var y_true = try NDArray(f32).init(allocator, &.{3}, &.{ 1.0, 2.0, 3.0 });
/// defer y_true.deinit();
/// var y_pred = try NDArray(f32).init(allocator, &.{3}, &.{ 1.0, 2.0, 2.0 });
/// defer y_pred.deinit();
///
/// const mae_val = try loss.mae(allocator, &y_true, &y_pred);
/// // mae_val is 0.3333 ( (0 + 0 + 1) / 3 )
/// ```
pub fn mae(allocator: Allocator, y_true: *const NDArray(f32), y_pred: *const NDArray(f32)) !f32 {
    _ = allocator;
    if (y_true.size() != y_pred.size()) return core.Error.ShapeMismatch;

    var sum_abs_diff: f32 = 0;
    for (y_true.data, 0..) |val, i| {
        sum_abs_diff += @abs(val - y_pred.data[i]);
    }
    return sum_abs_diff / @as(f32, @floatFromInt(y_true.size()));
}

/// Computes the Binary Cross-Entropy loss between true labels and predicted probabilities.
///
/// This loss function is commonly used in binary classification tasks.
///
/// Logic:
/// loss = -mean(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
///
/// Arguments:
///     allocator: The allocator (unused, but kept for API consistency).
///     y_true: The array of true labels.
///     y_pred: The array of predicted probabilities.
///
/// Returns:
///     The computed Binary Cross-Entropy loss.
///
/// Example:
/// ```zig
/// var y_true = try NDArray(f32).init(allocator, &.{2}, &.{ 1.0, 0.0 });
/// defer y_true.deinit();
/// var y_pred = try NDArray(f32).init(allocator, &.{2}, &.{ 0.9, 0.1 });
/// defer y_pred.deinit();
///
/// const bce_val = try loss.binaryCrossEntropy(allocator, &y_true, &y_pred);
/// ```
pub fn binaryCrossEntropy(allocator: Allocator, y_true: *const NDArray(f32), y_pred: *const NDArray(f32)) !f32 {
    _ = allocator;
    if (y_true.size() != y_pred.size()) return core.Error.ShapeMismatch;

    var sum: f32 = 0;
    const epsilon: f32 = 1e-7;

    for (y_true.data, 0..) |val, i| {
        var pred = y_pred.data[i];
        if (pred < epsilon) pred = epsilon;
        if (pred > 1.0 - epsilon) pred = 1.0 - epsilon;

        sum += -(val * std.math.log(f32, std.math.e, pred) + (1.0 - val) * std.math.log(f32, std.math.e, 1.0 - pred));
    }

    return sum / @as(f32, @floatFromInt(y_true.size()));
}

/// Computes the MSE loss for Tensors.
pub fn mseTensor(allocator: Allocator, y_pred: *Tensor(f32), y_true: *Tensor(f32)) !*Tensor(f32) {
    return y_pred.mse_loss(allocator, y_true);
}

/// Computes the Cross Entropy loss for Tensors.
pub fn crossEntropyTensor(allocator: Allocator, y_pred: *Tensor(f32), y_true: *Tensor(f32)) !*Tensor(f32) {
    return y_pred.cross_entropy_loss(allocator, y_true);
}

test "ml loss mse" {
    const allocator = std.testing.allocator;
    var true_vals = try NDArray(f32).init(allocator, &.{2});
    defer true_vals.deinit(allocator);
    try true_vals.set(&.{0}, 1.0);
    try true_vals.set(&.{1}, 0.0);

    var pred_vals = try NDArray(f32).init(allocator, &.{2});
    defer pred_vals.deinit(allocator);
    try pred_vals.set(&.{0}, 0.9);
    try pred_vals.set(&.{1}, 0.1);

    // MSE = ((1-0.9)^2 + (0-0.1)^2) / 2 = (0.01 + 0.01) / 2 = 0.01
    const l = try mse(allocator, &true_vals, &pred_vals);
    try std.testing.expectApproxEqAbs(l, 0.01, 1e-4);
}

test "ml loss cross entropy" {
    const allocator = std.testing.allocator;
    var true_vals = try NDArray(f32).init(allocator, &.{2});
    defer true_vals.deinit(allocator);
    try true_vals.set(&.{0}, 1.0);
    try true_vals.set(&.{1}, 0.0);

    var pred_vals = try NDArray(f32).init(allocator, &.{2});
    defer pred_vals.deinit(allocator);
    try pred_vals.set(&.{0}, 0.9);
    try pred_vals.set(&.{1}, 0.1);

    // BCE = -(1*log(0.9) + 0*log(1-0.9) + 0*log(0.1) + 1*log(1-0.1)) / 2
    //     = -(log(0.9) + log(0.9)) / 2 = -log(0.9) ~= 0.10536
    const l = try binaryCrossEntropy(allocator, &true_vals, &pred_vals);
    try std.testing.expectApproxEqAbs(l, 0.10536, 1e-4);
}
