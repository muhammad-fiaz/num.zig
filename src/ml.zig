/// Machine Learning module.
///
/// This module provides building blocks for creating and training neural networks.
/// It includes:
/// - `activations`: Activation functions (ReLU, Sigmoid, etc.).
/// - `loss`: Loss functions (MSE, CrossEntropy, etc.).
/// - `optim`: Optimizers (SGD, Adam, etc.).
/// - `layers`: Neural network layers (Dense, Dropout, etc.).
///
/// Example:
/// ```zig
/// const ml = num.ml;
/// // Use activation functions
/// var a = try NDArray(f32).init(allocator, &.{3}, &.{1.0, -1.0, 0.0});
/// defer a.deinit();
/// var relu = try ml.activations.relu(allocator, f32, a);
/// defer relu.deinit();
/// ```
pub const activations = @import("ml/activations.zig");
pub const loss = @import("ml/loss.zig");
pub const optim = @import("ml/optim.zig");
pub const layers = @import("ml/layers.zig");
pub const models = @import("ml/models.zig");

test {
    _ = activations;
    _ = loss;
    _ = optim;
    _ = layers;
    _ = models;
}
