const std = @import("std");
const core = @import("../core.zig");
const autograd = @import("../autograd/tensor.zig");
const NDArray = core.NDArray;
const Tensor = autograd.Tensor;
const Allocator = std.mem.Allocator;

/// Implements the Stochastic Gradient Descent (SGD) optimizer.
///
/// SGD updates parameters using the gradient of the loss function with respect to the parameters.
///
/// Logic:
/// param = param - learning_rate * grad
pub const SGD = struct {
    learning_rate: f32,

    /// Initializes a new SGD optimizer.
    ///
    /// Arguments:
    ///     lr: The learning rate.
    ///
    /// Returns:
    ///     A new SGD optimizer instance.
    ///
    /// Example:
    /// ```zig
    /// var opt = SGD.init(0.01);
    /// ```
    pub fn init(lr: f32) SGD {
        return SGD{ .learning_rate = lr };
    }

    /// Updates the parameters using the computed gradients.
    ///
    /// Arguments:
    ///     self: The SGD optimizer instance.
    ///     param: The parameter tensor to update.
    ///
    /// Example:
    /// ```zig
    /// opt.update(weights_tensor);
    /// ```
    pub fn update(self: SGD, param: *Tensor(f32)) void {
        if (param.grad) |grad| {
            for (param.data.data, 0..) |*p, i| {
                p.* -= self.learning_rate * grad.data[i];
            }
        }
    }
};

/// Implements the Stochastic Gradient Descent (SGD) optimizer with momentum.
///
/// Momentum helps accelerate SGD in the relevant direction and dampens oscillations.
/// Note: This optimizer maintains state for a SINGLE parameter tensor.
/// You must create a separate instance for each parameter tensor (e.g., one for weights, one for bias).
///
/// Logic:
/// velocity = momentum * velocity - learning_rate * grad
/// param = param + velocity
pub const Momentum = struct {
    learning_rate: f32,
    momentum: f32,
    velocity: ?NDArray(f32),

    /// Initializes a new Momentum optimizer.
    ///
    /// Arguments:
    ///     lr: The learning rate.
    ///     momentum: The momentum factor (usually around 0.9).
    ///
    /// Returns:
    ///     A new Momentum optimizer instance.
    ///
    /// Example:
    /// ```zig
    /// var opt = Momentum.init(0.01, 0.9);
    /// defer opt.deinit(allocator);
    /// ```
    pub fn init(lr: f32, momentum: f32) Momentum {
        return Momentum{ .learning_rate = lr, .momentum = momentum, .velocity = null };
    }

    /// Deinitializes the optimizer, freeing associated memory.
    pub fn deinit(self: *Momentum, allocator: Allocator) void {
        if (self.velocity) |*v| v.deinit(allocator);
    }

    /// Updates the parameters using the computed gradients.
    ///
    /// Arguments:
    ///     self: The Momentum optimizer instance.
    ///     allocator: The allocator to use for initializing velocity.
    ///     param: The parameter tensor to update.
    ///
    /// Example:
    /// ```zig
    /// try opt.update(allocator, param_tensor);
    /// ```
    pub fn update(self: *Momentum, allocator: Allocator, param: *Tensor(f32)) !void {
        if (param.grad) |grad| {
            if (self.velocity == null) {
                self.velocity = try NDArray(f32).zeros(allocator, param.data.shape);
            }

            for (param.data.data, 0..) |*p, i| {
                const g = grad.data[i];
                const v = self.velocity.?.data[i];
                const new_v = self.momentum * v - self.learning_rate * g;
                self.velocity.?.data[i] = new_v;
                p.* += new_v;
            }
        }
    }
};

/// Implements the Adam optimizer.
///
/// Adam is an adaptive learning rate optimization algorithm that's designed specifically for training deep neural networks.
/// Note: This optimizer maintains state for a SINGLE parameter tensor.
///
/// Logic:
/// m = beta1 * m + (1 - beta1) * grad
/// v = beta2 * v + (1 - beta2) * grad^2
/// m_hat = m / (1 - beta1^t)
/// v_hat = v / (1 - beta2^t)
/// param = param - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
pub const Adam = struct {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    m: ?NDArray(f32),
    v: ?NDArray(f32),
    t: usize,

    /// Initializes a new Adam optimizer.
    ///
    /// Arguments:
    ///     lr: The learning rate (default 0.001).
    ///     beta1: The exponential decay rate for the first moment estimates (default 0.9).
    ///     beta2: The exponential decay rate for the second moment estimates (default 0.999).
    ///     epsilon: A small constant for numerical stability (default 1e-8).
    ///
    /// Returns:
    ///     A new Adam optimizer instance.
    ///
    /// Example:
    /// ```zig
    /// var opt = Adam.init(0.001, 0.9, 0.999, 1e-8);
    /// defer opt.deinit(allocator);
    /// ```
    pub fn init(lr: f32, beta1: f32, beta2: f32, epsilon: f32) Adam {
        return Adam{
            .learning_rate = lr,
            .beta1 = beta1,
            .beta2 = beta2,
            .epsilon = epsilon,
            .m = null,
            .v = null,
            .t = 0,
        };
    }

    /// Deinitializes the optimizer, freeing associated memory.
    pub fn deinit(self: *Adam, allocator: Allocator) void {
        if (self.m) |*m| m.deinit(allocator);
        if (self.v) |*v| v.deinit(allocator);
    }

    /// Updates the parameters using the computed gradients.
    ///
    /// Arguments:
    ///     self: The Adam optimizer instance.
    ///     allocator: The allocator to use for initializing moments.
    ///     param: The parameters to update.
    ///     grad: The gradients of the loss with respect to the parameters.
    ///
    /// Example:
    /// ```zig
    /// try opt.update(allocator, &weights, &grads);
    /// ```
    pub fn update(self: *Adam, allocator: Allocator, param: *Tensor(f32)) !void {
        if (param.grad) |grad| {
            if (self.m == null) {
                self.m = try NDArray(f32).zeros(allocator, param.data.shape);
                self.v = try NDArray(f32).zeros(allocator, param.data.shape);
            }

            self.t += 1;
            const t_float = @as(f32, @floatFromInt(self.t));

            // Bias correction
            const correction1 = 1.0 - std.math.pow(f32, self.beta1, t_float);
            const correction2 = 1.0 - std.math.pow(f32, self.beta2, t_float);

            var m = &self.m.?;
            var v = &self.v.?;

            for (param.data.data, 0..) |*p, i| {
                const g = grad.data[i];

                // Update biased first moment estimate
                m.data[i] = self.beta1 * m.data[i] + (1.0 - self.beta1) * g;

                // Update biased second raw moment estimate
                v.data[i] = self.beta2 * v.data[i] + (1.0 - self.beta2) * g * g;

                // Compute bias-corrected first moment estimate
                const m_hat = m.data[i] / correction1;

                // Compute bias-corrected second raw moment estimate
                const v_hat = v.data[i] / correction2;

                // Update parameters
                p.* -= self.learning_rate * m_hat / (@sqrt(v_hat) + self.epsilon);
            }
        }
    }
};

test "ml optim sgd" {
    const allocator = std.testing.allocator;
    var param_data = try NDArray(f32).init(allocator, &.{1});
    param_data.fill(1.0);
    var param = try Tensor(f32).init(allocator, param_data, true);
    defer param.deinit(allocator);

    param.grad.?.fill(0.1);

    const opt = SGD.init(0.1);
    opt.update(param);

    // 1.0 - 0.1 * 0.1 = 0.99
    try std.testing.expectApproxEqAbs((try param.data.get(&.{0})), 0.99, 1e-4);
}

test "ml optim momentum" {
    const allocator = std.testing.allocator;
    var param_data = try NDArray(f32).init(allocator, &.{1});
    param_data.fill(1.0);
    var param = try Tensor(f32).init(allocator, param_data, true);
    defer param.deinit(allocator);

    param.grad.?.fill(0.1);

    var opt = Momentum.init(0.1, 0.9);
    defer opt.deinit(allocator);

    try opt.update(allocator, param);

    // v = 0.9*0 - 0.1*0.1 = -0.01
    // p = 1.0 + (-0.01) = 0.99
    try std.testing.expectApproxEqAbs((try param.data.get(&.{0})), 0.99, 1e-4);

    try opt.update(allocator, param);
    // v = 0.9*(-0.01) - 0.1*0.1 = -0.009 - 0.01 = -0.019
    // p = 0.99 - 0.019 = 0.971
    try std.testing.expectApproxEqAbs((try param.data.get(&.{0})), 0.971, 1e-4);
}

test "ml optim adam" {
    const allocator = std.testing.allocator;
    var param_data = try NDArray(f32).init(allocator, &.{1});
    param_data.fill(1.0);
    var param = try Tensor(f32).init(allocator, param_data, true);
    defer param.deinit(allocator);

    param.grad.?.fill(0.1);

    var opt = Adam.init(0.001, 0.9, 0.999, 1e-8);
    defer opt.deinit(allocator);

    try opt.update(allocator, param);
    // Just check it changed
    try std.testing.expect((try param.data.get(&.{0})) < 1.0);
}
