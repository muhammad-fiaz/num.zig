const std = @import("std");
const core = @import("../core.zig");
const NDArray = core.NDArray;
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
    ///     param: The parameters to update.
    ///     grad: The gradients of the loss with respect to the parameters.
    ///
    /// Example:
    /// ```zig
    /// opt.update(&weights, &grads);
    /// ```
    pub fn update(self: SGD, param: *NDArray(f32), grad: *const NDArray(f32)) void {
        // param -= lr * grad
        // Assumes shapes match and contiguous for speed
        if (param.flags().c_contiguous and grad.flags().c_contiguous) {
            for (param.data, 0..) |*p, i| {
                p.* -= self.learning_rate * grad.data[i];
            }
        } else {
            // Fallback (simplified, ideally use iterators)
            for (param.data, 0..) |*p, i| {
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
    /// defer opt.deinit();
    /// ```
    pub fn init(lr: f32, momentum: f32) Momentum {
        return Momentum{ .learning_rate = lr, .momentum = momentum, .velocity = null };
    }

    /// Deinitializes the optimizer, freeing associated memory.
    pub fn deinit(self: *Momentum) void {
        if (self.velocity) |*v| v.deinit();
    }

    /// Updates the parameters using the computed gradients.
    ///
    /// Arguments:
    ///     self: The Momentum optimizer instance.
    ///     allocator: The allocator to use for initializing velocity.
    ///     param: The parameters to update.
    ///     grad: The gradients of the loss with respect to the parameters.
    ///
    /// Example:
    /// ```zig
    /// try opt.update(allocator, &weights, &grads);
    /// ```
    pub fn update(self: *Momentum, allocator: Allocator, param: *NDArray(f32), grad: *const NDArray(f32)) !void {
        if (self.velocity == null) {
            self.velocity = try NDArray(f32).zeros(allocator, param.shape);
        }

        var v = &self.velocity.?;

        // v = momentum * v - lr * grad
        // param += v

        // Optimization: Contiguous check
        if (param.flags().c_contiguous and grad.flags().c_contiguous and v.flags().c_contiguous) {
            for (v.data, 0..) |*val, i| {
                val.* = self.momentum * val.* - self.learning_rate * grad.data[i];
                param.data[i] += val.*;
            }
        } else {
            // Fallback
            for (v.data, 0..) |*val, i| {
                val.* = self.momentum * val.* - self.learning_rate * grad.data[i];
                param.data[i] += val.*;
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
    /// defer opt.deinit();
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
    pub fn deinit(self: *Adam) void {
        if (self.m) |*m| m.deinit();
        if (self.v) |*v| v.deinit();
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
    pub fn update(self: *Adam, allocator: Allocator, param: *NDArray(f32), grad: *const NDArray(f32)) !void {
        if (self.m == null) {
            self.m = try NDArray(f32).zeros(allocator, param.shape);
            self.v = try NDArray(f32).zeros(allocator, param.shape);
        }

        self.t += 1;
        const t_float = @as(f32, @floatFromInt(self.t));

        // Bias correction
        const correction1 = 1.0 - std.math.pow(f32, self.beta1, t_float);
        const correction2 = 1.0 - std.math.pow(f32, self.beta2, t_float);

        var m = &self.m.?;
        var v = &self.v.?;

        for (param.data, 0..) |*p, i| {
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
};

test "ml optim sgd" {
    const allocator = std.testing.allocator;
    var param = try NDArray(f32).init(allocator, &.{1});
    defer param.deinit();
    param.fill(1.0);

    var grad = try NDArray(f32).init(allocator, &.{1});
    defer grad.deinit();
    grad.fill(0.1);

    const opt = SGD.init(0.1);
    opt.update(&param, &grad);

    // 1.0 - 0.1 * 0.1 = 0.99
    try std.testing.expectApproxEqAbs((try param.get(&.{0})), 0.99, 1e-4);
}
