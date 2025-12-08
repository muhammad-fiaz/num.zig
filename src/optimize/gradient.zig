const std = @import("std");
const Allocator = std.mem.Allocator;
const core = @import("../core.zig");
const NDArray = core.NDArray;

/// Gradient Descent Optimizer.
///
/// Updates parameters using the gradient of the objective function.
///
/// Logic:
/// param = param - learning_rate * grad
pub fn GradientDescent(comptime T: type) type {
    return struct {
        const Self = @This();

        learning_rate: T,

        /// Initializes a new Gradient Descent optimizer.
        ///
        /// Arguments:
        ///     lr: The learning rate.
        pub fn init(lr: T) Self {
            return Self{ .learning_rate = lr };
        }

        /// Updates parameters in-place.
        ///
        /// Arguments:
        ///     param: The parameter array to update.
        ///     grad: The gradient array.
        pub fn step(self: Self, param: *NDArray(T), grad: NDArray(T)) !void {
            if (param.size() != grad.size()) return core.Error.ShapeMismatch;

            // Optimization: Check for contiguity
            if (param.flags().c_contiguous and grad.flags().c_contiguous) {
                for (param.data, 0..) |*p, i| {
                    p.* -= self.learning_rate * grad.data[i];
                }
            } else {
                for (param.data, 0..) |*p, i| {
                    p.* -= self.learning_rate * grad.data[i];
                }
            }
        }
    };
}

/// Adam Optimizer.
///
/// Adaptive Moment Estimation (Adam) is an optimization algorithm that adapts the learning rate
/// for each parameter.
///
/// Note: This optimizer maintains state for a SINGLE parameter array.
pub fn Adam(comptime T: type) type {
    return struct {
        const Self = @This();

        learning_rate: T,
        beta1: T,
        beta2: T,
        epsilon: T,
        m: ?NDArray(T),
        v: ?NDArray(T),
        t: usize,

        /// Initializes a new Adam optimizer.
        ///
        /// Arguments:
        ///     lr: Learning rate (default 0.001).
        ///     beta1: Exponential decay rate for first moment (default 0.9).
        ///     beta2: Exponential decay rate for second moment (default 0.999).
        ///     epsilon: Small constant for stability (default 1e-8).
        pub fn init(lr: T, beta1: T, beta2: T, epsilon: T) Self {
            return Self{
                .learning_rate = lr,
                .beta1 = beta1,
                .beta2 = beta2,
                .epsilon = epsilon,
                .m = null,
                .v = null,
                .t = 0,
            };
        }

        /// Frees resources associated with the optimizer.
        pub fn deinit(self: *Self, allocator: Allocator) void {
            if (self.m) |*m| m.deinit(allocator);
            if (self.v) |*v| v.deinit(allocator);
        }

        /// Updates parameters in-place.
        ///
        /// Arguments:
        ///     allocator: Allocator for initializing state.
        ///     param: The parameter array to update.
        ///     grad: The gradient array.
        pub fn step(self: *Self, allocator: Allocator, param: *NDArray(T), grad: NDArray(T)) !void {
            if (param.size() != grad.size()) return core.Error.ShapeMismatch;

            // Initialize state if needed
            if (self.m == null) {
                self.m = try NDArray(T).zeros(allocator, param.shape);
            }
            if (self.v == null) {
                self.v = try NDArray(T).zeros(allocator, param.shape);
            }

            self.t += 1;
            const t_cast = @as(T, @floatFromInt(self.t));

            const m_data = self.m.?.data;
            const v_data = self.v.?.data;
            const p_data = param.data;
            const g_data = grad.data;

            for (p_data, 0..) |*p, i| {
                const g = g_data[i];

                m_data[i] = self.beta1 * m_data[i] + (1.0 - self.beta1) * g;
                v_data[i] = self.beta2 * v_data[i] + (1.0 - self.beta2) * g * g;

                const m_hat = m_data[i] / (1.0 - std.math.pow(T, self.beta1, t_cast));
                const v_hat = v_data[i] / (1.0 - std.math.pow(T, self.beta2, t_cast));

                p.* -= self.learning_rate * m_hat / (std.math.sqrt(v_hat) + self.epsilon);
            }
        }
    };
}

/// RMSProp Optimizer.
///
/// Root Mean Square Propagation (RMSProp) is an adaptive learning rate method.
///
/// Note: This optimizer maintains state for a SINGLE parameter array.
pub fn RMSProp(comptime T: type) type {
    return struct {
        const Self = @This();

        learning_rate: T,
        decay_rate: T,
        epsilon: T,
        cache: ?NDArray(T),

        /// Initializes a new RMSProp optimizer.
        ///
        /// Arguments:
        ///     lr: Learning rate.
        ///     decay_rate: Decay rate (default 0.9).
        ///     epsilon: Small constant (default 1e-8).
        pub fn init(lr: T, decay_rate: T, epsilon: T) Self {
            return Self{
                .learning_rate = lr,
                .decay_rate = decay_rate,
                .epsilon = epsilon,
                .cache = null,
            };
        }

        /// Frees resources.
        pub fn deinit(self: *Self, allocator: Allocator) void {
            if (self.cache) |*c| c.deinit(allocator);
        }

        /// Updates parameters in-place.
        pub fn step(self: *Self, allocator: Allocator, param: *NDArray(T), grad: NDArray(T)) !void {
            if (param.size() != grad.size()) return core.Error.ShapeMismatch;

            if (self.cache == null) {
                self.cache = try NDArray(T).zeros(allocator, param.shape);
            }

            const c_data = self.cache.?.data;
            const p_data = param.data;
            const g_data = grad.data;

            for (p_data, 0..) |*p, i| {
                const g = g_data[i];

                // cache = decay * cache + (1 - decay) * g^2
                c_data[i] = self.decay_rate * c_data[i] + (1.0 - self.decay_rate) * g * g;

                // param = param - lr * g / (sqrt(cache) + epsilon)
                p.* -= self.learning_rate * g / (std.math.sqrt(c_data[i]) + self.epsilon);
            }
        }
    };
}

test "optimize gradient descent" {
    const allocator = std.testing.allocator;
    var param = try NDArray(f32).init(allocator, &.{2});
    defer param.deinit(allocator);
    param.data[0] = 1.0;
    param.data[1] = 2.0;

    var grad = try NDArray(f32).init(allocator, &.{2});
    defer grad.deinit(allocator);
    grad.data[0] = 0.1;
    grad.data[1] = 0.5;

    const gd = GradientDescent(f32).init(0.1);
    try gd.step(&param, grad);

    try std.testing.expectApproxEqAbs(param.data[0], 0.99, 1e-4);
    try std.testing.expectApproxEqAbs(param.data[1], 1.95, 1e-4);
}

test "optimize adam" {
    const allocator = std.testing.allocator;
    var param = try NDArray(f32).init(allocator, &.{1});
    defer param.deinit(allocator);
    param.fill(1.0);

    var grad = try NDArray(f32).init(allocator, &.{1});
    defer grad.deinit(allocator);
    grad.fill(0.1);

    var opt = Adam(f32).init(0.1, 0.9, 0.999, 1e-8);
    defer opt.deinit(allocator);

    try opt.step(allocator, &param, grad);

    // First step of Adam with zero init:
    // m = 0.1 * 0.1 = 0.01
    // v = 0.001 * 0.01 = 0.00001
    // m_hat = 0.01 / 0.1 = 0.1
    // v_hat = 0.00001 / 0.001 = 0.01
    // p = 1.0 - 0.1 * 0.1 / (0.1 + 1e-8) = 1.0 - 0.1 = 0.9
    try std.testing.expectApproxEqAbs(param.data[0], 0.9, 1e-3);
}

test "optimize rmsprop" {
    const allocator = std.testing.allocator;
    var param = try NDArray(f32).init(allocator, &.{1});
    defer param.deinit(allocator);
    param.fill(1.0);

    var grad = try NDArray(f32).init(allocator, &.{1});
    defer grad.deinit(allocator);
    grad.fill(0.1);

    var opt = RMSProp(f32).init(0.1, 0.9, 1e-8);
    defer opt.deinit(allocator);

    try opt.step(allocator, &param, grad);

    // cache = 0.9*0 + 0.1*0.01 = 0.001
    // p = 1.0 - 0.1 * 0.1 / sqrt(0.001)
    // p = 1.0 - 0.01 / 0.0316 = 1.0 - 0.316 = 0.684
    try std.testing.expectApproxEqAbs(param.data[0], 0.6837, 1e-3);
}
