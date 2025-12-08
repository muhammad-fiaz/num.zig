const std = @import("std");
const Allocator = std.mem.Allocator;
const core = @import("../core.zig");
const linalg = @import("../linalg.zig");
const reduction = @import("../reduction.zig");
const NDArray = core.NDArray;

/// Represents a node in the computation graph for automatic differentiation.
///
/// A Tensor wraps an NDArray and tracks operations to enable gradient computation (backpropagation).
/// It supports dynamic graph construction.
pub fn Tensor(comptime T: type) type {
    return struct {
        const Self = @This();

        /// The underlying data of the tensor.
        data: NDArray(T),
        /// The gradient of the tensor. Populated after `backward()` is called.
        grad: ?NDArray(T),
        /// Whether this tensor requires gradient computation.
        requires_grad: bool,
        /// Parent tensors in the computation graph.
        parents: std.ArrayListUnmanaged(*Self),
        /// The operation that produced this tensor.
        op: OpType,
        /// Context for the backward pass (e.g., saved tensors).
        ctx: ?*anyopaque,

        /// Supported operations for the computation graph.
        pub const OpType = enum {
            None,
            Add,
            Mul,
            Sub,
            Div,
            MatMul,
            ReLU,
            Sigmoid,
            Tanh,
            Softmax,
            MSE,
            CrossEntropy,
        };

        /// Initializes a new Tensor.
        ///
        /// Arguments:
        ///     allocator: The allocator to use.
        ///     data: The NDArray data. The Tensor takes ownership.
        ///     requires_grad: If true, gradients will be computed for this tensor.
        pub fn init(allocator: Allocator, data: NDArray(T), requires_grad: bool) !*Self {
            const self = try allocator.create(Self);
            const data_mut = data;
            // errdefer data_mut.deinit(allocator); // Handled by caller if init fails before assignment?
            // Actually if allocator.create fails, data is not freed. Caller should handle.
            // But if NDArray.zeros fails (for grad), we need to cleanup.

            errdefer allocator.destroy(self);

            const grad = if (requires_grad) try NDArray(T).zeros(allocator, data.shape) else null;

            self.* = Self{
                .data = data_mut,
                .grad = grad,
                .requires_grad = requires_grad,
                .parents = .{},
                .op = .None,
                .ctx = null,
            };
            return self;
        }

        /// Frees resources associated with the Tensor.
        pub fn deinit(self: *Self, allocator: Allocator) void {
            self.data.deinit(allocator);
            if (self.grad) |*g| g.deinit(allocator);
            self.parents.deinit(allocator);
            allocator.destroy(self);
        }

        /// Performs the backward pass to compute gradients.
        ///
        /// This method traverses the computation graph in reverse topological order
        /// and accumulates gradients for all tensors with `requires_grad=true`.
        pub fn backward(self: *Self, allocator: Allocator) !void {
            if (!self.requires_grad) return;

            // Seed gradient with 1s if not already set (usually for the scalar loss)
            if (self.grad) |*g| {
                var all_zeros = true;
                for (g.data) |val| {
                    if (val != 0) {
                        all_zeros = false;
                        break;
                    }
                }
                if (all_zeros) {
                    g.fill(1);
                }
            }

            // Topological sort
            var topo = std.ArrayListUnmanaged(*Self){};
            defer topo.deinit(allocator);
            var visited = std.AutoHashMapUnmanaged(*Self, void){};
            defer visited.deinit(allocator);

            try self.buildTopo(allocator, &topo, &visited);

            // Reverse iterate
            var i = topo.items.len;
            while (i > 0) {
                i -= 1;
                const node = topo.items[i];
                try node.backwardStep(allocator);
            }
        }

        fn buildTopo(self: *Self, allocator: Allocator, topo: *std.ArrayListUnmanaged(*Self), visited: *std.AutoHashMapUnmanaged(*Self, void)) !void {
            if (visited.contains(self)) return;
            try visited.put(allocator, self, {});

            for (self.parents.items) |parent| {
                try parent.buildTopo(allocator, topo, visited);
            }
            try topo.append(allocator, self);
        }

        fn backwardStep(self: *Self, allocator: Allocator) !void {
            if (!self.requires_grad) return;
            const g = self.grad orelse return;

            switch (self.op) {
                .Add => {
                    // z = x + y => dz/dx = 1 * dz/dz, dz/dy = 1 * dz/dz
                    for (self.parents.items) |parent| {
                        if (parent.requires_grad and parent.grad != null) {
                            // Check if shapes match
                            if (std.mem.eql(usize, parent.grad.?.shape, g.shape)) {
                                for (parent.grad.?.data, 0..) |*pg, i| {
                                    pg.* += g.data[i];
                                }
                            } else {
                                // Handle broadcast reduction (N, M) -> (1, M)
                                if (parent.grad.?.rank() == 2 and parent.grad.?.shape[0] == 1 and
                                    parent.grad.?.shape[1] == g.shape[1] and g.rank() == 2)
                                {
                                    const N = g.shape[0];
                                    const M = g.shape[1];
                                    for (0..N) |r| {
                                        for (0..M) |c| {
                                            parent.grad.?.data[c] += g.data[r * M + c];
                                        }
                                    }
                                }
                                // Handle broadcast reduction (N, M) -> (M,)
                                else if (parent.grad.?.rank() == 1 and parent.grad.?.shape[0] == g.shape[1] and g.rank() == 2) {
                                    const N = g.shape[0];
                                    const M = g.shape[1];
                                    for (0..N) |r| {
                                        for (0..M) |c| {
                                            parent.grad.?.data[c] += g.data[r * M + c];
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                .Sub => {
                    // z = x - y => dz/dx = 1, dz/dy = -1
                    if (self.parents.items.len == 2) {
                        const x = self.parents.items[0];
                        const y = self.parents.items[1];
                        if (x.requires_grad and x.grad != null) {
                            for (x.grad.?.data, 0..) |*pg, i| {
                                pg.* += g.data[i];
                            }
                        }
                        if (y.requires_grad and y.grad != null) {
                            for (y.grad.?.data, 0..) |*pg, i| {
                                pg.* -= g.data[i];
                            }
                        }
                    }
                },
                .Mul => {
                    // z = x * y => dz/dx = y * dz/dz, dz/dy = x * dz/dz
                    if (self.parents.items.len == 2) {
                        const x = self.parents.items[0];
                        const y = self.parents.items[1];

                        if (x.requires_grad and x.grad != null) {
                            for (x.grad.?.data, 0..) |*pg, i| {
                                pg.* += y.data.data[i] * g.data[i];
                            }
                        }
                        if (y.requires_grad and y.grad != null) {
                            for (y.grad.?.data, 0..) |*pg, i| {
                                pg.* += x.data.data[i] * g.data[i];
                            }
                        }
                    }
                },
                .Div => {
                    // z = x / y => dz/dx = 1/y, dz/dy = -x/y^2
                    if (self.parents.items.len == 2) {
                        const x = self.parents.items[0];
                        const y = self.parents.items[1];

                        if (x.requires_grad and x.grad != null) {
                            for (x.grad.?.data, 0..) |*pg, i| {
                                pg.* += (1.0 / y.data.data[i]) * g.data[i];
                            }
                        }
                        if (y.requires_grad and y.grad != null) {
                            for (y.grad.?.data, 0..) |*pg, i| {
                                const y_val = y.data.data[i];
                                pg.* += (-x.data.data[i] / (y_val * y_val)) * g.data[i];
                            }
                        }
                    }
                },
                .MatMul => {
                    // Z = X @ Y
                    // dL/dX = dL/dZ @ Y^T
                    // dL/dY = X^T @ dL/dZ
                    if (self.parents.items.len == 2) {
                        const x = self.parents.items[0];
                        const y = self.parents.items[1];

                        if (x.requires_grad and x.grad != null) {
                            // dX = G @ Y.T
                            var yt = try y.data.transpose(allocator);
                            defer yt.deinit(allocator);
                            var dx = try linalg.matmul(T, allocator, &g, &yt);
                            defer dx.deinit(allocator);

                            for (x.grad.?.data, 0..) |*pg, i| {
                                pg.* += dx.data[i];
                            }
                        }
                        if (y.requires_grad and y.grad != null) {
                            // dY = X.T @ G
                            var xt = try x.data.transpose(allocator);
                            defer xt.deinit(allocator);
                            var dy = try linalg.matmul(T, allocator, &xt, &g);
                            defer dy.deinit(allocator);

                            for (y.grad.?.data, 0..) |*pg, i| {
                                pg.* += dy.data[i];
                            }
                        }
                    }
                },
                .ReLU => {
                    // y = max(0, x)
                    // dy/dx = 1 if x > 0 else 0
                    if (self.parents.items.len == 1) {
                        const x = self.parents.items[0];
                        if (x.requires_grad and x.grad != null) {
                            for (x.grad.?.data, 0..) |*pg, i| {
                                if (x.data.data[i] > 0) {
                                    pg.* += g.data[i];
                                }
                            }
                        }
                    }
                },
                .Sigmoid => {
                    // y = sigmoid(x)
                    // dy/dx = y * (1 - y)
                    if (self.parents.items.len == 1) {
                        const x = self.parents.items[0];
                        if (x.requires_grad and x.grad != null) {
                            for (x.grad.?.data, 0..) |*pg, i| {
                                const y_val = self.data.data[i];
                                pg.* += y_val * (1.0 - y_val) * g.data[i];
                            }
                        }
                    }
                },
                .Tanh => {
                    // y = tanh(x)
                    // dy/dx = 1 - y^2
                    if (self.parents.items.len == 1) {
                        const x = self.parents.items[0];
                        if (x.requires_grad and x.grad != null) {
                            for (x.grad.?.data, 0..) |*pg, i| {
                                const y_val = self.data.data[i];
                                pg.* += (1.0 - y_val * y_val) * g.data[i];
                            }
                        }
                    }
                },
                .Softmax => {
                    if (self.parents.items.len == 1) {
                        const x = self.parents.items[0];
                        if (x.requires_grad and x.grad != null) {
                            const y_data = self.data;
                            const g_data = g;

                            if (y_data.rank() == 1) {
                                var sum_yg: T = 0;
                                for (y_data.data, 0..) |y_val, i| {
                                    sum_yg += y_val * g_data.data[i];
                                }
                                for (x.grad.?.data, 0..) |*pg, i| {
                                    const y_val = y_data.data[i];
                                    pg.* += y_val * (g_data.data[i] - sum_yg);
                                }
                            } else if (y_data.rank() == 2) {
                                const rows = y_data.shape[0];
                                const cols = y_data.shape[1];
                                for (0..rows) |r| {
                                    var sum_yg: T = 0;
                                    for (0..cols) |c| {
                                        const idx = r * cols + c;
                                        sum_yg += y_data.data[idx] * g_data.data[idx];
                                    }
                                    for (0..cols) |c| {
                                        const idx = r * cols + c;
                                        const y_val = y_data.data[idx];
                                        x.grad.?.data[idx] += y_val * (g_data.data[idx] - sum_yg);
                                    }
                                }
                            }
                        }
                    }
                },
                .MSE => {
                    if (self.parents.items.len == 2) {
                        const pred = self.parents.items[0];
                        const target = self.parents.items[1];
                        if (pred.requires_grad and pred.grad != null) {
                            const n = @as(T, @floatFromInt(pred.data.size()));
                            const grad_scale = g.data[0] * 2.0 / n;
                            for (pred.grad.?.data, 0..) |*pg, i| {
                                pg.* += grad_scale * (pred.data.data[i] - target.data.data[i]);
                            }
                        }
                    }
                },
                .CrossEntropy => {
                    if (self.parents.items.len == 2) {
                        const pred = self.parents.items[0];
                        const target = self.parents.items[1];
                        if (pred.requires_grad and pred.grad != null) {
                            var batch_size: usize = 1;
                            if (pred.data.rank() > 1) {
                                batch_size = pred.data.shape[0];
                            }
                            const grad_scale = g.data[0] / @as(T, @floatFromInt(batch_size));
                            const epsilon: T = 1e-7;

                            for (pred.grad.?.data, 0..) |*pg, i| {
                                var p = pred.data.data[i];
                                if (p < epsilon) p = epsilon;
                                if (p > 1.0 - epsilon) p = 1.0 - epsilon;
                                pg.* += grad_scale * (-target.data.data[i] / p);
                            }
                        }
                    }
                },
                .None => {},
            }
        }
        /// Adds two tensors.
        pub fn add(self: *Self, allocator: Allocator, other: *Self) !*Self {
            var res_data: NDArray(T) = undefined;
            const same_shape = std.mem.eql(usize, self.data.shape, other.data.shape);

            if (same_shape) {
                res_data = try NDArray(T).init(allocator, self.data.shape);
                for (res_data.data, 0..) |*d, i| {
                    d.* = self.data.data[i] + other.data.data[i];
                }
            } else {
                // Check for (N, M) + (1, M) broadcast (Dense layer bias)
                if (self.data.rank() == 2 and other.data.rank() == 2 and
                    other.data.shape[0] == 1 and other.data.shape[1] == self.data.shape[1])
                {
                    res_data = try NDArray(T).init(allocator, self.data.shape);
                    const N = self.data.shape[0];
                    const M = self.data.shape[1];
                    for (0..N) |r| {
                        for (0..M) |c| {
                            res_data.data[r * M + c] = self.data.data[r * M + c] + other.data.data[c];
                        }
                    }
                }
                // Check for (N, M) + (M,) broadcast
                else if (self.data.rank() == 2 and other.data.rank() == 1 and
                    other.data.shape[0] == self.data.shape[1])
                {
                    res_data = try NDArray(T).init(allocator, self.data.shape);
                    const N = self.data.shape[0];
                    const M = self.data.shape[1];
                    for (0..N) |r| {
                        for (0..M) |c| {
                            res_data.data[r * M + c] = self.data.data[r * M + c] + other.data.data[c];
                        }
                    }
                } else {
                    return error.ShapeMismatch;
                }
            }

            var result = try Self.init(allocator, res_data, self.requires_grad or other.requires_grad);
            errdefer result.deinit(allocator);

            result.op = .Add;
            try result.parents.append(allocator, self);
            try result.parents.append(allocator, other);
            return result;
        }

        /// Multiplies two tensors element-wise.
        pub fn mul(self: *Self, allocator: Allocator, other: *Self) !*Self {
            var res_data = try NDArray(T).init(allocator, self.data.shape);
            errdefer res_data.deinit(allocator);
            for (res_data.data, 0..) |*d, i| {
                d.* = self.data.data[i] * other.data.data[i];
            }

            var result = try Self.init(allocator, res_data, self.requires_grad or other.requires_grad);
            errdefer result.deinit(allocator);
            result.op = .Mul;
            try result.parents.append(allocator, self);
            try result.parents.append(allocator, other);
            return result;
        }

        /// Matrix multiplication of two tensors.
        pub fn matmul(self: *Self, allocator: Allocator, other: *Self) !*Self {
            var res_data = try linalg.matmul(T, allocator, &self.data, &other.data);
            errdefer res_data.deinit(allocator);

            var result = try Self.init(allocator, res_data, self.requires_grad or other.requires_grad);
            errdefer result.deinit(allocator);
            result.op = .MatMul;
            try result.parents.append(allocator, self);
            try result.parents.append(allocator, other);
            return result;
        }

        pub fn relu(self: *Self, allocator: Allocator) !*Self {
            var res_data = try NDArray(T).init(allocator, self.data.shape);
            errdefer res_data.deinit(allocator);
            for (res_data.data, 0..) |*d, i| {
                const val = self.data.data[i];
                d.* = if (val > 0) val else 0;
            }

            var result = try Self.init(allocator, res_data, self.requires_grad);
            errdefer result.deinit(allocator);
            result.op = .ReLU;
            try result.parents.append(allocator, self);
            return result;
        }

        pub fn sigmoid(self: *Self, allocator: Allocator) !*Self {
            var res_data = try NDArray(T).init(allocator, self.data.shape);
            errdefer res_data.deinit(allocator);
            for (res_data.data, 0..) |*d, i| {
                const val = self.data.data[i];
                d.* = 1.0 / (1.0 + std.math.exp(-val));
            }

            var result = try Self.init(allocator, res_data, self.requires_grad);
            errdefer result.deinit(allocator);
            result.op = .Sigmoid;
            try result.parents.append(allocator, self);
            return result;
        }

        pub fn tanh(self: *Self, allocator: Allocator) !*Self {
            var res_data = try NDArray(T).init(allocator, self.data.shape);
            errdefer res_data.deinit(allocator);
            for (res_data.data, 0..) |*d, i| {
                const val = self.data.data[i];
                d.* = std.math.tanh(val);
            }

            var result = try Self.init(allocator, res_data, self.requires_grad);
            errdefer result.deinit(allocator);
            result.op = .Tanh;
            try result.parents.append(allocator, self);
            return result;
        }

        pub fn softmax(self: *Self, allocator: Allocator) !*Self {
            var res_data = try NDArray(T).init(allocator, self.data.shape);
            errdefer res_data.deinit(allocator);

            if (self.data.rank() == 1) {
                var sum_exp: T = 0;
                for (self.data.data) |val| {
                    sum_exp += std.math.exp(val);
                }
                for (res_data.data, 0..) |*d, i| {
                    d.* = std.math.exp(self.data.data[i]) / sum_exp;
                }
            } else if (self.data.rank() == 2) {
                const rows = self.data.shape[0];
                const cols = self.data.shape[1];

                for (0..rows) |r| {
                    var sum_exp: T = 0;
                    for (0..cols) |c| {
                        const val = self.data.data[r * cols + c];
                        sum_exp += std.math.exp(val);
                    }
                    for (0..cols) |c| {
                        const val = self.data.data[r * cols + c];
                        res_data.data[r * cols + c] = std.math.exp(val) / sum_exp;
                    }
                }
            } else {
                return error.NotImplemented;
            }

            var result = try Self.init(allocator, res_data, self.requires_grad);
            errdefer result.deinit(allocator);
            result.op = .Softmax;
            try result.parents.append(allocator, self);
            return result;
        }

        pub fn mse_loss(self: *Self, allocator: Allocator, target: *Self) !*Self {
            var sum_sq_diff: T = 0;
            for (self.data.data, 0..) |val, i| {
                const diff = val - target.data.data[i];
                sum_sq_diff += diff * diff;
            }
            const loss = sum_sq_diff / @as(T, @floatFromInt(self.data.size()));

            var res_data = try NDArray(T).init(allocator, &.{1});
            res_data.data[0] = loss;

            var result = try Self.init(allocator, res_data, self.requires_grad);
            errdefer result.deinit(allocator);
            result.op = .MSE;
            try result.parents.append(allocator, self);
            try result.parents.append(allocator, target);
            return result;
        }

        pub fn cross_entropy_loss(self: *Self, allocator: Allocator, target: *Self) !*Self {
            var sum_loss: T = 0;
            const epsilon: T = 1e-7;

            for (self.data.data, 0..) |val, i| {
                var pred = val;
                if (pred < epsilon) pred = epsilon;
                if (pred > 1.0 - epsilon) pred = 1.0 - epsilon;
                sum_loss += -target.data.data[i] * std.math.log(T, std.math.e, pred);
            }

            var batch_size: usize = 1;
            if (self.data.rank() > 1) {
                batch_size = self.data.shape[0];
            }
            const loss = sum_loss / @as(T, @floatFromInt(batch_size));

            var res_data = try NDArray(T).init(allocator, &.{1});
            res_data.data[0] = loss;

            var result = try Self.init(allocator, res_data, self.requires_grad);
            errdefer result.deinit(allocator);
            result.op = .CrossEntropy;
            try result.parents.append(allocator, self);
            try result.parents.append(allocator, target);
            return result;
        }
    };
}

test "autograd tensor" {
    const allocator = std.testing.allocator;

    var data1 = try NDArray(f64).init(allocator, &.{1});
    data1.data[0] = 2.0;
    var t1 = try Tensor(f64).init(allocator, data1, true);
    defer t1.deinit(allocator);

    var data2 = try NDArray(f64).init(allocator, &.{1});
    data2.data[0] = 3.0;
    var t2 = try Tensor(f64).init(allocator, data2, true);
    defer t2.deinit(allocator);

    var t3 = try t1.mul(allocator, t2);
    defer t3.deinit(allocator);

    try std.testing.expectEqual(t3.data.data[0], 6.0);

    try t3.backward(allocator);

    // dz/dx = y = 3
    try std.testing.expectEqual(t1.grad.?.data[0], 3.0);
    // dz/dy = x = 2
    try std.testing.expectEqual(t2.grad.?.data[0], 2.0);
}
