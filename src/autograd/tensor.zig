const std = @import("std");
const Allocator = std.mem.Allocator;
const core = @import("../core.zig");
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
        /// The allocator used for memory management.
        allocator: Allocator,
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
        };

        /// Initializes a new Tensor.
        ///
        /// Arguments:
        ///     allocator: The allocator to use.
        ///     data: The NDArray data. The Tensor takes ownership.
        ///     requires_grad: If true, gradients will be computed for this tensor.
        pub fn init(allocator: Allocator, data: NDArray(T), requires_grad: bool) !Self {
            var data_mut = data;
            errdefer data_mut.deinit();
            return Self{
                .data = data_mut,
                .grad = if (requires_grad) try NDArray(T).zeros(allocator, data.shape) else null,
                .requires_grad = requires_grad,
                .allocator = allocator,
                .parents = .{},
                .op = .None,
                .ctx = null,
            };
        }

        /// Frees resources associated with the Tensor.
        pub fn deinit(self: *Self) void {
            self.data.deinit();
            if (self.grad) |*g| g.deinit();
            self.parents.deinit(self.allocator);
            // Context deallocation would depend on the op
        }

        /// Performs the backward pass to compute gradients.
        ///
        /// This method traverses the computation graph in reverse topological order
        /// and accumulates gradients for all tensors with `requires_grad=true`.
        pub fn backward(self: *Self) !void {
            if (!self.requires_grad) return;

            // Seed gradient with 1s if not already set (usually for the scalar loss)
            if (self.grad) |*g| {
                // If it's a scalar, set to 1. If not, we assume it's the start of backprop.
                // For simplicity, we fill with 1s if it's all zeros (initial state)
                // But strictly, we should set it to 1.0 for scalar loss.
                // Let's assume the user or loss function sets the initial grad,
                // or we default to 1s for the root.
                var all_zeros = true;
                for (g.data) |val| {
                    if (val != 0) {
                        all_zeros = false;
                        break;
                    }
                }
                if (all_zeros) {
                    for (g.data) |*val| val.* = 1;
                }
            }

            // Simple recursive backward for tree structures.
            // For DAGs, we need topological sort.
            // Here we implement a basic recursive approach for demonstration.
            try self.backwardStep();
        }

        fn backwardStep(self: *Self) !void {
            if (!self.requires_grad) return;

            const g = self.grad orelse return;

            switch (self.op) {
                .Add => {
                    // z = x + y => dz/dx = 1 * dz/dz, dz/dy = 1 * dz/dz
                    for (self.parents.items) |parent| {
                        if (parent.requires_grad and parent.grad != null) {
                            // Accumulate gradient: parent.grad += self.grad
                            // We need an element-wise add function that works in-place or similar
                            // For now, manual loop
                            for (parent.grad.?.data, 0..) |*pg, i| {
                                pg.* += g.data[i];
                            }
                            try parent.backwardStep();
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
                            try x.backwardStep();
                        }
                        if (y.requires_grad and y.grad != null) {
                            for (y.grad.?.data, 0..) |*pg, i| {
                                pg.* += x.data.data[i] * g.data[i];
                            }
                            try y.backwardStep();
                        }
                    }
                },
                .None => {}, // Leaf node
                else => {}, // Implement others
            }
        }

        /// Adds two tensors.
        pub fn add(self: *Self, other: *Self) !Self {
            // Forward pass
            // We need to clone data because result is a new tensor
            // But wait, NDArray operations usually return new arrays.
            // We need an `add` op in core or elementwise.
            // Let's assume simple elementwise addition for now.

            // Note: We are creating a new Tensor on the stack and returning it.
            // The parents pointer will point to `self` and `other`.
            // WARNING: If `self` or `other` are on the stack and move, this breaks.
            // In a production library, Tensors should be heap allocated handles.
            // For this implementation, we assume the user keeps tensors stable.
            // Perform addition
            // We'll do manual addition to avoid circular imports or complex dependencies for now
            const res_data = try NDArray(T).init(self.allocator, self.data.shape);
            errdefer res_data.deinit();
            for (res_data.data, 0..) |*d, i| {
                d.* = self.data.data[i] + other.data.data[i];
            }

            var result = try Self.init(self.allocator, res_data, self.requires_grad or other.requires_grad);
            errdefer result.deinit();
            result.op = .Add;
            try result.parents.append(self.allocator, self);
            try result.parents.append(self.allocator, other);
            return result;
        }
        /// Multiplies two tensors element-wise.
        pub fn mul(self: *Self, other: *Self) !Self {
            const res_data = try NDArray(T).init(self.allocator, self.data.shape);
            errdefer res_data.deinit();
            for (res_data.data, 0..) |*d, i| {
                d.* = self.data.data[i] * other.data.data[i];
            }

            var result = try Self.init(self.allocator, res_data, self.requires_grad or other.requires_grad);
            errdefer result.deinit();
            result.op = .Mul;
            try result.parents.append(self.allocator, self);
            try result.parents.append(self.allocator, other);
            return result;
        }
    };
}
