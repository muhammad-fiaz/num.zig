const std = @import("std");
const Allocator = std.mem.Allocator;
const core = @import("../core.zig");
const NDArray = core.NDArray;

/// Simple Gradient Descent Optimizer.
pub fn GradientDescent(comptime T: type) type {
    return struct {
        const Self = @This();

        learning_rate: T,

        pub fn init(lr: T) Self {
            return Self{ .learning_rate = lr };
        }

        /// Updates parameters in-place: param = param - lr * grad
        pub fn step(self: Self, param: *NDArray(T), grad: NDArray(T)) !void {
            if (param.size() != grad.size()) return core.Error.ShapeMismatch;

            for (param.data, 0..) |*p, i| {
                p.* -= self.learning_rate * grad.data[i];
            }
        }
    };
}
