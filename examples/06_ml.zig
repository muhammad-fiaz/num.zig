const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== ML Example ===\n\n", .{});

    // Simple Dense Layer Forward Pass
    var layer = try num.ml.layers.Dense.init(allocator, 3, 2, .XavierUniform);
    defer layer.deinit();

    var input = try num.NDArray(f32).init(allocator, &.{ 1, 3 });
    defer input.deinit();
    input.fill(0.5);

    var output = try layer.forward(allocator, &input);
    defer output.deinit();

    // Apply Activation
    var activated = try num.ml.activations.relu(allocator, &output);
    defer activated.deinit();

    std.debug.print("Dense Layer Output (ReLU):\n", .{});
    for (0..activated.shape[0]) |i| {
        for (0..activated.shape[1]) |j| {
            std.debug.print("{d:.4} ", .{try activated.get(&.{ i, j })});
        }
        std.debug.print("\n", .{});
    }

    // Loss Calculation
    var target = try num.NDArray(f32).init(allocator, &.{ 1, 2 });
    defer target.deinit();
    try target.set(&.{ 0, 0 }, 1.0);
    try target.set(&.{ 0, 1 }, 0.0);

    const loss = try num.ml.loss.mse(allocator, &target, &activated);
    std.debug.print("\nMSE Loss: {d:.4}\n", .{loss});
}
