const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== ML Example ===\n\n", .{});

    // Simple Dense Layer Forward Pass
    var layer = try num.ml.layers.Dense.init(allocator, 3, 2, .XavierUniform);
    defer layer.deinit(allocator);

    var input_data = try num.NDArray(f32).init(allocator, &.{ 1, 3 });
    errdefer input_data.deinit(allocator);
    input_data.fill(0.5);

    const Tensor = num.autograd.Tensor;
    var input = try Tensor(f32).init(allocator, input_data, false);
    defer input.deinit(allocator);

    var output = try layer.forward(allocator, input);
    defer output.deinit(allocator);

    // Apply Activation
    var activated = try output.relu(allocator);
    defer activated.deinit(allocator);

    std.debug.print("Dense Layer Output (ReLU):\n", .{});
    const rows = activated.data.shape[0];
    const cols = activated.data.shape[1];
    for (0..rows) |i| {
        for (0..cols) |j| {
            std.debug.print("{d:.4} ", .{try activated.data.get(&.{ i, j })});
        }
        std.debug.print("\n", .{});
    }

    // Loss Calculation
    var target_data = try num.NDArray(f32).init(allocator, &.{ 1, 2 });
    errdefer target_data.deinit(allocator);
    try target_data.set(&.{ 0, 0 }, 1.0);
    try target_data.set(&.{ 0, 1 }, 0.0);

    var target = try Tensor(f32).init(allocator, target_data, false);
    defer target.deinit(allocator);

    var loss = try activated.mse_loss(allocator, target);
    defer loss.deinit(allocator);

    std.debug.print("\nMSE Loss: {d:.4}\n", .{loss.data.data[0]});
}
