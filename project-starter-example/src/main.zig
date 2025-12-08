const std = @import("std");
const num = @import("num");
const NDArray = num.NDArray;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n", .{});
    std.debug.print("=================================================================\n", .{});
    std.debug.print("           NUM.ZIG STARTER EXAMPLE - Numerical Computing Demo\n", .{});
    std.debug.print("=================================================================\n", .{});
    std.debug.print("\n", .{});

    // 1. Create a 2x3 matrix using arange (0, 1, 2, 3, 4, 5)
    // arange creates a 1D array, so we reshape it to 2x3
    var a = try NDArray(f32).arange(allocator, 0.0, 6.0, 1.0);
    defer a.deinit(allocator);
    try a.reshape(allocator, &.{ 2, 3 });

    std.debug.print("Matrix A (2x3):\n", .{});
    try printMatrix(a);

    // 2. Create a 3x2 matrix of ones
    var b = try NDArray(f32).ones(allocator, &.{ 3, 2 });
    defer b.deinit(allocator);

    std.debug.print("Matrix B (3x2 - all ones):\n", .{});
    try printMatrix(b);

    // 3. Matrix Multiplication (Dot Product)
    // Result will be (2x3) @ (3x2) = (2x2)
    var c = try num.linalg.matmul(f32, allocator, &a, &b);
    defer c.deinit(allocator);

    std.debug.print("Matrix C = A @ B (2x2):\n", .{});
    try printMatrix(c);

    // 4. Simple Reduction (Sum)
    const total_sum = try num.stats.sum(f32, allocator, &c);
    std.debug.print("Sum of all elements in C: {d:.2}\n", .{total_sum});
    std.debug.print("\n", .{});
}

fn printMatrix(matrix: NDArray(f32)) !void {
    const shape = matrix.shape;
    // Handle 1D case for robustness, though we use 2D here
    if (shape.len == 1) {
        std.debug.print("[ ", .{});
        for (0..shape[0]) |i| {
            const val = try matrix.get(&.{i});
            std.debug.print("{d:.1} ", .{val});
        }
        std.debug.print("]\n\n", .{});
        return;
    }

    const rows = shape[0];
    const cols = shape[1];

    for (0..rows) |i| {
        std.debug.print("[ ", .{});
        for (0..cols) |j| {
            const val = try matrix.get(&.{ i, j });
            std.debug.print("{d:.1} ", .{val});
        }
        std.debug.print("]\n", .{});
    }
    std.debug.print("\n", .{});
}
