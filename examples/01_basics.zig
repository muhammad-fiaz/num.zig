const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== Basics Example ===\n\n", .{});

    // 1. Creation
    var a = try num.NDArray(f32).zeros(allocator, &.{ 2, 3 });
    defer a.deinit();
    std.debug.print("Zeros (2x3):\n", .{});
    for (0..a.shape[0]) |i| {
        for (0..a.shape[1]) |j| {
            std.debug.print("{d:.2} ", .{try a.get(&.{ i, j })});
        }
        std.debug.print("\n", .{});
    }

    var b = try num.NDArray(f32).arange(allocator, 0, 10, 1);
    defer b.deinit();
    std.debug.print("\nArange (0-10):\n", .{});
    for (0..b.shape[0]) |i| {
        std.debug.print("{d:.2} ", .{try b.get(&.{i})});
    }
    std.debug.print("\n", .{});

    var c = try num.NDArray(f32).linspace(allocator, 0, 1, 5);
    defer c.deinit();
    std.debug.print("\nLinspace (0-1, 5 steps):\n", .{});
    for (0..c.shape[0]) |i| {
        std.debug.print("{d:.2} ", .{try c.get(&.{i})});
    }
    std.debug.print("\n", .{});

    var d = try num.NDArray(f32).ones(allocator, &.{ 2, 2 });
    defer d.deinit();
    std.debug.print("\nOnes (2x2):\n", .{});
    for (0..d.shape[0]) |i| {
        for (0..d.shape[1]) |j| {
            std.debug.print("{d:.2} ", .{try d.get(&.{ i, j })});
        }
        std.debug.print("\n", .{});
    }

    var e = try num.NDArray(f32).eye(allocator, 3);
    defer e.deinit();
    std.debug.print("\nIdentity (3x3):\n", .{});
    for (0..e.shape[0]) |i| {
        for (0..e.shape[1]) |j| {
            std.debug.print("{d:.2} ", .{try e.get(&.{ i, j })});
        }
        std.debug.print("\n", .{});
    }

    std.debug.print("\nArray Properties (Identity):\n", .{});
    std.debug.print("Rank: {d}\n", .{e.rank()});
    std.debug.print("Size: {d}\n", .{e.size()});
    std.debug.print("Shape: {any}\n", .{e.shape});

    // 2. IO
    try num.io.save(f32, b, "test_save.num");
    std.debug.print("\nSaved array to test_save.num\n", .{});

    var loaded = try num.io.load(allocator, f32, "test_save.num");
    defer loaded.deinit();
    std.debug.print("Loaded array size: {d}\n", .{loaded.size()});
}
