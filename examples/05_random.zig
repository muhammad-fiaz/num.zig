const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== Random Example ===\n\n", .{});

    var rng = num.random.Random.init(12345);

    var u = try rng.uniform(allocator, &.{5});
    defer u.deinit(allocator);
    std.debug.print("Uniform [0, 1):\n", .{});
    for (0..u.shape[0]) |i| {
        std.debug.print("{d:.4} ", .{try u.get(&.{i})});
    }
    std.debug.print("\n\n", .{});

    var n = try rng.normal(allocator, &.{5}, 0.0, 1.0);
    defer n.deinit(allocator);
    std.debug.print("Normal (mean=0, std=1):\n", .{});
    for (0..n.shape[0]) |i| {
        std.debug.print("{d:.4} ", .{try n.get(&.{i})});
    }
    std.debug.print("\n\n", .{});

    var ints = try rng.randint(allocator, &.{5}, 0, 10);
    defer ints.deinit(allocator);
    std.debug.print("Randint [0, 10):\n", .{});
    for (0..ints.shape[0]) |i| {
        std.debug.print("{d} ", .{try ints.get(&.{i})});
    }
    std.debug.print("\n", .{});

    // Permutation
    std.debug.print("\n--- Permutation ---\n", .{});
    var perm = try rng.permutation(allocator, 10);
    defer perm.deinit(allocator);
    std.debug.print("Permutation of 10:\n", .{});
    for (0..perm.size()) |i| {
        std.debug.print("{d} ", .{try perm.get(&.{i})});
    }
    std.debug.print("\n", .{});

    // Choice
    std.debug.print("\n--- Choice ---\n", .{});
    var sample = try rng.choice(allocator, usize, perm, 3, false);
    defer sample.deinit(allocator);
    std.debug.print("Sample 3 from perm (no replace): ", .{});
    for (0..sample.size()) |i| {
        std.debug.print("{d} ", .{try sample.get(&.{i})});
    }
    std.debug.print("\n", .{});
}
