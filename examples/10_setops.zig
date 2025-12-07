const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== Set Operations Example ===\n\n", .{});

    var a = try num.NDArray(i32).init(allocator, &.{6});
    defer a.deinit();
    // [1, 2, 3, 2, 1, 4]
    try a.set(&.{0}, 1);
    try a.set(&.{1}, 2);
    try a.set(&.{2}, 3);
    try a.set(&.{3}, 2);
    try a.set(&.{4}, 1);
    try a.set(&.{5}, 4);

    // Unique
    var u = try num.setops.unique(allocator, i32, a);
    defer u.deinit();

    std.debug.print("Original: [1, 2, 3, 2, 1, 4]\n", .{});
    std.debug.print("Unique: ", .{});
    for (0..u.size()) |i| {
        std.debug.print("{d} ", .{try u.get(&.{i})});
    }
    std.debug.print("\n\n", .{});

    // In1d
    var b = try num.NDArray(i32).init(allocator, &.{3});
    defer b.deinit();
    // [2, 4, 5]
    try b.set(&.{0}, 2);
    try b.set(&.{1}, 4);
    try b.set(&.{2}, 5);

    var mask = try num.setops.in1d(allocator, i32, a, b);
    defer mask.deinit();

    std.debug.print("Test elements in [2, 4, 5]: ", .{});
    for (0..mask.size()) |i| {
        std.debug.print("{} ", .{try mask.get(&.{i})});
    }
    std.debug.print("\n", .{});
}
