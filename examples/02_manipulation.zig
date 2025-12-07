const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== Manipulation Example ===\n\n", .{});

    var a = try num.NDArray(f32).arange(allocator, 0, 12, 1);
    defer a.deinit();

    // Reshape
    var b = try a.reshape(&.{ 3, 4 });
    defer b.deinit();
    std.debug.print("Reshaped to 3x4:\n", .{});
    for (0..b.shape[0]) |i| {
        for (0..b.shape[1]) |j| {
            std.debug.print("{d:.1} ", .{try b.get(&.{ i, j })});
        }
        std.debug.print("\n", .{});
    }

    // Transpose
    var c = try b.transpose();
    defer c.deinit();
    std.debug.print("\nTransposed to 4x3:\n", .{});
    for (0..c.shape[0]) |i| {
        for (0..c.shape[1]) |j| {
            std.debug.print("{d:.1} ", .{try c.get(&.{ i, j })});
        }
        std.debug.print("\n", .{});
    }

    // Flatten
    var d = try c.flatten();
    defer d.deinit();
    std.debug.print("\nFlattened size: {d}\n", .{d.size()});

    // vstack
    std.debug.print("\n--- vstack ---\n", .{});
    var v1 = try num.NDArray(f32).ones(allocator, &.{ 2, 3 });
    defer v1.deinit();
    var v2 = try num.NDArray(f32).full(allocator, &.{ 2, 3 }, 2.0);
    defer v2.deinit();

    // Create a slice of arrays for vstack
    const v_arrays = [_]num.NDArray(f32){ v1, v2 };
    var v_stacked = try num.manipulation.vstack(allocator, f32, &v_arrays);
    defer v_stacked.deinit();

    std.debug.print("Stacked shape: {any}\n", .{v_stacked.shape});
    std.debug.print("Value at [0,0]: {d}\n", .{try v_stacked.get(&.{ 0, 0 })});
    std.debug.print("Value at [2,0]: {d}\n", .{try v_stacked.get(&.{ 2, 0 })});

    // hstack
    std.debug.print("\n--- hstack ---\n", .{});
    var h_stacked = try num.manipulation.hstack(allocator, f32, &v_arrays);
    defer h_stacked.deinit();
    std.debug.print("H-Stacked shape: {any}\n", .{h_stacked.shape}); // Should be (2, 6)

    // Tile
    std.debug.print("\n--- Tile ---\n", .{});
    var t1 = try num.NDArray(f32).init(allocator, &.{2});
    defer t1.deinit();
    try t1.set(&.{0}, 1);
    try t1.set(&.{1}, 2);

    var tiled = try num.manipulation.tile(allocator, f32, t1, &.{ 2, 2 });
    defer tiled.deinit();
    std.debug.print("Tiled shape: {any}\n", .{tiled.shape});
    std.debug.print("Values:\n", .{});
    for (0..tiled.shape[0]) |i| {
        for (0..tiled.shape[1]) |j| {
            std.debug.print("{d} ", .{try tiled.get(&.{ i, j })});
        }
        std.debug.print("\n", .{});
    }

    // Repeat
    std.debug.print("\n--- Repeat ---\n", .{});
    var repeated = try num.manipulation.repeat(allocator, f32, t1, 3, 0);
    defer repeated.deinit();
    std.debug.print("Repeated shape: {any}\n", .{repeated.shape});
    std.debug.print("Values: ", .{});
    for (0..repeated.shape[0]) |i| {
        std.debug.print("{d} ", .{try repeated.get(&.{i})});
    }
    std.debug.print("\n", .{});

    // Flip
    std.debug.print("\n--- Flip ---\n", .{});
    var flipped = try num.manipulation.flip(allocator, f32, t1, 0);
    defer flipped.deinit();
    std.debug.print("Original: 1 2\n", .{});
    std.debug.print("Flipped: {d} {d}\n", .{ try flipped.get(&.{0}), try flipped.get(&.{1}) });

    // Roll
    std.debug.print("\n--- Roll ---\n", .{});
    var r1 = try num.NDArray(f32).arange(allocator, 0, 5, 1);
    defer r1.deinit();
    var rolled = try num.manipulation.roll(allocator, f32, r1, 2, 0);
    defer rolled.deinit();
    std.debug.print("Original: 0 1 2 3 4\n", .{});
    std.debug.print("Rolled (shift 2): ", .{});
    for (0..rolled.shape[0]) |i| {
        std.debug.print("{d} ", .{try rolled.get(&.{i})});
    }
    std.debug.print("\n", .{});
}
