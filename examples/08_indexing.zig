const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== Indexing & Sorting Example ===\n\n", .{});

    var a = try num.NDArray(i32).arange(allocator, 0, 10, 1);
    defer a.deinit(allocator);

    // Boolean Mask
    std.debug.print("--- Boolean Mask ---\n", .{});
    var mask = try num.NDArray(bool).init(allocator, a.shape);
    defer mask.deinit(allocator);

    // Create mask: elements > 5
    for (0..a.size()) |i| {
        const val = try a.get(&.{i});
        try mask.set(&.{i}, val > 5);
    }

    var filtered = try num.indexing.booleanMask(allocator, i32, a, mask);
    defer filtered.deinit(allocator);

    std.debug.print("Elements > 5: ", .{});
    for (0..filtered.size()) |i| {
        std.debug.print("{d} ", .{try filtered.get(&.{i})});
    }
    std.debug.print("\n", .{});

    // Where
    std.debug.print("\n--- Where ---\n", .{});
    var x = try num.NDArray(i32).full(allocator, a.shape, 100);
    defer x.deinit(allocator);
    var y = try num.NDArray(i32).full(allocator, a.shape, -1);
    defer y.deinit(allocator);

    var w = try num.indexing.where(allocator, i32, mask, x, y);
    defer w.deinit(allocator);

    std.debug.print("Where(mask, 100, -1): ", .{});
    for (0..w.size()) |i| {
        std.debug.print("{d} ", .{try w.get(&.{i})});
    }
    std.debug.print("\n", .{});

    // Nonzero
    std.debug.print("\n--- Nonzero ---\n", .{});
    var nz_indices = try num.sort.nonzero(allocator, i32, a);
    defer nz_indices.deinit(allocator);
    // a is 0..9, so 0 is at index 0. Non-zero are indices 1..9.
    std.debug.print("Non-zero count: {d}\n", .{nz_indices.shape[1]});

    // Slicing
    std.debug.print("\n--- Slicing ---\n", .{});
    // Slice a: start=2, end=8, step=2 -> {2, 4, 6}
    const slices = &[_]num.indexing.Slice{
        .{ .range = .{ .start = 2, .end = 8, .step = 2 } },
    };
    var view = try num.indexing.slice(allocator, i32, a, slices);
    defer view.deinit(allocator);

    std.debug.print("Slice [2:8:2]: ", .{});
    for (0..view.size()) |i| {
        std.debug.print("{d} ", .{try view.get(&.{i})});
    }
    std.debug.print("\n", .{});

    // Take
    std.debug.print("\n--- Take ---\n", .{});
    var indices = try num.NDArray(usize).init(allocator, &.{3});
    defer indices.deinit(allocator);
    try indices.set(&.{0}, 0);
    try indices.set(&.{1}, 9);
    try indices.set(&.{2}, 5);

    var taken = try num.indexing.take(allocator, i32, a, indices, 0);
    defer taken.deinit(allocator);

    std.debug.print("Take indices [0, 9, 5]: ", .{});
    for (0..taken.size()) |i| {
        std.debug.print("{d} ", .{try taken.get(&.{i})});
    }
    std.debug.print("\n", .{});
}
