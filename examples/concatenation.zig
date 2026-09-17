//! Demonstrates array joining, stacking, splitting, tiling, and padding.

const std = @import("std");
const num = @import("num");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    const d1 = [_]f64{ 1.0, 2.0 };
    const d2 = [_]f64{ 3.0, 4.0 };
    var a = try num.fromSlice(allocator, f64, .{ .data = &d1, .shape = &.{2} });
    defer a.deinit();
    var b = try num.fromSlice(allocator, f64, .{ .data = &d2, .shape = &.{2} });
    defer b.deinit();

    // 1. Concatenate along axis 0
    const list = [_]num.Array{ a, b };
    var joined = try num.manip.concat(&list, .{ .axis = 0 });
    defer joined.deinit();
    std.debug.print("Concatenated length: {d}\n", .{joined.elementCount()});

    // 2. Stack along new dimension: (2,) + (2,) -> (2, 2)
    var stacked = try num.manip.stack(&list, .{ .axis = 0 });
    defer stacked.deinit();
    std.debug.print("Stacked shape: [{d}, {d}]\n", .{ stacked.shape_dims[0], stacked.shape_dims[1] });

    // 3. Tile array
    const reps = [_]usize{3};
    var tiled = try num.manip.tile(a, .{ .reps = &reps });
    defer tiled.deinit();
    std.debug.print("Tiled 3x length: {d}\n", .{tiled.elementCount()});

    // 4. Pad array with constant value
    const pad_widths = [_][2]usize{.{ 1, 1 }};
    var padded = try num.manip.pad(a, .{ .pad_width = &pad_widths, .constant_value = 0.0 });
    defer padded.deinit();
    std.debug.print("Padded length: {d}, pad[0]: {d:.1}, pad[1]: {d:.1}\n", .{
        padded.elementCount(),
        try padded.get(f64, &.{0}),
        try padded.get(f64, &.{1}),
    });
}
