//! Demonstrates delimited text I/O round-tripping.

const std = @import("std");
const num = @import("num");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    const vals = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    var a = try num.fromSlice(allocator, f64, .{ .data = &vals, .shape = &.{ 2, 3 } });
    defer a.deinit();

    // Save as CSV text.
    try num.io.savetxt(allocator, a, "matrix.csv", .{ .delimiter = "," });
    defer {
        var io_threaded: std.Io.Threaded = .init(allocator, .{});
        defer io_threaded.deinit();
        std.Io.Dir.cwd().deleteFile(io_threaded.io(), "matrix.csv") catch {};
    }

    var loaded = try num.io.loadtxt(allocator, "matrix.csv", .{ .delimiter = "," });
    defer loaded.deinit();

    std.debug.print("loaded shape={any} [0,0]={d:.1} [1,2]={d:.1}\n", .{
        loaded.shapeSlice(),
        try loaded.get(f64, &.{ 0, 0 }),
        try loaded.get(f64, &.{ 1, 2 }),
    });
}
